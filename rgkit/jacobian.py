"""Velocity kinematics, Jacobians, singularity measures and statics (MFoR Chapter 5).

Everything here is derived step by step in the notebooks of
``MFoR/05-velocity-kinematics-and-statics``; this module keeps the same formulas in one
tested place so that later chapters (dynamics, trajectory generation, control) can import them.

Conventions
-----------
* A *velocity vector* (twist) is stacked **linear above angular**, ``[v; w]``, exactly as in
  Craig's ``^0 J`` and as MuJoCo's ``mj_jac*`` functions return it (``[jacp; jacr]``).
* A *force vector* (wrench) is ``[f; n]``: force above moment.
* ``J`` is the **geometric** Jacobian unless a name says ``analytical``.  Its angular part is the
  spatial angular velocity ``w`` defined by ``Rdot = [w]x R``.
* D-H tables are those of :mod:`rgkit.kinematics` (rows ``(alpha, a, d, theta, kind)``); the
  default is Craig's *modified* convention.
* Frame notation follows Craig: ``R_BA`` is ``^B_A R``, the orientation of ``{A}`` described in
  ``{B}``; ``T_AB`` is ``^A_B T``.
"""

from __future__ import annotations

import numpy as np

from .transforms import (rotx, roty, rotz, skew, unskew, make_T, rot_of, pos_of, inv_T, transl,
                         axis_angle_from_rot)
from .kinematics import fk_dh, dh_modified

__all__ = [
    # angular velocity
    "angular_velocity_space", "angular_velocity_body", "rotation_vector",
    # Jacobians
    "jacobian_geometric", "jacobian_numeric", "jacobian_position_numeric", "planar_jacobian",
    "jacobian_in_frame", "shift_reference_point", "velocity_transform", "wrench_transform",
    "propagate_velocity", "jacobian_by_propagation", "propagate_force", "joint_torques",
    # Euler-rate maps and analytical Jacobian
    "euler_rate_matrix", "euler_rate_matrix_body", "analytical_jacobian",
    # singularity and manipulability measures
    "singular_values", "manipulability", "condition_number", "isotropy", "min_singular_value",
    "ellipse_points", "ellipsoid_surface",
    # inverses and null space
    "pinv", "dls_pinv", "weighted_pinv", "nullspace_projector", "adaptive_damping",
    # MuJoCo helper and real-robot tables
    "mujoco_jacobian", "UR5E_DH", "UR5E_TOOL", "UR5E_BASE_IN_WORLD", "PANDA_DH", "PANDA_TOOL",
    "PANDA_BASE_IN_WORLD", "PUMA_LIKE_DH",
]


# ---------------------------------------------------------------------------
# Angular velocity of a rotating frame
# ---------------------------------------------------------------------------
def angular_velocity_space(R, Rdot) -> np.ndarray:
    """Angular velocity ``w`` with ``Rdot = [w]x R``, i.e. ``[w]x = Rdot R^T``, in the reference frame."""
    return unskew(np.asarray(Rdot, float) @ np.asarray(R, float).T)


def angular_velocity_body(R, Rdot) -> np.ndarray:
    """Angular velocity written in the *rotating* frame: ``[w_b]x = R^T Rdot`` (equals ``R^T w``)."""
    R = np.asarray(R, float)
    return unskew(R.T @ np.asarray(Rdot, float))


def rotation_vector(R) -> np.ndarray:
    """Rotation vector ``theta * k`` (angle times unit axis) of a rotation matrix."""
    axis, theta = axis_angle_from_rot(R)
    return axis * theta


# ---------------------------------------------------------------------------
# Geometric Jacobian of a D-H chain
# ---------------------------------------------------------------------------
def _axes_and_origins(table, q, convention="modified"):
    """Joint axes ``z_i`` and a point ``p_i`` on each axis, in the base frame."""
    Ts = fk_dh(table, q, convention)
    idx = range(1, len(table) + 1) if convention == "modified" else range(0, len(table))
    z = np.array([Ts[i][:3, 2] for i in idx])
    p = np.array([Ts[i][:3, 3] for i in idx])
    return Ts, z, p


def jacobian_geometric(table, q, convention: str = "modified", tool=None) -> np.ndarray:
    """Geometric Jacobian ``^0 J`` (6 x n) of the tool, expressed in the base frame {0}.

    Column ``i`` is ``[z_i x (p_n - p_i); z_i]`` for a revolute joint and ``[z_i; 0]`` for a
    prismatic joint, where ``z_i`` is the joint axis, ``p_i`` a point on it and ``p_n`` the tool
    origin (the origin of the last frame, moved by ``tool`` if given; ``tool`` is a fixed
    4 x 4 transform from the last D-H frame to the tool frame).
    """
    q = np.zeros(len(table)) if q is None else np.asarray(q, float)
    Ts, z, p = _axes_and_origins(table, q, convention)
    Tn = Ts[-1] if tool is None else Ts[-1] @ np.asarray(tool, float)
    pn = Tn[:3, 3]
    J = np.zeros((6, len(table)))
    for i, row in enumerate(table):
        if row[4] == "R":
            J[:3, i] = np.cross(z[i], pn - p[i])
            J[3:, i] = z[i]
        else:
            J[:3, i] = z[i]
    return J


def jacobian_numeric(fk, q, h: float = 1e-6) -> np.ndarray:
    """6 x n geometric Jacobian of any pose function ``fk(q) -> 4 x 4`` by central differences.

    The linear part is ``dp/dq_j``; the angular part is the spatial angular velocity, obtained from
    the rotation vector of ``R(q + h e_j) R(q - h e_j)^T`` divided by ``2h``.
    """
    q = np.asarray(q, float)
    J = np.zeros((6, q.size))
    for j in range(q.size):
        dq = np.zeros(q.size)
        dq[j] = h
        Tp, Tm = fk(q + dq), fk(q - dq)
        J[:3, j] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
        J[3:, j] = rotation_vector(Tp[:3, :3] @ Tm[:3, :3].T) / (2 * h)
    return J


def jacobian_position_numeric(f, q, h: float = 1e-6) -> np.ndarray:
    """Jacobian ``df/dq`` (m x n) of any vector function ``f(q)`` by central differences."""
    q = np.asarray(q, float)
    cols = []
    for j in range(q.size):
        dq = np.zeros(q.size)
        dq[j] = h
        cols.append((np.asarray(f(q + dq), float) - np.asarray(f(q - dq), float)) / (2 * h))
    return np.array(cols).T


def planar_jacobian(lengths, q, orientation: bool = False) -> np.ndarray:
    """Closed-form Jacobian of the tip of a planar n-link revolute arm (2 x n, or 3 x n with the
    rows ``[x_dot; y_dot; phi_dot]`` when ``orientation=True``)."""
    L, q = np.asarray(lengths, float), np.asarray(q, float)
    phi = np.cumsum(q)
    n = len(L)
    J = np.zeros((3 if orientation else 2, n))
    for j in range(n):
        J[0, j] = -np.sum(L[j:] * np.sin(phi[j:]))
        J[1, j] = np.sum(L[j:] * np.cos(phi[j:]))
        if orientation:
            J[2, j] = 1.0
    return J


# ---------------------------------------------------------------------------
# Changing frame, changing reference point
# ---------------------------------------------------------------------------
def jacobian_in_frame(J, R_BA) -> np.ndarray:
    """Re-express a Jacobian given in {A} in frame {B}: ``^B J = diag(^B_A R, ^B_A R) ^A J``."""
    R = np.asarray(R_BA, float)
    Z = np.zeros((3, 3))
    return np.block([[R, Z], [Z, R]]) @ np.asarray(J, float)


def shift_reference_point(J, r) -> np.ndarray:
    """Jacobian of the point ``P' = P + r`` rigidly attached to the same body as ``P``.

    ``r`` must be written in the frame in which ``J`` is expressed. Since ``v' = v + w x r``,
    ``J' = [[I, -[r]x], [0, I]] J``.
    """
    X = np.eye(6)
    X[:3, 3:] = -skew(r)
    return X @ np.asarray(J, float)


def velocity_transform(T_AB) -> np.ndarray:
    """6 x 6 matrix mapping ``[^B v; ^B w]`` to ``[^A v; ^A w]`` for a rigid body carrying {A} and {B}.

    ``v`` is the velocity of the body point at the origin of the respective frame, both vectors
    are written in their own frame, and ``T_AB = ^A_B T`` is the (constant) pose of {B} in {A}:

        X = [[R, [p]x R], [0, R]].
    """
    R, p = rot_of(T_AB), pos_of(T_AB)
    X = np.zeros((6, 6))
    X[:3, :3] = R
    X[:3, 3:] = skew(p) @ R
    X[3:, 3:] = R
    return X


def wrench_transform(T_AB) -> np.ndarray:
    """6 x 6 matrix mapping ``[^B f; ^B n]`` to the equivalent ``[^A f; ^A n]`` (moment about origin of {A}).

    ``W = [[R, 0], [[p]x R, R]] = X^{-T}``: forces and velocities are dual, so power is invariant.
    """
    R, p = rot_of(T_AB), pos_of(T_AB)
    W = np.zeros((6, 6))
    W[:3, :3] = R
    W[3:, :3] = skew(p) @ R
    W[3:, 3:] = R
    return W


# ---------------------------------------------------------------------------
# Craig's recursive velocity and force propagation (modified D-H convention)
# ---------------------------------------------------------------------------
def propagate_velocity(table, q, qd, tool=None):
    """Outward iteration of Craig Section 5.5.

    Returns ``(omegas, vs)``: lists of length ``n + 1`` with the angular velocity and the linear
    velocity of the origin of every link frame {0}..{n}, each written in its **own** frame
    (frame {0} is at rest).  If ``tool`` is given a final entry for the tool frame is appended.
    Joint ``i`` is revolute if ``table[i][4] == 'R'`` and prismatic if ``'P'``.
    """
    q, qd = np.asarray(q, float), np.asarray(qd, float)
    w, v = np.zeros(3), np.zeros(3)
    ws, vs = [w], [v]
    zhat = np.array([0.0, 0.0, 1.0])
    for (alpha, a, d, theta, kind), qi, qdi in zip(table, q, qd):
        T = dh_modified(alpha, a, d + (qi if kind == "P" else 0.0), theta + (qi if kind == "R" else 0.0))
        R_next_prev, P = rot_of(T).T, pos_of(T)          # ^{i+1}_i R and ^i P_{i+1}
        v = R_next_prev @ (v + np.cross(w, P))
        w = R_next_prev @ w
        if kind == "R":
            w = w + qdi * zhat
        else:
            v = v + qdi * zhat
        ws.append(w)
        vs.append(v)
    if tool is not None:
        Tt = np.asarray(tool, float)
        Rt, Pt = rot_of(Tt).T, pos_of(Tt)
        vs.append(Rt @ (vs[-1] + np.cross(ws[-1], Pt)))
        ws.append(Rt @ ws[-1])
    return ws, vs


def jacobian_by_propagation(table, q, tool=None) -> np.ndarray:
    """``^n J`` (Jacobian in the *last* frame, or the tool frame) by running the velocity iteration
    once per joint with unit joint rate (Craig Section 5.6)."""
    n = len(table)
    J = np.zeros((6, n))
    for j in range(n):
        qd = np.zeros(n)
        qd[j] = 1.0
        ws, vs = propagate_velocity(table, q, qd, tool)
        J[:3, j], J[3:, j] = vs[-1], ws[-1]
    return J


def propagate_force(table, q, f_tool, n_tool=None, tool=None):
    """Inward iteration of Craig Section 5.8 for the static torques.

    ``f_tool`` and ``n_tool`` are the force and moment that the tool exerts on the environment,
    written in the tool frame (the last D-H frame if ``tool`` is ``None``).  Returns
    ``(tau, fs, ns)`` with the joint torques/forces and the force and moment that link ``i - 1``
    exerts on link ``i``, in frame {i}.
    """
    q = np.asarray(q, float)
    n = len(table)
    f = np.asarray(f_tool, float).copy()
    m = np.zeros(3) if n_tool is None else np.asarray(n_tool, float).copy()
    if tool is not None:                                  # bring the load from the tool frame to frame {n}
        Rt, Pt = rot_of(tool), pos_of(tool)
        f = Rt @ f
        m = Rt @ m + np.cross(Pt, f)
    Ts = [None] + [dh_modified(al, a, d + (qi if k == "P" else 0.0), th + (qi if k == "R" else 0.0))
                   for (al, a, d, th, k), qi in zip(table, q)]
    tau = np.zeros(n)
    fs, ns = [None] * (n + 1), [None] * (n + 1)
    fs[n], ns[n] = f, m
    for i in range(n, 0, -1):
        tau[i - 1] = ns[i][2] if table[i - 1][4] == "R" else fs[i][2]
        if i > 1:
            Rm, Pm = rot_of(Ts[i]), pos_of(Ts[i])         # ^{i-1}_i R and ^{i-1}P_i
            fs[i - 1] = Rm @ fs[i]
            ns[i - 1] = Rm @ ns[i] + np.cross(Pm, fs[i - 1])
    return tau, fs, ns


def joint_torques(J, F) -> np.ndarray:
    """Static joint torques that balance a tool wrench ``F`` exerted on the environment: ``tau = J^T F``."""
    return np.asarray(J, float).T @ np.asarray(F, float)


# ---------------------------------------------------------------------------
# Euler-angle rates and the analytical Jacobian
# ---------------------------------------------------------------------------
_ELEM = {"x": rotx, "y": roty, "z": rotz}
_E = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


def euler_rate_matrix(angles, order: str = "zyx", frame: str = "current") -> np.ndarray:
    """Matrix ``E`` with ``w = E @ angle_rates`` (``w`` in the *reference* frame).

    Uses the conventions of :func:`rgkit.transforms.euler_to_rot`: ``order`` lists the axes in the
    order the rotations are applied and ``frame`` is ``'current'`` (Euler angles) or ``'fixed'``
    (fixed angles, e.g. Craig's X-Y-Z roll-pitch-yaw with ``order='xyz', frame='fixed'``).
    ``angles`` and the columns of ``E`` are in the order of ``order``.
    """
    o, (a1, a2, a3) = order.lower(), np.asarray(angles, float)
    R1, R2, R3 = (_ELEM[c](a) for c, a in zip(o, (a1, a2, a3)))
    if frame == "current":
        cols = [_E[o[0]], R1 @ _E[o[1]], R1 @ R2 @ _E[o[2]]]
    else:
        cols = [R3 @ R2 @ _E[o[0]], R3 @ _E[o[1]], _E[o[2]]]
    return np.column_stack(cols)


def euler_rate_matrix_body(angles, order: str = "zyx", frame: str = "current") -> np.ndarray:
    """Like :func:`euler_rate_matrix` but ``w`` is written in the *rotating* frame: ``E_b = R^T E``."""
    from .transforms import euler_to_rot
    return euler_to_rot(angles, order, frame).T @ euler_rate_matrix(angles, order, frame)


def analytical_jacobian(J_base, angles, order: str = "xyz", frame: str = "fixed") -> np.ndarray:
    """Analytical Jacobian ``J_a = diag(I, E^{-1}) J`` mapping joint rates to ``[p_dot; angle_rates]``.

    ``J_base`` must be the geometric Jacobian expressed in the base frame. The matrix ``E`` is
    singular at the representation singularities of the chosen angle set, where this function
    raises ``numpy.linalg.LinAlgError`` or returns very large entries.
    """
    E = euler_rate_matrix(angles, order, frame)
    J = np.asarray(J_base, float)
    return np.vstack([J[:3], np.linalg.solve(E, J[3:])])


# ---------------------------------------------------------------------------
# Singularity and manipulability measures
# ---------------------------------------------------------------------------
def singular_values(J) -> np.ndarray:
    """Singular values of ``J`` in decreasing order."""
    return np.linalg.svd(np.asarray(J, float), compute_uv=False)


def manipulability(J) -> float:
    """Yoshikawa's measure ``w = sqrt(det(J J^T))`` (the product of the singular values); 0 if ``m > n``."""
    J = np.asarray(J, float)
    if J.shape[0] > J.shape[1]:
        return 0.0
    return float(np.prod(singular_values(J)))


def min_singular_value(J) -> float:
    """Smallest singular value of the ``m x n`` matrix ``J`` (zero when it has deficient row rank)."""
    J = np.asarray(J, float)
    s = singular_values(J)
    return float(s[-1]) if J.shape[0] <= J.shape[1] else 0.0


def condition_number(J) -> float:
    """``sigma_max / sigma_min``; ``inf`` at a singularity."""
    s = singular_values(J)
    return float(s[0] / s[-1]) if s[-1] > 1e-14 * max(s[0], 1e-300) else float("inf")


def isotropy(J) -> float:
    """Inverse condition number ``sigma_min / sigma_max`` in [0, 1]; 1 means an isotropic (round) ellipsoid."""
    s = singular_values(J)
    return float(s[-1] / s[0]) if s[0] > 0 else 0.0


def ellipse_points(J, kind: str = "velocity", n: int = 120, center=(0.0, 0.0), scale: float = 1.0):
    """Points of the velocity or force ellipse of a 2 x m Jacobian, as a ``(2, n)`` array.

    ``kind='velocity'``: the image of the unit ball ``||qdot|| = 1``, semi-axes ``sigma_i u_i``.
    ``kind='force'``: the tool forces reachable with ``||tau|| = 1``, semi-axes ``u_i / sigma_i``.
    """
    U, s, _ = np.linalg.svd(np.asarray(J, float))
    axes = s if kind == "velocity" else 1.0 / np.maximum(s, 1e-12)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    pts = U[:, :2] @ (axes[:2, None] * np.vstack([np.cos(t), np.sin(t)]))
    return np.asarray(center, float).reshape(2, 1) + scale * pts


def ellipsoid_surface(J, kind: str = "velocity", n: int = 28, center=(0.0, 0.0, 0.0), scale: float = 1.0):
    """Surface grids ``(X, Y, Z)`` of the 3-D velocity (or force) ellipsoid of a 3 x m Jacobian."""
    U, s, _ = np.linalg.svd(np.asarray(J, float))
    axes = s[:3] if kind == "velocity" else 1.0 / np.maximum(s[:3], 1e-12)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2 + 1))
    sph = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])
    pts = np.tensordot(U[:, :3] * axes, sph, axes=(1, 0)) * scale + np.asarray(center, float).reshape(3, 1, 1)
    return pts[0], pts[1], pts[2]


# ---------------------------------------------------------------------------
# Generalised inverses and the null space
# ---------------------------------------------------------------------------
def pinv(J, rcond: float = 1e-10) -> np.ndarray:
    """Moore-Penrose pseudoinverse by the SVD; singular values below ``rcond * sigma_max`` are dropped."""
    return np.linalg.pinv(np.asarray(J, float), rcond=rcond)


def dls_pinv(J, lam: float) -> np.ndarray:
    """Damped least-squares inverse ``J^T (J J^T + lam^2 I)^{-1}``, evaluated by the SVD.

    Each singular value ``s`` is inverted as ``s / (s^2 + lam^2)`` instead of ``1 / s``, so the
    result stays bounded (by ``1 / (2 lam)``) at a singularity.
    """
    U, s, Vt = np.linalg.svd(np.asarray(J, float), full_matrices=False)
    return Vt.T @ np.diag(s / (s ** 2 + lam ** 2)) @ U.T


def weighted_pinv(J, W) -> np.ndarray:
    """Right inverse minimising ``qdot^T W qdot``: ``W^{-1} J^T (J W^{-1} J^T)^{-1}`` (``W`` symmetric > 0)."""
    J, W = np.asarray(J, float), np.asarray(W, float)
    Wi = np.linalg.inv(W)
    return Wi @ J.T @ np.linalg.inv(J @ Wi @ J.T)


def nullspace_projector(J, rcond: float = 1e-10) -> np.ndarray:
    """``I - J^+ J``: orthogonal projector onto the null space of ``J`` (the self-motion directions)."""
    J = np.asarray(J, float)
    return np.eye(J.shape[1]) - pinv(J, rcond) @ J


def adaptive_damping(J, eps: float = 0.05, lam_max: float = 0.1) -> float:
    """Damping factor that is zero away from singularities and grows smoothly to ``lam_max``:

    ``lam^2 = (1 - (s_min / eps)^2) lam_max^2`` if ``s_min < eps`` else ``0``.
    """
    s_min = singular_values(J)[-1]
    return 0.0 if s_min >= eps else float(lam_max * np.sqrt(1.0 - (s_min / eps) ** 2))


# ---------------------------------------------------------------------------
# MuJoCo helper
# ---------------------------------------------------------------------------
def mujoco_jacobian(model, data, site: str | None = None, body: str | None = None, point=None) -> np.ndarray:
    """6 x nv Jacobian ``[jacp; jacr]`` from MuJoCo, expressed in the **world** frame.

    Give a ``site`` name, or a ``body`` name (its frame origin), or a ``body`` together with a
    world-frame ``point`` that is fixed to that body. The call refreshes kinematics but does not
    change ``qpos``.
    """
    import mujoco

    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    if site is not None:
        mujoco.mj_jacSite(model, data, jacp, jacr, model.site(site).id)
    elif point is not None:
        mujoco.mj_jac(model, data, jacp, jacr, np.asarray(point, float), model.body(body).id)
    else:
        mujoco.mj_jacBody(model, data, jacp, jacr, model.body(body).id)
    return np.vstack([jacp, jacr])


# ---------------------------------------------------------------------------
# D-H tables (Craig, modified) for the MuJoCo Menagerie models used in the notebooks
# ---------------------------------------------------------------------------
_pi2 = np.pi / 2

#: Universal Robots UR5e as modelled in MuJoCo Menagerie (dimensions read from the MJCF file),
#: described by Craig's modified D-H parameters relative to the model's ``base`` body frame.
UR5E_DH = [(0.0, 0.0, 0.163, 0.0, "R"),
           (-_pi2, 0.0, 0.138, 0.0, "R"),
           (0.0, 0.425, -0.131, 0.0, "R"),
           (0.0, 0.392, 0.127, 0.0, "R"),
           (-_pi2, 0.0, 0.100, 0.0, "R"),
           (_pi2, 0.0, 0.0, 0.0, "R")]
#: Fixed flange-to-tool transform of the model's ``attachment_site``.
UR5E_TOOL = transl(0, 0, 0.1) @ make_T(rotz(np.pi))
#: Pose of the model's ``base`` body in the world frame (rotated 180 degrees about Z).
UR5E_BASE_IN_WORLD = make_T(rotz(np.pi))

#: Franka Emika Panda: Franka's published (Craig-modified) D-H parameters, which coincide with
#: the link frames of the Menagerie MJCF model.
PANDA_DH = [(0.0, 0.0, 0.333, 0.0, "R"),
            (-_pi2, 0.0, 0.0, 0.0, "R"),
            (_pi2, 0.0, 0.316, 0.0, "R"),
            (_pi2, 0.0825, 0.0, 0.0, "R"),
            (-_pi2, -0.0825, 0.384, 0.0, "R"),
            (_pi2, 0.0, 0.0, 0.0, "R"),
            (_pi2, 0.088, 0.0, 0.0, "R")]
#: Flange (0.107 m along z7), the hand's frame (turned -45 degrees about z) and a tool point
#: 0.1034 m beyond the hand origin.
PANDA_TOOL = transl(0, 0, 0.107) @ make_T(rotz(-np.pi / 4)) @ transl(0, 0, 0.1034)
PANDA_BASE_IN_WORLD = np.eye(4)

#: A PUMA-type six-axis arm (spherical wrist), after Craig's convention; dimensions in metres
#: are rounded, illustrative values of the classic PUMA 560 geometry.
PUMA_LIKE_DH = [(0.0, 0.0, 0.0, 0.0, "R"),
                (-_pi2, 0.0, 0.0, 0.0, "R"),
                (0.0, 0.432, 0.150, 0.0, "R"),
                (-_pi2, -0.020, 0.432, 0.0, "R"),
                (_pi2, 0.0, 0.0, 0.0, "R"),
                (-_pi2, 0.0, 0.0, 0.0, "R")]

"""Tools for the Chapter 8 case studies (collaborative arms, industrial arms, humanoids, safety).

Chapter 8 applies the material of Chapters 2-7 end to end on real robot models.  Because it is
meant to be read on its own, the small pieces it needs -- a geometric Jacobian, a damped
least-squares inverse-kinematics solver, a rule that *reads a Denavit-Hartenberg table off a
robot model*, a product-of-exponentials forward kinematics, the effective mass of a manipulator
along a direction, a support-polygon test and a rest-to-rest trajectory -- are implemented here
in one tested place and kept short enough to be read.

Conventions (as in the rest of the repository)
----------------------------------------------
* Craig's **modified** D-H convention: ``i-1_i T = Rx(alpha_{i-1}) Dx(a_{i-1}) Rz(theta_i) Dz(d_i)``;
  a table is a list of rows ``(alpha, a, d, theta, kind)`` as in :mod:`rgkit.kinematics`.
* Twists and Jacobians are ordered **linear part first, then angular part**
  ``[v; omega]`` and expressed in the base (world) frame unless stated otherwise.
* Angles are radians, lengths metres.
"""

from __future__ import annotations

import os
from typing import Callable, NamedTuple

import numpy as np

from .transforms import axis_angle_from_rot, make_T, skew
from .kinematics import fk_dh

__all__ = [
    "pose_error", "jacobian_dh", "IKResult", "ik_dls", "sigma_min", "manipulability",
    "null_vector", "self_motion", "swivel_angle",
    "joint_axes", "chain_joint_ids", "DHFit", "dh_from_axes", "screw_axes", "exp_screw", "poe_fk",
    "mass_matrix", "site_jacobian", "point_jacobian", "operational_inertia", "effective_mass",
    "support_polygon", "polygon_margin", "quintic", "RestToRestPath", "save_gif",
]


# ---------------------------------------------------------------------------------------------
# Pose error, geometric Jacobian and damped least-squares inverse kinematics
# ---------------------------------------------------------------------------------------------
def pose_error(T_cur, T_des) -> np.ndarray:
    """Six-vector ``[p_des - p_cur ; k theta]`` that carries ``T_cur`` to ``T_des``.

    The rotational part is the angle-axis vector of ``R_des R_cur^T`` (a rotation about a *fixed*
    axis of the base frame), the natural error for a Jacobian expressed in the base frame.
    """
    T_cur, T_des = np.asarray(T_cur, float), np.asarray(T_des, float)
    dp = T_des[:3, 3] - T_cur[:3, 3]
    axis, ang = axis_angle_from_rot(T_des[:3, :3] @ T_cur[:3, :3].T)
    return np.concatenate([dp, axis * ang])


def jacobian_dh(table, q, tool=None, base=None) -> np.ndarray:
    """Geometric Jacobian ``J(q)`` (6 x n) of a modified-D-H chain, expressed in the base frame.

    Column ``i`` is ``[z_i x (p_e - p_i) ; z_i]`` for a revolute joint and ``[z_i ; 0]`` for a
    prismatic joint, where ``z_i`` and ``p_i`` are the axis and origin of frame ``{i}`` and
    ``p_e`` is the origin of the tool frame (``tool`` is a fixed flange-to-tool transform; ``base``
    is a fixed world-to-frame-{0} transform).  Because Craig's frame ``{i}`` has its Z axis on joint
    ``i``, the columns come straight from the cumulative transforms of :func:`rgkit.kinematics.fk_dh`.
    """
    Ts = fk_dh(table, q)
    B = np.eye(4) if base is None else np.asarray(base, float)
    Ts = [B @ T for T in Ts]
    T_end = Ts[-1] if tool is None else Ts[-1] @ np.asarray(tool, float)
    p_e = T_end[:3, 3]
    n = len(table)
    J = np.zeros((6, n))
    for i in range(n):
        z, p = Ts[i + 1][:3, 2], Ts[i + 1][:3, 3]
        if table[i][4] == "R":
            J[:3, i], J[3:, i] = np.cross(z, p_e - p), z
        else:
            J[:3, i] = z
    return J


class IKResult(NamedTuple):
    q: np.ndarray
    converged: bool
    iterations: int
    errors: np.ndarray        # |e| after every iteration (pose-error norm, metres/radians mixed)


def ik_dls(fk: Callable, jac: Callable, T_des, q0, lam: float = 0.05, tol_pos: float = 1e-6,
           tol_rot: float = 1e-6, max_iter: int = 200, step_max: float = 0.5,
           limits=None) -> IKResult:
    """Numerical inverse kinematics by damped least squares (Levenberg-Marquardt type).

    Each iteration solves ``dq = J^T (J J^T + lam^2 I)^-1 e`` for the pose error ``e`` and clips the
    step to ``step_max`` radians.  ``fk(q) -> 4x4`` and ``jac(q) -> 6 x n`` are supplied by the
    caller, so the same solver serves a D-H model and a MuJoCo model alike.  ``limits`` is an optional
    ``(lo, hi)`` pair of joint-limit arrays used only to clip the iterate.
    """
    q = np.array(q0, float)
    errs = []
    for k in range(max_iter + 1):
        e = pose_error(fk(q), T_des)
        errs.append(np.linalg.norm(e))
        if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot:
            return IKResult(q, True, k, np.array(errs))
        if k == max_iter:
            break
        J = jac(q)
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(J.shape[0]), e)
        nrm = np.linalg.norm(dq)
        if nrm > step_max:
            dq *= step_max / nrm
        q = q + dq
        if limits is not None:
            q = np.clip(q, limits[0], limits[1])
    return IKResult(q, False, max_iter, np.array(errs))


def sigma_min(J) -> float:
    """Smallest singular value of ``J`` -- the distance of the configuration from a singularity."""
    return float(np.linalg.svd(np.asarray(J, float), compute_uv=False)[-1])


def manipulability(J) -> float:
    """Yoshikawa's manipulability measure ``sqrt(det(J J^T))``."""
    J = np.asarray(J, float)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def null_vector(J) -> np.ndarray:
    """Unit vector spanning the null space of a ``(n-1) x n``-rank Jacobian (the last right-singular vector).

    For a 7-joint arm with a 6 x 7 Jacobian of rank 6 this is the direction of **self-motion**: joint
    velocities along it leave the tool pose unchanged.  The sign is fixed so that the largest component is
    positive; :func:`self_motion` keeps the sign continuous along a path.
    """
    _, _, Vt = np.linalg.svd(np.atleast_2d(J))
    v = Vt[-1]
    return v if v[np.argmax(np.abs(v))] > 0 else -v


def self_motion(fk: Callable, jac: Callable, q0, n_steps: int = 60, ds: float = 0.02):
    """Follow the null-space direction from ``q0`` with the tool pose held fixed (returns ``(n_steps+1, n)``).

    Each step moves ``ds`` radians (joint-space arc length) along :func:`null_vector` (continuous in sign) and
    then removes the second-order drift with a few damped-least-squares iterations back to the starting pose,
    so the tool pose stays within about ``1e-10`` of its initial value.  A negative ``ds`` walks the other way.
    """
    q = np.array(q0, float)
    T0 = fk(q)
    path, prev = [q.copy()], None
    for _ in range(n_steps):
        n = null_vector(jac(q))
        if prev is not None and n @ prev < 0:
            n = -n
        if prev is None and ds < 0:
            n = -n
        prev = n
        q = q + abs(ds) * n
        for _ in range(5):                                  # drift correction
            e = pose_error(fk(q), T0)
            if np.linalg.norm(e) < 1e-12:
                break
            J = jac(q)
            q = q + J.T @ np.linalg.solve(J @ J.T + 1e-10 * np.eye(J.shape[0]), e)
        path.append(q.copy())
    return np.array(path)


def swivel_angle(p_shoulder, p_elbow, p_wrist, up=(0.0, 0.0, 1.0)):
    """Elbow **swivel angle** about the shoulder-wrist line, the usual parameter of a 7-joint arm's self-motion.

    Returns ``(phi, radius, d_sw)``: the signed angle (radians) of the elbow's offset from the shoulder-wrist
    line, measured from the projection of ``up`` and about the direction shoulder -> wrist; the radius of the
    circle the elbow would describe; and the shoulder-wrist distance.
    """
    s, e, w = (np.asarray(x, float) for x in (p_shoulder, p_elbow, p_wrist))
    u = (w - s) / np.linalg.norm(w - s)
    ev = (e - s) - ((e - s) @ u) * u
    ref = np.asarray(up, float)
    ref = ref - (ref @ u) * u
    ref /= np.linalg.norm(ref)
    phi = float(np.arctan2(ev @ np.cross(u, ref), ev @ ref))
    return phi, float(np.linalg.norm(ev)), float(np.linalg.norm(w - s))


# ---------------------------------------------------------------------------------------------
# Reading a kinematic description off a MuJoCo model
# ---------------------------------------------------------------------------------------------
def chain_joint_ids(model, end_body) -> list:
    """Ids of the hinge/slide joints on the path from the world to ``end_body`` (root first).

    ``end_body`` is a body name or id.  Free joints are skipped, so the result is the list of joints
    of the *serial chain* that ends in that body.
    """
    import mujoco

    b = end_body if isinstance(end_body, (int, np.integer)) else mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, end_body)
    path = []
    while b > 0:
        js = [j for j in range(model.njnt) if model.jnt_bodyid[j] == b
              and model.jnt_type[j] in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
        path.append(js)
        b = model.body_parentid[b]
    return [j for js in reversed(path) for j in js]


def joint_axes(model, data, joint_ids):
    """World-frame joint axes ``omega_i`` (unit) and points ``p_i`` on them, at the current ``data.qpos``.

    ``data`` must be up to date (call ``mujoco.mj_forward`` first).  Returns two ``(n, 3)`` arrays.
    """
    ids = list(joint_ids)
    return data.xaxis[ids].copy(), data.xanchor[ids].copy()


class DHFit(NamedTuple):
    table: list               # modified D-H rows (alpha, a, d, theta_offset, "R"), one per joint
    T_base_0: np.ndarray      # fixed transform from the model's base frame to D-H frame {0}


def _line_feet(p1, d1, p2, d2):
    """Feet of the common perpendicular of two non-parallel lines ``p + t d`` (3-vectors)."""
    n = np.cross(d1, d2)
    A = np.column_stack([d1, -d2, n])
    s, t, _ = np.linalg.solve(A, p2 - p1)
    return p1 + s * d1, p2 + t * d2


def dh_from_axes(omegas, points, tol: float = 1e-9) -> DHFit:
    """Read a **modified D-H table** off a chain of revolute joint axes given at the zero configuration.

    The inputs are the unit axes ``omega_i`` and one point ``p_i`` on each axis (``i = 1..n``), all in
    the base frame, exactly what :func:`joint_axes` returns for a MuJoCo model at ``qpos = 0``.  Craig's
    frame-assignment recipe is applied literally:

    * ``Z_i`` is the joint axis; ``X_i`` lies on the common normal of axes ``i`` and ``i+1`` pointing
      from axis ``i`` to axis ``i+1`` (``a_i`` is its length).  If the axes intersect, ``X_i = Z_i x Z_{i+1}``;
      if they are parallel the origin is placed so that ``d_i = 0`` and the normal is any perpendicular.
    * the origin ``O_i`` is where that normal leaves axis ``i``; ``alpha_i`` turns ``Z_i`` into ``Z_{i+1}``
      about ``X_i``; ``d_i`` and the offset of ``theta_i`` follow from the positions of the previous
      normal's foot and the angle between consecutive ``X`` axes;
    * frame ``{0}`` coincides with ``{1}`` at zero, and ``X_n = X_{n-1}`` so the last offset is zero.

    The table is **one** valid description of the arm, not *the* description: other choices of the free
    signs and parallel-axis origins give other tables for the same robot.  The returned ``T_base_0``
    and the fixed tool transform (found by the caller, see the notebook) absorb the frame offsets, so
    ``T_base_0 @ fk_dh(table, q)[-1] @ T_n_tool`` reproduces the robot for every ``q``.
    """
    Z = np.asarray(omegas, float)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    P = np.asarray(points, float)
    n = len(Z)
    X = [None] * n            # X_i (0-indexed joint i)
    O = [None] * n            # origins of the frames
    F = [None] * n            # foot of the previous normal on axis i
    alpha, a = np.zeros(n), np.zeros(n)
    for i in range(n - 1):
        cross = np.cross(Z[i], Z[i + 1])
        if np.linalg.norm(cross) > tol:                       # intersecting or skew axes
            Pi, Qi = _line_feet(P[i], Z[i], P[i + 1], Z[i + 1])
            a[i] = np.linalg.norm(Qi - Pi)
            X[i] = (Qi - Pi) / a[i] if a[i] > tol else cross / np.linalg.norm(cross)
            O[i], F[i + 1] = Pi, Qi
        else:                                                 # parallel (or anti-parallel) axes
            v = P[i + 1] - P[i]
            v = v - (v @ Z[i]) * Z[i]
            if np.linalg.norm(v) > tol:
                a[i] = np.linalg.norm(v)
                X[i] = v / a[i]
            else:                                             # collinear axes: any normal will do
                if i > 0:
                    X[i] = X[i - 1]
                else:
                    t = np.eye(3)[np.argmin(np.abs(Z[i]))]
                    X[i] = np.cross(Z[i], t)
                    X[i] /= np.linalg.norm(X[i])
            # origin along the axis: take the foot of the previous normal so that d_i = 0
            if i == 0:
                O[0] = P[0] + ((P[1] - P[0]) @ Z[0]) * Z[0]
            else:
                O[i] = F[i]
            F[i + 1] = O[i] + a[i] * X[i]
        alpha[i] = np.arctan2(X[i] @ cross, Z[i] @ Z[i + 1])
    X[n - 1] = X[n - 2]
    O[n - 1] = F[n - 1]
    F[0] = O[0]                                               # d_1 = 0 by construction
    table = []
    for i in range(n):
        al = 0.0 if i == 0 else alpha[i - 1]
        ai = 0.0 if i == 0 else a[i - 1]
        d = (O[i] - F[i]) @ Z[i]
        Xp = X[i] if i == 0 else X[i - 1]
        th = np.arctan2(np.cross(Xp, X[i]) @ Z[i], Xp @ X[i])
        table.append((float(al), float(ai), float(d), float(th), "R"))
    T0 = make_T(np.column_stack([X[0], np.cross(Z[0], X[0]), Z[0]]), O[0])
    return DHFit(table, T0)


# ---------------------------------------------------------------------------------------------
# Product of exponentials
# ---------------------------------------------------------------------------------------------
def screw_axes(omegas, points) -> np.ndarray:
    """Screw axes ``S_i = [omega_i ; -omega_i x p_i]`` (n x 6) of revolute joints, in the base frame."""
    w = np.asarray(omegas, float)
    p = np.asarray(points, float)
    return np.column_stack([w, -np.cross(w, p)])


def exp_screw(S, theta: float) -> np.ndarray:
    """Matrix exponential ``exp([S] theta)`` of a unit screw ``S = [omega; v]`` with ``|omega| = 1``."""
    w, v = np.asarray(S[:3], float), np.asarray(S[3:], float)
    K = skew(w)
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(3) + s * K + (1 - c) * K @ K
    G = np.eye(3) * theta + (1 - c) * K + (theta - s) * K @ K
    return make_T(R, G @ v)


def poe_fk(S, M, q) -> np.ndarray:
    """Product of exponentials ``exp([S_1] q_1) ... exp([S_n] q_n) M`` (Lynch & Park, space form)."""
    T = np.eye(4)
    for Si, qi in zip(S, q):
        T = T @ exp_screw(Si, qi)
    return T @ np.asarray(M, float)


# ---------------------------------------------------------------------------------------------
# Jacobians and effective mass from MuJoCo
# ---------------------------------------------------------------------------------------------
def mass_matrix(model, data) -> np.ndarray:
    """Dense joint-space inertia matrix ``M(q)`` (nv x nv); ``data`` must be up to date (``mj_forward``).

    The diagonal contains the rotor ("armature") inertia of each joint, as MuJoCo adds it.
    """
    import mujoco

    M = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, M)            # recent releases
    except TypeError:
        mujoco.mj_fullM(model, M, data.qM)         # earlier 3.x releases
    return M


def site_jacobian(model, data, site) -> np.ndarray:
    """6 x nv Jacobian ``[Jv; Jw]`` of a MuJoCo site (name or id) in the world frame (``data`` up to date)."""
    import mujoco

    sid = site if isinstance(site, (int, np.integer)) else mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, site)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, sid)
    return np.vstack([jp, jr])


def point_jacobian(model, data, body, point) -> np.ndarray:
    """6 x nv Jacobian of a point fixed in ``body`` (given by its world coordinates ``point``)."""
    import mujoco

    bid = body if isinstance(body, (int, np.integer)) else mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, body)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jp, jr, np.asarray(point, float), bid)
    return np.vstack([jp, jr])


def operational_inertia(M, Jv) -> np.ndarray:
    """Cartesian (operational-space) inertia ``Lambda = (J M^-1 J^T)^-1`` seen at the point of ``Jv`` (3 x n)."""
    Jv = np.atleast_2d(Jv)
    return np.linalg.inv(Jv @ np.linalg.solve(M, Jv.T))


def effective_mass(M, Jv, u) -> float:
    """Effective (reflected) mass at a point along the direction ``u``: ``1 / (u^T J M^-1 J^T u)``.

    It is the mass of a point that would accelerate at the same rate as the robot's point does when
    pushed along ``u`` with the same force, all other joints being free to follow.
    """
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    Jv = np.atleast_2d(Jv)
    inv_m = u @ Jv @ np.linalg.solve(M, Jv.T @ u)
    return float(1.0 / inv_m)


# ---------------------------------------------------------------------------------------------
# Support polygon and static stability
# ---------------------------------------------------------------------------------------------
def support_polygon(points_xy) -> np.ndarray:
    """Convex hull (counter-clockwise vertices) of 2-D contact points: the **support polygon**."""
    pts = sorted({(round(float(x), 12), round(float(y), 12)) for x, y in np.asarray(points_xy)})
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def polygon_margin(p, poly) -> float:
    """Signed distance from ``p`` to the boundary of a convex polygon (positive inside, negative outside).

    This is the **static stability margin** when ``p`` is the ground projection of the centre of mass
    and ``poly`` is the support polygon.
    """
    p = np.asarray(p, float)
    poly = np.asarray(poly, float)
    m = len(poly)
    inside_d, outside_d = np.inf, np.inf
    inside = True
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        e = b - a
        L = np.linalg.norm(e)
        if L == 0:
            continue
        signed = (e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])) / L        # > 0 on the left (inside for ccw)
        if signed < 0:
            inside = False
        t = np.clip(((p - a) @ e) / L ** 2, 0.0, 1.0)
        d = np.linalg.norm(p - (a + t * e))
        outside_d = min(outside_d, d)
        inside_d = min(inside_d, abs(signed))
    return float(inside_d) if inside else -float(outside_d)


# ---------------------------------------------------------------------------------------------
# Trajectories and animation assets
# ---------------------------------------------------------------------------------------------
def quintic(tau):
    """Quintic time scaling ``s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5`` and its derivative ``s'(tau)``.

    ``s(0) = 0``, ``s(1) = 1`` and the first and second derivatives vanish at both ends, so a segment
    starts and stops at rest with zero acceleration.
    """
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5, 30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4


class RestToRestPath:
    """Piecewise joint-space path through via points, every segment rest-to-rest (quintic scaling).

    ``waypoints`` is a list of joint vectors, ``durations`` the time of each segment (one fewer).
    The object is callable: ``path(t)`` returns ``(q, qd)``; after the last time it holds the end point.
    """

    def __init__(self, waypoints, durations):
        self.q = np.asarray(waypoints, float)
        self.T = np.asarray(durations, float)
        assert len(self.q) == len(self.T) + 1
        self.t_knots = np.concatenate([[0.0], np.cumsum(self.T)])

    @property
    def duration(self) -> float:
        return float(self.t_knots[-1])

    def __call__(self, t):
        t = float(t)
        if t >= self.duration:
            return self.q[-1].copy(), np.zeros_like(self.q[-1])
        k = int(np.searchsorted(self.t_knots, t, side="right") - 1)
        k = max(k, 0)
        tau = (t - self.t_knots[k]) / self.T[k]
        s, ds = quintic(tau)
        dq = self.q[k + 1] - self.q[k]
        return self.q[k] + s * dq, ds * dq / self.T[k]


def save_gif(frames, path, fps: float = 12.0, max_width: int = 320, colors: int = 64) -> bool:
    """Write RGB frames (``uint8`` arrays) as a small animated GIF -- only if ``RG_SAVE_ASSETS=1``.

    The notebook builder sets that variable, so the GIFs used by the chapter README are produced when a
    notebook is built for publication and not during an interactive session.  Returns True if written.
    """
    if os.environ.get("RG_SAVE_ASSETS", "0") != "1":
        return False
    from pathlib import Path
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imgs = []
    for f in frames:
        im = Image.fromarray(np.asarray(f))
        if im.width > max_width:
            im = im.resize((max_width, int(im.height * max_width / im.width)), Image.LANCZOS)
        imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=colors))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return True

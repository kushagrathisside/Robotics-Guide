"""Rigid-body and manipulator dynamics (Craig, *Introduction to Robotics*, Chapter 6).

This module collects, in one tested place, everything that the notebooks of
MFoR Chapter 6 derive step by step:

* **Mass distribution** — inertia tensors of simple solids, the parallel-axis
  theorem, rotation of a tensor, principal axes, composite bodies.
* **A single rigid body** — Newton's and Euler's equations, torque-free motion.
* **Serial manipulators** — a :class:`SerialChain` made of :class:`Link` objects
  (revolute or prismatic), the recursive Newton–Euler algorithm :func:`rnea`
  (outward velocity/acceleration iteration, inward force/torque iteration),
  the joint-space mass matrix, bias and gravity torques, the Coriolis matrix from
  Christoffel symbols, kinetic and potential energy, and forward dynamics.
* **Lagrangian dynamics** — a numerical route (``M = sum m Jv'Jv + Jw' I Jw``) and a
  symbolic route with SymPy (:func:`sym_energies`, :func:`euler_lagrange`).
* **Linearity in the parameters** — the regressor ``Y(q, qd, qdd)`` with ``tau = Y pi``.
* **Simulation** — fixed-step integrators (Euler, semi-implicit Euler, midpoint, RK4).
* **Bridges to MuJoCo** — :func:`chain_from_mujoco` reads a serial chain and its
  mass properties from a compiled ``MjModel``; :func:`mujoco_mass_matrix` returns the
  dense ``M(q)`` in a way that works on every MuJoCo 3.x release.

Conventions
-----------
* A chain has ``n`` links numbered ``1..n``; frame ``{0}`` is the base (world) frame.
* **Link frame.** Frame ``{i}`` is fixed in link ``i``. The joint axis ``a_i`` is a unit
  vector given in frame ``{i}`` and passes through the origin of ``{i}`` (as in Craig's
  convention, where the axis is ``Z_i``). With joint variable ``q_i`` the transform from
  ``{i-1}`` to ``{i}`` is ``T_fix_i`` followed by a rotation of ``q_i`` about ``a_i``
  (revolute) or a translation of ``q_i`` along ``a_i`` (prismatic).
* **Inertial data** of link ``i`` are its mass ``m``, the centre-of-mass position ``com``
  and the inertia tensor ``I_com`` about the centre of mass, written in the axes of
  ``{i}``. The tensor holds the usual entries ``I_xx = int (y^2+z^2) dm`` and
  ``I_xy = -int x y dm`` (Craig's convention), so a *product of inertia* appears with a minus sign.
* Every vector in :func:`rnea` is expressed in the frame of the link it belongs to, as in
  Craig's iterative Newton–Euler formulation. Gravity is applied, as Craig suggests, by giving
  the base the acceleration ``-g_vec``.
* Angles in radians, SI units everywhere.

The Newton–Euler routines use the "origin form" of the rigid-body equations (the force and
torque are expressed about the *origin of the link frame* with the first moment ``h = m com``
and the tensor ``I_o`` about that origin). This is algebraically identical to Craig's
centre-of-mass form (a test checks it) but makes the equations *linear in the ten inertial
parameters of each link* — which is what the regressor of Section 6.4 needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .transforms import skew, quat_mul, quat_to_rot

__all__ = [
    # mass distribution
    "inertia_matrix", "inertia_point_masses", "inertia_box", "inertia_cylinder", "inertia_sphere",
    "inertia_ellipsoid", "inertia_rod", "inertia_capsule", "parallel_axis", "parallel_axis_inverse",
    "rotate_inertia", "principal_axes", "is_physical_inertia", "combine_bodies", "box_from_inertia",
    # single rigid body
    "newton_euler_body", "angular_acceleration", "rotational_energy", "angular_momentum",
    "free_body_rhs", "free_body_state", "free_body_rotation",
    # manipulators
    "Link", "SerialChain", "planar_chain", "dh_chain", "link_frames", "link_jacobians",
    "geometric_jacobian", "rnea", "inverse_dynamics", "mass_matrix", "mass_matrix_lagrange",
    "bias_torque", "gravity_torque", "mass_matrix_derivative", "coriolis_matrix",
    "kinetic_energy", "potential_energy", "forward_dynamics", "cartesian_inertia",
    # parameters
    "link_pi", "chain_pi", "chain_with_pi", "regressor", "numerical_rank",
    # simulation
    "step_euler", "step_semi_implicit", "step_midpoint", "step_rk4", "integrate", "simulate",
    # symbolic
    "sym_rodrigues", "sym_energies", "euler_lagrange",
    # MuJoCo bridges
    "chain_from_mujoco", "mujoco_mass_matrix", "mujoco_dofs", "mujoco_rigid_copy",
]

GRAVITY = np.array([0.0, 0.0, -9.81])


# ===========================================================================
# 1. Mass distribution
# ===========================================================================
def inertia_matrix(Ixx, Iyy, Izz, Ixy=0.0, Ixz=0.0, Iyz=0.0) -> np.ndarray:
    """Inertia tensor from moments ``Ixx, Iyy, Izz`` and **products** ``Ixy = int x y dm`` etc.

    The off-diagonal entries of the tensor are the *negatives* of the products of inertia
    (Craig's convention), so ``inertia_matrix(1, 2, 3, Ixy=0.1)[0, 1] == -0.1``.
    """
    return np.array([[Ixx, -Ixy, -Ixz],
                     [-Ixy, Iyy, -Iyz],
                     [-Ixz, -Iyz, Izz]], dtype=float)


def inertia_point_masses(masses, points, about=None) -> np.ndarray:
    """Inertia tensor of a set of point masses about the point ``about`` (default: the origin).

    ``I = sum_k m_k (|r_k|^2 E - r_k r_k^T)`` with ``r_k = p_k - about``.
    """
    masses = np.asarray(masses, float)
    pts = np.asarray(points, float).reshape(-1, 3)
    if about is not None:
        pts = pts - np.asarray(about, float)
    I = np.zeros((3, 3))
    for m, r in zip(masses, pts):
        I += m * (r @ r * np.eye(3) - np.outer(r, r))
    return I


def inertia_box(m, a, b, c) -> np.ndarray:
    """Solid cuboid of mass ``m`` and **full** side lengths ``a, b, c`` along x, y, z, about its centre."""
    return np.diag([m * (b * b + c * c) / 12.0, m * (a * a + c * c) / 12.0, m * (a * a + b * b) / 12.0])


def _axis_perm(I_axis_z: np.ndarray, axis: int) -> np.ndarray:
    """Re-express a tensor of a body symmetric about z so that the symmetry axis is ``axis`` (0, 1, 2)."""
    d = np.diag(I_axis_z).copy()               # (Ixx, Iyy, Izz) with z the symmetry axis
    out = np.empty(3)
    out[axis] = d[2]
    others = [i for i in range(3) if i != axis]
    out[others] = d[0]
    return np.diag(out)


def inertia_cylinder(m, r, h, axis=2) -> np.ndarray:
    """Solid circular cylinder: radius ``r``, height ``h``, symmetry axis ``axis`` (0, 1 or 2), about its centre."""
    t = m * (3 * r * r + h * h) / 12.0
    return _axis_perm(np.diag([t, t, 0.5 * m * r * r]), axis)


def inertia_sphere(m, r) -> np.ndarray:
    """Solid sphere of radius ``r`` about its centre: ``(2/5) m r^2`` on the diagonal."""
    return np.eye(3) * (0.4 * m * r * r)


def inertia_ellipsoid(m, a, b, c) -> np.ndarray:
    """Solid ellipsoid with **semi**-axes ``a, b, c`` along x, y, z, about its centre."""
    return np.diag([m * (b * b + c * c) / 5.0, m * (a * a + c * c) / 5.0, m * (a * a + b * b) / 5.0])


def inertia_rod(m, L, axis=0) -> np.ndarray:
    """Thin uniform rod of length ``L`` along ``axis`` about its centre: ``m L^2 / 12`` about the two transverse axes."""
    d = np.full(3, m * L * L / 12.0)
    d[axis] = 0.0
    return np.diag(d)


def inertia_capsule(m, r, h, axis=2) -> np.ndarray:
    """Uniform capsule: cylinder of radius ``r`` and length ``h`` plus two hemispherical caps, about its centre."""
    v_cyl, v_sph = np.pi * r * r * h, 4.0 / 3.0 * np.pi * r ** 3
    m_cyl, m_sph = m * v_cyl / (v_cyl + v_sph), m * v_sph / (v_cyl + v_sph)    # m_sph: both hemispheres
    I_ax = 0.5 * m_cyl * r * r + 0.4 * m_sph * r * r
    I_tr = m_cyl * (3 * r * r + h * h) / 12.0 + m_sph * (0.4 * r * r + 0.25 * h * h + 0.375 * h * r)
    return _axis_perm(np.diag([I_tr, I_tr, I_ax]), axis)


def parallel_axis(I_com, m, d) -> np.ndarray:
    """Tensor about a point displaced by ``d`` from the centre of mass: ``I + m (d.d E - d d^T)`` (Steiner)."""
    d = np.asarray(d, float)
    return np.asarray(I_com, float) + m * (d @ d * np.eye(3) - np.outer(d, d))


def parallel_axis_inverse(I_o, m, d) -> np.ndarray:
    """Inverse of :func:`parallel_axis`: tensor about the centre of mass from the tensor about a point ``d`` away."""
    d = np.asarray(d, float)
    return np.asarray(I_o, float) - m * (d @ d * np.eye(3) - np.outer(d, d))


def rotate_inertia(I, R) -> np.ndarray:
    """Re-express a tensor in rotated axes: ``R I R^T`` (``R`` maps the old components to the new ones' frame)."""
    R = np.asarray(R, float)
    return R @ np.asarray(I, float) @ R.T


def principal_axes(I):
    """Principal moments (ascending) and the matrix whose columns are the principal axes (right-handed)."""
    w, V = np.linalg.eigh(0.5 * (I + I.T))
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1.0
    return w, V


def is_physical_inertia(I, tol=1e-12) -> bool:
    """True if ``I`` is symmetric, positive semi-definite and obeys the triangle inequalities of its principal moments."""
    I = np.asarray(I, float)
    if not np.allclose(I, I.T, atol=1e-12):
        return False
    w = np.linalg.eigvalsh(I)
    return bool(w[0] >= -tol and w[2] <= w[0] + w[1] + tol)


def combine_bodies(parts):
    """Composite of rigid bodies given in one common frame.

    ``parts`` is a list of ``(m, com, I_com)``. Returns ``(m, com, I_com)`` of the composite body, using
    the parallel-axis theorem to refer every tensor to the common centre of mass.
    """
    m = sum(p[0] for p in parts)
    com = sum(p[0] * np.asarray(p[1], float) for p in parts) / m
    I = sum(parallel_axis(p[2], p[0], np.asarray(p[1], float) - com) for p in parts)
    return m, com, I


def box_from_inertia(m, I_principal):
    """Side lengths ``(a, b, c)`` of the uniform cuboid with mass ``m`` and the given principal moments (MuJoCo draws these)."""
    Ixx, Iyy, Izz = np.asarray(I_principal, float)
    a2 = 6.0 * (Iyy + Izz - Ixx) / m
    b2 = 6.0 * (Ixx + Izz - Iyy) / m
    c2 = 6.0 * (Ixx + Iyy - Izz) / m
    return np.sqrt([a2, b2, c2])


# ===========================================================================
# 2. A single rigid body
# ===========================================================================
def newton_euler_body(m, I_c, a_c, omega, alpha):
    """Newton's and Euler's equations for one body (all vectors in one frame whose axes are those of ``I_c``).

    Returns ``(F, N)``: the force through the centre of mass, ``F = m a_c``, and the torque about the centre of
    mass, ``N = I alpha + omega x (I omega)``, needed to produce the motion.
    """
    omega = np.asarray(omega, float)
    return m * np.asarray(a_c, float), I_c @ np.asarray(alpha, float) + np.cross(omega, I_c @ omega)


def angular_acceleration(I_c, omega, N):
    """Euler's equation solved for ``alpha``: ``I^{-1} (N - omega x I omega)``."""
    omega = np.asarray(omega, float)
    return np.linalg.solve(I_c, np.asarray(N, float) - np.cross(omega, I_c @ omega))


def rotational_energy(I_c, omega) -> float:
    """``(1/2) omega^T I omega``."""
    omega = np.asarray(omega, float)
    return 0.5 * float(omega @ I_c @ omega)


def angular_momentum(I_c, omega) -> np.ndarray:
    """``L = I omega`` (in the frame of the tensor)."""
    return I_c @ np.asarray(omega, float)


def free_body_rhs(I_c, torque=None):
    """Right-hand side ``f(t, x)`` of a free rigid body; the state is ``x = [quaternion (w,x,y,z), omega_body]``.

    ``torque`` is an optional function ``torque(t, x) -> body-frame torque``; default none (torque-free motion).
    """
    I_c = np.asarray(I_c, float)

    def f(t, x):
        q, w = x[:4], x[4:]
        N = np.zeros(3) if torque is None else np.asarray(torque(t, x), float)
        qdot = 0.5 * quat_mul(q, np.concatenate([[0.0], w]))
        wdot = np.linalg.solve(I_c, N - np.cross(w, I_c @ w))
        return np.concatenate([qdot, wdot])
    return f


def free_body_state(q=(1.0, 0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0)):
    """Pack an orientation quaternion and a body-frame angular velocity into the state of :func:`free_body_rhs`."""
    return np.concatenate([np.asarray(q, float), np.asarray(omega, float)])


def free_body_rotation(I_c, omega0, t_end, dt, q0=(1.0, 0.0, 0.0, 0.0), torque=None):
    """Integrate a free rigid body with RK4. Returns ``(t, quats, omegas, rotations)`` (body-to-world matrices)."""
    f = free_body_rhs(I_c, torque)
    t, X = integrate(f, free_body_state(q0, omega0), t_end, dt, method="rk4")
    X[:, :4] /= np.linalg.norm(X[:, :4], axis=1, keepdims=True)
    Rs = np.array([quat_to_rot(x[:4]) for x in X])
    return t, X[:, :4], X[:, 4:], Rs


# ===========================================================================
# 3. Serial manipulators
# ===========================================================================
def _cross(a, b):
    """Cross product of two 3-vectors (several times faster than ``np.cross`` for single vectors)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _rodrigues(axis, theta):
    """Rotation by ``theta`` about the unit ``axis``; analytic in ``theta`` (works with complex ``theta``)."""
    a = np.asarray(axis, float)
    K = skew(a)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


@dataclass
class Link:
    """One link of a serial chain (see the module docstring for the conventions)."""

    m: float
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    I_com: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))      # rotation part of T_fix  ({i-1} -> {i} at q = 0)
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))    # origin of {i} in {i-1} at q = 0
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    kind: str = "R"
    name: str = ""

    def __post_init__(self):
        self.m = float(self.m)
        self.com = np.asarray(self.com, float).reshape(3)
        self.I_com = np.asarray(self.I_com, float).reshape(3, 3)
        self.R = np.asarray(self.R, float).reshape(3, 3)
        self.p = np.asarray(self.p, float).reshape(3)
        self.axis = np.asarray(self.axis, float).reshape(3)
        self.axis = self.axis / np.linalg.norm(self.axis)
        self._K = skew(self.axis)                     # cached for the Rodrigues formula (axis must not be changed afterwards)
        self._KK = self._K @ self._K

    @property
    def h(self) -> np.ndarray:
        """First moment ``m * com`` (about the origin of the link frame)."""
        return self.m * self.com

    @property
    def I_o(self) -> np.ndarray:
        """Inertia tensor about the **origin of the link frame** (parallel-axis theorem)."""
        return parallel_axis(self.I_com, self.m, self.com)

    @property
    def T_fix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.p
        return T


@dataclass
class SerialChain:
    """A serial chain of rigid links plus optional joint-level data (armature, viscous and Coulomb friction)."""

    links: list
    armature: np.ndarray = None
    damping: np.ndarray = None
    frictionloss: np.ndarray = None
    name: str = ""

    def __post_init__(self):
        n = len(self.links)
        self.armature = np.zeros(n) if self.armature is None else np.asarray(self.armature, float)
        self.damping = np.zeros(n) if self.damping is None else np.asarray(self.damping, float)
        self.frictionloss = np.zeros(n) if self.frictionloss is None else np.asarray(self.frictionloss, float)

    @property
    def n(self) -> int:
        return len(self.links)

    def without_joint_effects(self) -> "SerialChain":
        """A copy with zero armature, damping and Coulomb friction (pure rigid-link dynamics)."""
        return SerialChain(self.links, name=self.name)


def planar_chain(lengths, masses, com=None, Izz=None, name="planar") -> SerialChain:
    """Planar n-R arm: joint axes ``z``, links along ``x``; ``com[i]`` is the distance of link ``i``'s centre
    of mass from joint ``i`` (default: half the link length) and ``Izz[i]`` its inertia about the centre of
    mass (default: uniform thin rod, ``m L^2 / 12``). Use gravity ``(0, -g, 0)`` for a vertical plane."""
    L = np.asarray(lengths, float)
    m = np.asarray(masses, float)
    c = L / 2.0 if com is None else np.asarray(com, float)
    Iz = m * L ** 2 / 12.0 if Izz is None else np.asarray(Izz, float)
    links = []
    for i in range(len(L)):
        links.append(Link(m[i], [c[i], 0, 0], np.diag([0.0, Iz[i], Iz[i]]),
                          p=[0.0 if i == 0 else L[i - 1], 0, 0], name=f"link{i + 1}"))
    return SerialChain(links, name=name)


def dh_chain(table, masses, coms, I_coms, name="dh-chain") -> SerialChain:
    """Chain from a **modified (Craig)** D-H table ``[(alpha, a, d, theta, kind), ...]`` (alpha, a of the previous link).

    ``T = Rx(alpha) Dx(a) Rz(theta) Dz(d)``; the joint variable adds to ``theta`` (revolute) or ``d`` (prismatic).
    The constant parts of ``theta`` and ``d`` are folded into the fixed transform; the axis is the link ``z``.
    """
    from .kinematics import dh_modified

    links = []
    for row, m, c, I in zip(table, masses, coms, I_coms):
        alpha, a, d, theta, kind = row
        T = dh_modified(alpha, a, d, theta)
        links.append(Link(m, c, I, R=T[:3, :3], p=T[:3, 3], kind=kind))
    return SerialChain(links, name=name)


def _joint_transform(L: Link, qi):
    if L.kind == "R":
        return L.R @ (np.eye(3) + np.sin(qi) * L._K + (1.0 - np.cos(qi)) * L._KK), L.p
    return L.R, L.p + L.R @ (L.axis * qi)


def link_frames(chain: SerialChain, q):
    """Poses ``[T_0_1, ..., T_0_n]`` (4x4) of the link frames in the base frame."""
    T, out = np.eye(4), []
    for L, qi in zip(chain.links, q):
        R, p = _joint_transform(L, qi)
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = R, p
        T = T @ step
        out.append(T.copy())
    return out


def link_jacobians(chain: SerialChain, q):
    """For every link: ``(Jv_c, Jw, R_0i, p_c)`` — Jacobians of the centre of mass (linear) and of the link
    (angular), both in the **base frame** (3 x n), the link's rotation and its centre of mass in the base frame."""
    n = chain.n
    Ts = link_frames(chain, q)
    out = []
    for i, (L, T) in enumerate(zip(chain.links, Ts)):
        R0i, o_i = T[:3, :3], T[:3, 3]
        p_c = o_i + R0i @ L.com
        Jv, Jw = np.zeros((3, n)), np.zeros((3, n))
        for j in range(i + 1):
            Lj, Tj = chain.links[j], Ts[j]
            z = Tj[:3, :3] @ Lj.axis
            o_j = Tj[:3, 3]
            if Lj.kind == "R":
                Jv[:, j], Jw[:, j] = np.cross(z, p_c - o_j), z
            else:
                Jv[:, j] = z
        out.append((Jv, Jw, R0i, p_c))
    return out


def geometric_jacobian(chain: SerialChain, q, point=None):
    """6 x n Jacobian (linear on top, angular below, base frame) of the point ``point`` fixed in the last link
    frame (default: its origin)."""
    n = chain.n
    Ts = link_frames(chain, q)
    pt = Ts[-1][:3, 3] if point is None else Ts[-1][:3, :3] @ np.asarray(point, float) + Ts[-1][:3, 3]
    J = np.zeros((6, n))
    for j in range(n):
        Lj, Tj = chain.links[j], Ts[j]
        z = Tj[:3, :3] @ Lj.axis
        if Lj.kind == "R":
            J[:3, j], J[3:, j] = np.cross(z, pt - Tj[:3, 3]), z
        else:
            J[:3, j] = z
    return J


def _rnea_core(chain, m, h, Io, q, qd, qdd, g, f_tip, n_tip, details=False):
    """Recursive Newton–Euler with the inertial data supplied as arrays (so that regressors can reuse it)."""
    n = chain.n
    q, qd, qdd = (np.asarray(x) for x in (q, qd, qdd))
    dt = np.result_type(q, qd, qdd, np.asarray(g), np.float64)
    w, wd, vd = np.zeros(3, dt), np.zeros(3, dt), -np.asarray(g).astype(dt)       # base: v_dot_0 = -g
    Rs, ps, Fs, Ns, W, Wd, Vd = [], [], [], [], [], [], []
    for i, L in enumerate(chain.links):                                          # ---- outward iteration
        R, p = _joint_transform(L, q[i])
        Rt = R.T
        a = L.axis
        if L.kind == "R":
            wi = Rt @ w + qd[i] * a
            wdi = Rt @ wd + _cross(Rt @ w, qd[i] * a) + qdd[i] * a
            vdi = Rt @ (_cross(wd, p) + _cross(w, _cross(w, p)) + vd)
        else:
            wi, wdi = Rt @ w, Rt @ wd
            vdi = Rt @ (_cross(wd, p) + _cross(w, _cross(w, p)) + vd) + 2.0 * _cross(wi, qd[i] * a) + qdd[i] * a
        F = m[i] * vdi + _cross(wdi, h[i]) + _cross(wi, _cross(wi, h[i]))
        N = Io[i] @ wdi + _cross(wi, Io[i] @ wi) + _cross(h[i], vdi)
        Rs.append(R), ps.append(p), Fs.append(F), Ns.append(N), W.append(wi), Wd.append(wdi), Vd.append(vdi)
        w, wd, vd = wi, wdi, vdi
    f = np.zeros(3, dt) if f_tip is None else np.asarray(f_tip).astype(dt)          # ---- inward iteration
    nn = np.zeros(3, dt) if n_tip is None else np.asarray(n_tip).astype(dt)
    Rn, pn = np.eye(3), np.zeros(3)
    tau, fs, ns = np.zeros(n, dt), [None] * n, [None] * n
    for i in reversed(range(n)):
        cross_term = _cross(pn, Rn @ f)
        f = Rn @ f + Fs[i]
        nn = Ns[i] + Rn @ nn + cross_term
        fs[i], ns[i] = f, nn
        tau[i] = (nn if chain.links[i].kind == "R" else f) @ chain.links[i].axis
        Rn, pn = Rs[i], ps[i]
    if details:
        return tau, dict(omega=W, omega_dot=Wd, v_dot=Vd, F=Fs, N=Ns, f=fs, n=ns)
    return tau


def _arrays(chain):
    m = np.array([L.m for L in chain.links])
    h = np.array([L.h for L in chain.links])
    Io = np.array([L.I_o for L in chain.links])
    return m, h, Io


def rnea(chain: SerialChain, q, qd=None, qdd=None, g=GRAVITY, f_tip=None, n_tip=None, details=False):
    """Recursive Newton–Euler inverse dynamics of the *rigid links* (Craig's iterative formulation).

    Returns the joint torques/forces ``tau`` that produce ``qdd`` at state ``(q, qd)`` in gravity ``g``
    (a 3-vector in the base frame). ``f_tip, n_tip`` is the force and moment **exerted by the last link on the
    environment**, expressed in the last link frame (``tau`` then includes ``J^T F``, as in Craig's statics).
    With ``details=True`` a dictionary of the intermediate vectors is returned as well. Armature and friction
    are *not* included here; see :func:`inverse_dynamics`.
    """
    n = chain.n
    qd = np.zeros(n) if qd is None else qd
    qdd = np.zeros(n) if qdd is None else qdd
    m, h, Io = _arrays(chain)
    return _rnea_core(chain, m, h, Io, q, qd, qdd, g, f_tip, n_tip, details)


def inverse_dynamics(chain: SerialChain, q, qd, qdd, g=GRAVITY, f_tip=None, n_tip=None, joint_effects=True):
    """``tau = M qdd + C qd + g`` plus, if ``joint_effects``, the armature ``J_a qdd``, viscous ``b qd`` and Coulomb ``f_c sign(qd)`` terms."""
    qd, qdd = np.asarray(qd, float), np.asarray(qdd, float)
    tau = rnea(chain, q, qd, qdd, g, f_tip, n_tip)
    if joint_effects:
        tau = tau + chain.armature * qdd + chain.damping * qd + chain.frictionloss * np.sign(qd)
    return tau


def mass_matrix(chain: SerialChain, q, joint_effects=False) -> np.ndarray:
    """Joint-space inertia matrix ``M(q)`` by n Newton–Euler passes (``qd = 0``, no gravity, ``qdd = e_j``)."""
    n = chain.n
    q = np.asarray(q)
    dt = np.result_type(q, np.float64)
    M = np.zeros((n, n), dt)
    z, g0 = np.zeros(n), np.zeros(3)
    arrays = _arrays(chain)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        M[:, j] = _rnea_core(chain, *arrays, q, z, e, g0, None, None)
    return M + np.diag(chain.armature) if joint_effects else M


def mass_matrix_lagrange(chain: SerialChain, q) -> np.ndarray:
    """Mass matrix from the Lagrangian formula ``M = sum_i  m_i Jv_i^T Jv_i + Jw_i^T (R_i I_i R_i^T) Jw_i``."""
    n = chain.n
    M = np.zeros((n, n))
    for L, (Jv, Jw, R0i, _) in zip(chain.links, link_jacobians(chain, q)):
        I0 = R0i @ L.I_com @ R0i.T
        M += L.m * Jv.T @ Jv + Jw.T @ I0 @ Jw
    return M


def bias_torque(chain: SerialChain, q, qd, g=GRAVITY) -> np.ndarray:
    """``C(q, qd) qd + g(q)`` — what MuJoCo calls ``qfrc_bias`` — from one Newton–Euler pass with ``qdd = 0``."""
    return rnea(chain, q, qd, np.zeros(chain.n), g)


def gravity_torque(chain: SerialChain, q, g=GRAVITY) -> np.ndarray:
    """``g(q)``: the torque that holds the arm still (``qd = qdd = 0``)."""
    return rnea(chain, q, np.zeros(chain.n), np.zeros(chain.n), g)


def mass_matrix_derivative(chain: SerialChain, q, h=1e-30) -> np.ndarray:
    """``dM[k, i, j] = d M_ij / d q_k`` to machine precision by the **complex-step** method (needs no step-size tuning)."""
    q = np.asarray(q, float)
    n = chain.n
    dM = np.zeros((n, n, n))
    for k in range(n):
        qc = q.astype(complex)
        qc[k] += 1j * h
        dM[k] = mass_matrix(chain, qc).imag / h
    return dM


def coriolis_matrix(chain: SerialChain, q, qd) -> np.ndarray:
    """Coriolis/centrifugal matrix with the Christoffel-symbol entries
    ``C_ij = sum_k (1/2)(d M_ij/d q_k + d M_ik/d q_j - d M_jk/d q_i) qd_k`` (so that ``Mdot - 2C`` is skew-symmetric)."""
    qd = np.asarray(qd, float)
    dM = mass_matrix_derivative(chain, q)
    n = chain.n
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = 0.5 * sum((dM[k, i, j] + dM[j, i, k] - dM[i, j, k]) * qd[k] for k in range(n))
    return C


def potential_energy(chain: SerialChain, q, g=GRAVITY) -> float:
    """``V = - sum_i m_i g . p_ci`` (zero at the base origin)."""
    g = np.asarray(g, float)
    return float(-sum(L.m * g @ pc for L, (_, _, _, pc) in zip(chain.links, link_jacobians(chain, q))))


def kinetic_energy(chain: SerialChain, q, qd, joint_effects=False) -> float:
    """``K = (1/2) qd^T M(q) qd``."""
    qd = np.asarray(qd, float)
    return 0.5 * float(qd @ mass_matrix(chain, q, joint_effects) @ qd)


def forward_dynamics(chain: SerialChain, q, qd, tau=None, g=GRAVITY, f_tip=None, n_tip=None, joint_effects=True):
    """``qdd = M^{-1} (tau - C qd - g)`` with armature, viscous damping and Coulomb friction when ``joint_effects``."""
    n = chain.n
    tau = np.zeros(n) if tau is None else np.asarray(tau, float)
    qd = np.asarray(qd, float)
    M = mass_matrix(chain, q, joint_effects)
    b = rnea(chain, q, qd, np.zeros(n), g, f_tip, n_tip)
    if joint_effects:
        b = b + chain.damping * qd + chain.frictionloss * np.sign(qd)
    return np.linalg.solve(M, tau - b)


def cartesian_inertia(chain: SerialChain, q, J=None) -> np.ndarray:
    """Operational-space inertia ``Lambda = (J M^{-1} J^T)^{-1}`` for the Jacobian ``J`` (default: full 6 x n tool Jacobian)."""
    J = geometric_jacobian(chain, q) if J is None else np.asarray(J, float)
    M = mass_matrix(chain, q)
    return np.linalg.inv(J @ np.linalg.solve(M, J.T))


# ===========================================================================
# 4. Linearity in the parameters
# ===========================================================================
def link_pi(L: Link) -> np.ndarray:
    """The ten inertial parameters of a link about its frame origin:
    ``[m, h_x, h_y, h_z, Io_xx, Io_yy, Io_zz, Io_xy, Io_xz, Io_yz]`` with ``h = m com`` and the tensor entries of ``I_o``."""
    Io = L.I_o
    return np.array([L.m, *L.h, Io[0, 0], Io[1, 1], Io[2, 2], Io[0, 1], Io[0, 2], Io[1, 2]])


def chain_pi(chain: SerialChain) -> np.ndarray:
    """Stacked parameter vector ``pi`` (length ``10 n``) of a chain."""
    return np.concatenate([link_pi(L) for L in chain.links])


def _unpack(pi, n):
    pi = np.asarray(pi, float).reshape(n, 10)
    m, h = pi[:, 0], pi[:, 1:4]
    Io = np.zeros((n, 3, 3))
    for i in range(n):
        xx, yy, zz, xy, xz, yz = pi[i, 4:]
        Io[i] = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    return m, h, Io


def chain_with_pi(chain: SerialChain, pi) -> SerialChain:
    """A copy of ``chain`` whose links carry the inertial parameters ``pi`` (links with ``m = 0`` get ``com = 0``)."""
    m, h, Io = _unpack(pi, chain.n)
    links = []
    for L, mi, hi, Ioi in zip(chain.links, m, h, Io):
        com = hi / mi if abs(mi) > 1e-15 else np.zeros(3)
        links.append(Link(mi, com, parallel_axis_inverse(Ioi, mi, com), L.R, L.p, L.axis, L.kind, L.name))
    return SerialChain(links, chain.armature, chain.damping, chain.frictionloss, chain.name)


def regressor(chain: SerialChain, q, qd, qdd, g=GRAVITY) -> np.ndarray:
    """The ``n x 10n`` regressor ``Y`` with ``tau = Y(q, qd, qdd) pi`` for the rigid links.

    The recursive Newton–Euler torques are linear in ``pi`` (see :func:`link_pi`), so the column of ``Y``
    belonging to one parameter is the torque vector obtained when that parameter is 1 and all others are 0.
    """
    n = chain.n
    Y = np.zeros((n, 10 * n))
    sym_index = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    g = np.asarray(g, float)
    for i in range(n):
        for k in range(10):
            m, h, Io = np.zeros(n), np.zeros((n, 3)), np.zeros((n, 3, 3))
            if k == 0:
                m[i] = 1.0
            elif k < 4:
                h[i, k - 1] = 1.0
            else:
                a, b = sym_index[k - 4]
                Io[i, a, b] = Io[i, b, a] = 1.0
            Y[:, 10 * i + k] = _rnea_core(chain, m, h, Io, q, qd, qdd, g, None, None)
    return Y


def numerical_rank(A, tol=None) -> int:
    """Rank from the singular values (relative tolerance ``1e-9`` of the largest by default)."""
    s = np.linalg.svd(A, compute_uv=False)
    tol = 1e-9 * s[0] if tol is None else tol
    return int((s > tol).sum())


# ===========================================================================
# 5. Simulation
# ===========================================================================
def step_euler(f, t, x, dt):
    """Explicit (forward) Euler step of ``x' = f(t, x)``."""
    return x + dt * f(t, x)


def step_midpoint(f, t, x, dt):
    """Explicit midpoint (second-order Runge–Kutta) step."""
    k1 = f(t, x)
    return x + dt * f(t + dt / 2, x + dt / 2 * k1)


def step_rk4(f, t, x, dt):
    """Classical fourth-order Runge–Kutta step."""
    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt / 2 * k1)
    k3 = f(t + dt / 2, x + dt / 2 * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def step_semi_implicit(f, t, x, dt):
    """Semi-implicit (symplectic) Euler for ``x = [q, qd]``: update the velocity first, then the position with the **new** velocity."""
    n = len(x) // 2
    q, qd = x[:n], x[n:]
    qdd = f(t, x)[n:]
    qd_new = qd + dt * qdd
    return np.concatenate([q + dt * qd_new, qd_new])


_STEPPERS = {"euler": step_euler, "midpoint": step_midpoint, "rk4": step_rk4, "semi-implicit": step_semi_implicit}


def integrate(f, x0, t_end, dt, method="rk4", t0=0.0):
    """Fixed-step integration of ``x' = f(t, x)``; returns ``(t, X)`` with one row of ``X`` per time sample."""
    step = _STEPPERS[method]
    N = int(round((t_end - t0) / dt))
    t = t0 + dt * np.arange(N + 1)
    X = np.empty((N + 1, len(x0)))
    X[0] = x0
    for k in range(N):
        X[k + 1] = step(f, t[k], X[k], dt)
    return t, X


def simulate(chain: SerialChain, q0, qd0, t_end, dt, tau_fn=None, g=GRAVITY, method="rk4", joint_effects=True):
    """Forward-dynamics simulation of a chain; ``tau_fn(t, q, qd)`` gives the joint torques (default zero).

    Returns ``(t, q, qd)`` with one row per time sample.
    """
    n = chain.n

    def f(t, x):
        q, qd = x[:n], x[n:]
        tau = None if tau_fn is None else tau_fn(t, q, qd)
        return np.concatenate([qd, forward_dynamics(chain, q, qd, tau, g, joint_effects=joint_effects)])

    t, X = integrate(f, np.concatenate([q0, qd0]).astype(float), t_end, dt, method)
    return t, X[:, :n], X[:, n:]


# ===========================================================================
# 6. Symbolic Lagrangian dynamics (SymPy)
# ===========================================================================
def sym_rodrigues(axis, q):
    """Symbolic rotation about the constant unit ``axis`` by the symbol ``q``."""
    import sympy as sp

    a1, a2, a3 = axis
    K = sp.Matrix([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])
    return sp.eye(3) + sp.sin(q) * K + (1 - sp.cos(q)) * K * K


def sym_energies(links, q, qd, gvec):
    """Kinetic and potential energy of a symbolic serial chain.

    ``links`` is a list of dicts with keys ``R`` and ``p`` (fixed transform from the previous frame, SymPy
    matrices), ``axis`` (constant unit vector), ``m``, ``com`` (3-vector) and ``I`` (inertia about the centre of
    mass in link axes) — all possibly symbolic; ``kind`` is optional (``"R"`` default). ``gvec`` is the gravity
    vector in the base frame. Returns ``(K, V)``.
    """
    import sympy as sp

    R_prev, p_prev, w_prev = sp.eye(3), sp.zeros(3, 1), sp.zeros(3, 1)
    K, V = 0, 0
    for L, qi, qdi in zip(links, q, qd):
        a = sp.Matrix(L["axis"])
        if L.get("kind", "R") == "R":
            R_i = R_prev * L["R"] * sym_rodrigues(L["axis"], qi)
            o_i = p_prev + R_prev * L["p"]
        else:
            R_i = R_prev * L["R"]
            o_i = p_prev + R_prev * (L["p"] + L["R"] * a * qi)
        z = R_i * a
        p_c = o_i + R_i * sp.Matrix(L["com"])
        v_c = sp.Matrix([sum(sp.diff(p_c[r], qj) * qdj for qj, qdj in zip(q, qd)) for r in range(3)])
        w = w_prev + (z * qdi if L.get("kind", "R") == "R" else sp.zeros(3, 1))
        I0 = R_i * L["I"] * R_i.T
        K += sp.Rational(1, 2) * L["m"] * (v_c.T * v_c)[0] + sp.Rational(1, 2) * (w.T * I0 * w)[0]
        V += -L["m"] * (sp.Matrix(gvec).T * p_c)[0]
        R_prev, p_prev, w_prev = R_i, o_i, w
    return K, V


def euler_lagrange(K, V, q, qd, simplify=None):
    """From energies to ``tau = M(q) qdd + C(q, qd) qd + g(q)``.

    Returns ``(M, C, g)`` as SymPy matrices: ``M = d^2 K / d qd^2``, ``g = dV/dq`` and
    ``C_kj = sum_i c_ijk qd_i`` with the Christoffel symbols of the first kind
    ``c_ijk = (1/2)(d m_kj/d q_i + d m_ki/d q_j - d m_ij/d q_k)``.
    """
    import sympy as sp

    simp = simplify or sp.simplify
    n = len(q)
    M = sp.Matrix(n, n, lambda i, j: simp(sp.diff(K, qd[i], qd[j])))
    g = sp.Matrix([simp(sp.diff(V, qi)) for qi in q])
    C = sp.zeros(n, n)
    for k in range(n):
        for j in range(n):
            C[k, j] = simp(sum(sp.Rational(1, 2) * (sp.diff(M[k, j], q[i]) + sp.diff(M[k, i], q[j]) - sp.diff(M[i, j], q[k])) * qd[i]
                               for i in range(n)))
    return M, C, g


# ===========================================================================
# 7. MuJoCo bridges (no import of mujoco needed to read a compiled model)
# ===========================================================================
def mujoco_mass_matrix(model, data) -> np.ndarray:
    """Dense ``M(q)`` of a MuJoCo model, valid for every MuJoCo 3.x signature of ``mj_fullM`` (call ``mj_forward`` first)."""
    import mujoco

    M = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, M)           # recent releases: (model, data, dst)
    except TypeError:
        mujoco.mj_fullM(model, M, data.qM)        # earlier 3.x releases: (model, dst, qM)
    return M


def mujoco_rigid_copy(model, constraints=False, passive=False, actuation=False):
    """Copy of a MuJoCo model with selected features switched off, so that ``mj_inverse`` / ``mj_step`` follow the
    rigid-body equations of this module exactly.

    ``constraints=False`` disables **all** constraints (contacts, joint limits, equality constraints and
    ``frictionloss``), ``passive=False`` disables joint damping and springs, ``actuation=False`` disables the
    actuators. Armature is a property of the mass matrix and is therefore never removed; see :attr:`SerialChain.armature`.
    """
    import copy
    import mujoco

    m = copy.copy(model)
    bits = mujoco.mjtDisableBit
    flags = int(model.opt.disableflags)
    if not constraints:
        flags |= int(bits.mjDSBL_CONSTRAINT)
    if not passive:                                  # older releases have one PASSIVE flag, newer ones SPRING and DAMPER
        for name in ("mjDSBL_PASSIVE", "mjDSBL_SPRING", "mjDSBL_DAMPER"):
            flags |= int(getattr(bits, name, 0))
    if not actuation:
        flags |= int(bits.mjDSBL_ACTUATION)
    m.opt.disableflags = flags
    return m


def _body_T(model, b):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = quat_to_rot(model.body_quat[b]), model.body_pos[b]
    return T


def _body_inertial(model, b):
    """``(m, com, I_com)`` of body ``b`` in the axes of the body frame (MuJoCo stores principal axes + ``iquat``)."""
    Ri = quat_to_rot(model.body_iquat[b])
    return float(model.body_mass[b]), model.body_ipos[b].copy(), Ri @ np.diag(model.body_inertia[b]) @ Ri.T


def _joint_displacement(model, j, qj):
    """4x4 motion of a body due to its single hinge/slide joint ``j`` at value ``qj`` (axis through ``jnt_pos``)."""
    a, jp = model.jnt_axis[j], model.jnt_pos[j]
    T = np.eye(4)
    if model.jnt_type[j] == 3:                                   # hinge
        R = _rodrigues(a, qj)
        T[:3, :3], T[:3, 3] = R, jp - R @ jp
    else:                                                        # slide
        T[:3, 3] = a * qj
    return T


def _subtree_off_path(model, b, path_set, T_to_frame, qpos_ref, acc):
    """Collect ``(m, com, I_com)`` of the descendants of body ``b`` that are not on the main path, expressed in the
    frame that ``T_to_frame`` (4x4: frame of ``b`` -> target frame) maps to."""
    for c in range(model.nbody):
        if model.body_parentid[c] != b or c == b or c in path_set:
            continue
        Tc = T_to_frame @ _body_T(model, c)
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == c:
                Tc = Tc @ _joint_displacement(model, j, qpos_ref[model.jnt_qposadr[j]])
        m, com, I = _body_inertial(model, c)
        if m > 0:
            R = Tc[:3, :3]
            acc.append((m, R @ com + Tc[:3, 3], R @ I @ R.T))
        _subtree_off_path(model, c, path_set, Tc, qpos_ref, acc)


def mujoco_dofs(model, tip_body):
    """Indices ``dofadr`` of the one-degree-of-freedom joints on the path from the world to ``tip_body``."""
    b = model.body(tip_body).id if isinstance(tip_body, str) else int(tip_body)
    out = []
    while b != 0:
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == b:
                out.append(int(model.jnt_dofadr[j]))
        b = int(model.body_parentid[b])
    return out[::-1]


def chain_from_mujoco(model, tip_body, name=None) -> SerialChain:
    """Read the serial chain ending at ``tip_body`` (name or id) from a compiled MuJoCo ``MjModel``.

    * Every body on the path from the world to the tip that carries one hinge or slide joint becomes a link.
    * Bodies without a joint are welded: before the first joint they are dropped (they are fixed to the world);
      after it their mass is merged into the preceding link with the parallel-axis theorem.
    * Bodies hanging off the path (for example the fingers of a gripper) are merged the same way, at the joint values
      ``model.qpos0``.
    * ``armature``, ``damping`` and ``frictionloss`` of the joints are copied into the chain.

    MuJoCo conventions handled here: orientations are scalar-first quaternions, ``body_ipos``/``body_iquat`` locate the
    centre of mass and the *principal* axes of the inertia, and the inertia of a body is about its centre of mass.
    """
    tip = model.body(tip_body).id if isinstance(tip_body, str) else int(tip_body)
    path = []
    b = tip
    while b != 0:
        path.append(b)
        b = int(model.body_parentid[b])
    path = path[::-1]
    path_set = set(path)
    links, dofs = [], []
    T_acc = np.eye(4)                  # frame of the previous jointed link -> frame of the current body
    for b in path:
        T_acc = T_acc @ _body_T(model, b)
        js = [j for j in range(model.njnt) if model.jnt_bodyid[j] == b]
        m, com, I = _body_inertial(model, b)
        parts = [(m, com, I)] if m > 0 else []
        if not js:
            if links:                  # weld to the previous link
                R = T_acc[:3, :3]
                links[-1]["parts"] += [(m, R @ com + T_acc[:3, 3], R @ I @ R.T)] if m > 0 else []
                extra = []
                _subtree_off_path(model, b, path_set, T_acc, model.qpos0, extra)
                links[-1]["parts"] += extra
            continue
        assert len(js) == 1 and model.jnt_type[js[0]] in (2, 3), "only single hinge or slide joints are supported on the path"
        j = js[0]
        jp = model.jnt_pos[j] if model.jnt_type[j] == 3 else np.zeros(3)       # a slide joint has no anchor point
        T_link = T_acc @ np.block([[np.eye(3), jp[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
        parts = [(m, com - jp, I)] if m > 0 else []
        extra = []
        shift = np.eye(4)
        shift[:3, 3] = -jp
        _subtree_off_path(model, b, path_set, shift, model.qpos0, extra)
        parts += extra
        links.append(dict(R=T_link[:3, :3], p=T_link[:3, 3], axis=model.jnt_axis[j], kind="R" if model.jnt_type[j] == 3 else "P",
                          parts=parts, name=model.body(b).name, dof=int(model.jnt_dofadr[j])))
        dofs.append(int(model.jnt_dofadr[j]))
        T_acc = shift                  # children are placed relative to the body frame, which sits -jp from the link frame
    out = []
    for L in links:
        m, com, I = combine_bodies(L["parts"]) if L["parts"] else (0.0, np.zeros(3), np.zeros((3, 3)))
        out.append(Link(m, com, I, L["R"], L["p"], L["axis"], L["kind"], L["name"]))
    d = np.array(dofs, int)
    return SerialChain(out, armature=model.dof_armature[d], damping=model.dof_damping[d],
                       frictionloss=model.dof_frictionloss[d], name=name or f"mujoco:{model.body(tip).name}")

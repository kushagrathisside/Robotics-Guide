"""Interaction control: environments, force/impedance/admittance laws and small simulators.

This module supports RMC Chapter 5 (*Force and Interaction Control*). Everything here is plain
NumPy so that every formula can be read, tested and compared with MuJoCo (see
:mod:`rgkit.interaction_mujoco`).

Sign conventions (used throughout Chapter 5)
--------------------------------------------
* ``F_ext`` is the force exerted **by the environment on the robot**.
* ``f`` is the **contact force**, the force exerted **by the robot on the environment**: ``f = -F_ext``
  along the contact normal. Pressing into a surface therefore gives ``f > 0``.
* In the one-degree-of-freedom models the robot moves along ``+x`` and the wall occupies ``x > x_w``,
  so ``F_ext = -f`` and a command ``u > 0`` pushes into the wall.
* Tasks in the plane use ``(x, y)`` and a 2R arm with joint angles ``q = (q1, q2)``.

Contents
--------
1. environment models                 :class:`Wall`, :class:`Surface`
2. natural and artificial constraints :class:`ConstraintFrame`, :func:`selection_from_constraints`
3. one-degree-of-freedom controllers  :class:`PositionPD`, :class:`Impedance1D`, :class:`Admittance1D`,
                                       :class:`ForceControl1D`, :class:`ForceVelocityLoop1D`, :func:`simulate_1dof`
4. linear analysis                    :func:`contact_force_position_control`, :func:`force_loop_spectral_radius`,
                                       :func:`max_stable_force_gain`, :func:`admittance_loop_spectral_radius`
5. planar 2R arm and task-space laws  :class:`Planar2R`, :func:`impedance_wrench`, :func:`hybrid_wrench`,
                                       :func:`simulate_2r`
6. stiffness and compliance           :func:`cartesian_stiffness`, :func:`cartesian_compliance`, :func:`ellipse_points`
7. series elastic actuation           :func:`simulate_sea_impact`, :func:`sea_force_loop_spectral_radius`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.linalg import expm

__all__ = [
    "Wall", "Surface", "ConstraintFrame", "selection_from_constraints", "rotate_selection",
    "PositionPD", "Impedance1D", "Admittance1D", "ForceControl1D", "ForceVelocityLoop1D", "simulate_1dof",
    "series_stiffness", "contact_force_position_control", "contact_force_impedance", "contact_force_admittance",
    "force_loop_spectral_radius", "max_stable_force_gain", "admittance_loop_spectral_radius",
    "Planar2R", "impedance_wrench", "hybrid_wrench", "hybrid_torque_joint_space", "line_trajectory",
    "OperationalSpace2R", "simulate_2r",
    "cartesian_stiffness", "cartesian_compliance", "ellipse_points", "force_ellipse_points",
    "simulate_sea_impact", "sea_force_loop_spectral_radius",
]


# ============================================================================
# 1. Environment models
# ============================================================================
@dataclass(frozen=True)
class Wall:
    """Unilateral spring-damper wall for the one-dimensional models.

    The robot moves along ``+x``; the wall surface is at ``x_w`` and the material occupies ``x > x_w``.
    :meth:`force` returns the contact force ``f >= 0`` exerted by the robot on the wall (equal and
    opposite to the force on the robot). The wall cannot pull (no adhesion), so the force is clipped at zero.
    """

    x_w: float = 0.0       # surface position (m)
    k: float = 1.0e4       # stiffness k_e (N/m)
    b: float = 0.0         # damping b_e (N s/m)

    def penetration(self, x: float) -> float:
        return max(0.0, x - self.x_w)

    def force(self, x: float, v: float = 0.0) -> float:
        if x <= self.x_w:
            return 0.0
        return max(0.0, self.k * (x - self.x_w) + self.b * v)


@dataclass(frozen=True)
class Surface:
    """A horizontal surface ``y = y_s(x)`` for planar tasks (robot above it, normal ``+y``).

    The environment force on the tool is ``F_ext = f_n n + f_t t`` where ``f_n = max(0, k d + b d_dot) >= 0``
    acts along the local unit normal ``n`` and ``d`` is the penetration depth measured along the normal.
    Coulomb friction ``f_t = -mu f_n tanh(v_t / v_s)`` opposes the tangential velocity ``v_t``.
    ``profile`` is an optional function ``x -> y_s`` (default: a flat surface at ``y0``).
    """

    y0: float = 0.0
    k: float = 2.0e4
    b: float = 20.0
    mu: float = 0.2
    v_s: float = 0.01
    profile: Callable[[float], float] | None = None
    slope: Callable[[float], float] | None = None   # d y_s / d x (needed only if ``profile`` is given)

    def height(self, x: float) -> float:
        return self.y0 if self.profile is None else float(self.profile(x))

    def tangent_normal(self, x: float):
        s = 0.0 if self.profile is None else (float(self.slope(x)) if self.slope else _numdiff(self.profile, x))
        t = np.array([1.0, s]) / np.hypot(1.0, s)
        n = np.array([-s, 1.0]) / np.hypot(1.0, s)
        return t, n

    def force_on_tool(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Environment force on the tool at position ``p`` with velocity ``v`` (2-vectors)."""
        t, n = self.tangent_normal(p[0])
        d = (self.height(p[0]) - p[1]) * n[1]            # penetration measured along the normal
        if d <= 0.0:
            return np.zeros(2)
        fn = max(0.0, self.k * d - self.b * float(v @ n))
        ft = -self.mu * fn * np.tanh(float(v @ t) / self.v_s)
        return fn * n + ft * t


def _numdiff(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


# ============================================================================
# 2. Natural and artificial constraints
# ============================================================================
_DOF_NAMES = ["x", "y", "z", "rx", "ry", "rz"]


@dataclass
class ConstraintFrame:
    """Natural and artificial constraints of a task in its constraint frame {C} (Mason, 1981).

    ``natural`` and ``artificial`` map each of the six degrees of freedom ``x, y, z, rx, ry, rz`` to
    ``('v', value)`` (a velocity is specified) or ``('f', value)`` (a force/torque is specified).
    The defining rules are checked by :meth:`validate`: along every DOF the environment fixes *exactly one* of
    velocity and force (natural), and the controller must then specify the *other* one (artificial).
    """

    name: str
    natural: dict
    artificial: dict

    def validate(self):
        for d in _DOF_NAMES:
            n, a = self.natural[d][0], self.artificial[d][0]
            if n == a:
                raise ValueError(f"{self.name}: DOF {d} has natural and artificial constraints of the same kind ({n})")
        return True

    def selection_diagonal(self):
        """1 where the controller commands motion (artificial = velocity), 0 where it commands force."""
        return np.array([1.0 if self.artificial[d][0] == "v" else 0.0 for d in _DOF_NAMES])

    def admissible_twist_mask(self):
        """True where the task allows motion (natural constraint is a force, usually zero)."""
        return np.array([self.natural[d][0] == "f" for d in _DOF_NAMES])

    def reciprocal(self, twist, wrench, tol=1e-9):
        """Instantaneous power ``wrench . twist`` of an admissible twist and a constraint wrench vanishes."""
        return abs(float(np.dot(twist, wrench))) < tol


def selection_from_constraints(cf: ConstraintFrame, dofs=None):
    """Diagonal selection matrix ``S`` (1 = position/velocity controlled) of a constraint frame."""
    d = cf.selection_diagonal()
    if dofs is not None:
        d = d[list(dofs)]
    return np.diag(d)


def rotate_selection(S, R):
    """Express a selection matrix defined in the constraint frame {C} in the base frame: ``R S R^T``.

    ``R`` is the rotation of {C} relative to the base frame (its columns are the axes of {C} in the base).
    """
    S, R = np.asarray(S, float), np.asarray(R, float)
    return R @ S @ R.T


# ============================================================================
# 3. One-degree-of-freedom controllers and simulator
# ============================================================================
def _as_traj(x):
    return x if callable(x) else (lambda t, x=x: x)


@dataclass
class PositionPD:
    """``u = kp (x_d - x) + kd (v_d - v) + m a_d`` (m a_d only if ``m`` is given: model feed-forward)."""
    kp: float
    kd: float
    x_d: float | Callable = 0.0
    m: float | None = None
    h: float = 1e-5

    def reset(self):
        self._xd = _as_traj(self.x_d)

    def update(self, t, x, v, f, Ts):
        xd = self._xd(t)
        vd = (self._xd(t + self.h) - self._xd(t - self.h)) / (2 * self.h)
        ad = (self._xd(t + self.h) - 2 * xd + self._xd(t - self.h)) / self.h ** 2
        return self.kp * (xd - x) + self.kd * (vd - v) + (self.m * ad if self.m is not None else 0.0)


@dataclass
class Impedance1D:
    """Target dynamics ``m_d (x'' - x_d'') + b_d (x' - x_d') + k_d (x - x_d) = F_ext = -f``.

    The robot is ``m x'' = u + F_ext``. With ``use_force=True`` the measured contact force is used to
    shape the inertia (full impedance control); with ``use_force=False`` the controller has no force
    measurement and the apparent inertia stays ``m`` (``m_d`` is ignored).
    """
    m: float
    m_d: float
    b_d: float
    k_d: float
    x_d: float | Callable = 0.0
    use_force: bool = True
    h: float = 1e-5

    def reset(self):
        self._xd = _as_traj(self.x_d)

    def update(self, t, x, v, f, Ts):
        xd = self._xd(t)
        vd = (self._xd(t + self.h) - self._xd(t - self.h)) / (2 * self.h)
        ad = (self._xd(t + self.h) - 2 * xd + self._xd(t - self.h)) / self.h ** 2
        e, ed = x - xd, v - vd
        if self.use_force:
            # x'' = a_d + (F_ext - b_d e' - k_d e)/m_d,  F_ext = -f,  u = m x'' - F_ext
            acc = ad + (-f - self.b_d * ed - self.k_d * e) / self.m_d
            return self.m * acc + f
        return self.m * ad - self.b_d * ed - self.k_d * e


@dataclass
class Admittance1D:
    """Position-controlled robot with an outer admittance loop.

    The virtual body ``m_a (x_c'' - x_d'') + b_a (x_c' - x_d') + k_a (x_c - x_d) = F_ext = -f`` is integrated
    at the control rate; the position-controlled robot (inner loop ``kp, kd``) tracks ``x_c``.
    """
    m_a: float
    b_a: float
    k_a: float
    kp: float
    kd: float
    x_d: float | Callable = 0.0
    m: float | None = None          # optional model feed-forward of the inner loop (m * x_c'')
    x_c0: float | None = None

    def reset(self):
        self._xd = _as_traj(self.x_d)
        self.xc = None
        self.vc = 0.0

    def update(self, t, x, v, f, Ts):
        if getattr(self, "xc", None) is None:
            self._xd = _as_traj(self.x_d)
            self.xc, self.vc = (x if self.x_c0 is None else self.x_c0), 0.0
        xd = self._xd(t)
        ac = (-f - self.b_a * self.vc - self.k_a * (self.xc - xd)) / self.m_a
        self.vc += Ts * ac                          # semi-implicit Euler for the virtual body
        self.xc += Ts * self.vc
        u = self.kp * (self.xc - x) + self.kd * (self.vc - v)
        return u + (self.m * ac if self.m is not None else 0.0)


@dataclass
class ForceControl1D:
    """Direct force control: ``u = f_d + k_f (f_d - f) + k_i int (f_d - f) - k_v v`` (feed-forward ``f_d`` optional)."""
    f_d: float | Callable
    k_f: float
    k_i: float = 0.0
    k_v: float = 0.0
    feedforward: bool = True
    i_max: float = 1e9

    def reset(self):
        self._fd = _as_traj(self.f_d)
        self.I = 0.0

    def update(self, t, x, v, f, Ts):
        if not hasattr(self, "I"):
            self.reset()
        e = self._fd(t) - f
        self.I = float(np.clip(self.I + Ts * e, -self.i_max, self.i_max))
        return (self._fd(t) if self.feedforward else 0.0) + self.k_f * e + self.k_i * self.I - self.k_v * v


@dataclass
class ForceVelocityLoop1D:
    """Force loop around an inner velocity loop: ``v_ref = k_f e + k_i int e``, ``u = k_v (v_ref - v)``."""
    f_d: float | Callable
    k_f: float            # (m/s)/N
    k_v: float            # N s/m  (inner velocity loop gain)
    k_i: float = 0.0

    def reset(self):
        self._fd = _as_traj(self.f_d)
        self.I = 0.0

    def update(self, t, x, v, f, Ts):
        if not hasattr(self, "I"):
            self.reset()
        e = self._fd(t) - f
        self.I += Ts * e
        return self.k_v * (self.k_f * e + self.k_i * self.I - v)


def simulate_1dof(controller, wall: Wall, m=1.0, c=0.0, x0=-0.01, v0=0.0, T=1.0, dt=1e-4, Ts=1e-3,
                  force_delay=0, f_noise=0.0, seed=0, u_max=None):
    """Simulate ``m x'' + c x' = u - f`` against ``wall`` with a sampled-data controller.

    The controller (see the classes above) is called every ``Ts`` seconds with the sampled state and the
    contact-force measurement delayed by ``force_delay`` samples (plus optional Gaussian noise); its output is
    held constant (zero-order hold) between samples. The plant is integrated with semi-implicit Euler at step
    ``dt`` (``Ts`` must be a multiple of ``dt``).

    Returns a dict with ``t, x, v, f`` (true contact force), ``f_meas`` and ``u`` at every integration step.
    """
    rng = np.random.default_rng(seed)
    n_sub = int(round(Ts / dt))
    n_steps = int(round(T / dt))
    controller.reset()
    x, v, u = x0, v0, 0.0
    hist = [0.0] * (force_delay + 1)
    out = {k: np.empty(n_steps) for k in ("t", "x", "v", "f", "f_meas", "u")}
    fm = 0.0
    for k in range(n_steps):
        t = k * dt
        f = wall.force(x, v)
        if k % n_sub == 0:
            hist.append(f)
            fm = hist.pop(0) + (f_noise * rng.standard_normal() if f_noise else 0.0)
            u = controller.update(t, x, v, fm, Ts)
            if u_max is not None:
                u = float(np.clip(u, -u_max, u_max))
        v += dt * (u - c * v - f) / m
        x += dt * v
        out["t"][k], out["x"][k], out["v"][k], out["f"][k], out["f_meas"][k], out["u"][k] = t, x, v, f, fm, u
    return out


# ============================================================================
# 4. Linear analysis
# ============================================================================
def series_stiffness(*ks):
    """Stiffness of springs in series: ``1 / sum(1/k_i)``."""
    return 1.0 / sum(1.0 / k for k in ks)


def contact_force_position_control(k_s, k_e, x_d, x_w):
    """Steady contact force of a stiffness-``k_s`` position servo (target ``x_d``) on a wall (``k_e``, surface ``x_w``).

    ``f = k_s k_e / (k_s + k_e) (x_d - x_w)`` for ``x_d > x_w``, else 0.
    """
    return series_stiffness(k_s, k_e) * max(0.0, x_d - x_w)


def contact_force_impedance(k_d, k_e, x_d, x_w, f_off=0.0):
    """Steady contact force of an impedance controller; ``f_off`` is a force offset in the target dynamics."""
    return series_stiffness(k_d, k_e) * max(0.0, x_d - x_w) + f_off * k_e / (k_d + k_e)


def _zoh(A, B, Ts):
    """Exact zero-order-hold discretisation of ``z' = A z + B u`` over one sample."""
    n, m = B.shape
    blk = np.zeros((n + m, n + m))
    blk[:n, :n], blk[:n, n:] = A * Ts, B * Ts
    E = expm(blk)
    return E[:n, :n], E[:n, n:]


def _history_shift(Acl, base, delay, new_force_row):
    """Fill the delay line of past force samples: states ``base .. base+delay-1`` hold ``f[k-1] .. f[k-delay]``."""
    if delay >= 1:
        Acl[base] = new_force_row
        for j in range(1, delay):
            Acl[base + j, base + j - 1] = 1.0


def force_loop_spectral_radius(m, k_e, k_f, k_v=0.0, Ts=1e-3, delay=0, law="direct", k_vi=None, b_e=0.0):
    """Spectral radius of the sampled-data force loop in contact (stable iff < 1).

    Plant: ``m x'' = u - k_e x - b_e x'`` (in contact, ``x`` measured from the surface), zero-order-hold input,
    sampled at ``Ts``. The contact force ``f = k_e x + b_e x'`` is measured with ``delay`` samples of delay.

    * ``law="direct"``:  ``u[k] = -(k_f f_meas[k] + k_v v[k])`` (the ``(1+k_f) f_d`` offset does not affect stability);
    * ``law="velocity_loop"``: ``u[k] = k_vi (k_f (f_d - f_meas[k]) - v[k])`` i.e. an outer force loop with gain
      ``k_f`` ((m/s)/N) around an inner velocity loop of gain ``k_vi`` (N s/m). Pass ``k_vi``.
    """
    A = np.array([[0.0, 1.0], [-k_e / m, -b_e / m]])
    Ad, Bd = _zoh(A, np.array([[0.0], [1.0 / m]]), Ts)
    n = 2 + delay
    if law == "direct":
        g_f, g_v = k_f, k_v
    elif law == "velocity_loop":
        g_f, g_v = k_vi * k_f, k_vi
    else:
        raise ValueError(law)
    Acl = np.zeros((n, n))
    Acl[:2, :2] = Ad
    f_now = np.zeros(n)
    f_now[0], f_now[1] = k_e, b_e                       # f[k] = k_e x[k] + b_e v[k]
    f_meas = f_now if delay == 0 else np.eye(n)[n - 1]   # oldest history state is f[k-delay]
    for i in range(2):
        Acl[i, :] += -Bd[i, 0] * (g_f * f_meas + g_v * np.eye(n)[1])
    _history_shift(Acl, 2, delay, f_now)
    return float(np.max(np.abs(np.linalg.eigvals(Acl))))


def max_stable_force_gain(m, k_e, k_v=0.0, Ts=1e-3, delay=0, law="direct", k_vi=None, b_e=0.0, hi=1e4):
    """Largest force gain ``k_f`` with spectral radius < 1 (bisection on ``(0, hi)``).

    ``k_f`` is dimensionless for ``law="direct"`` and in (m/s)/N for ``law="velocity_loop"``. Returns ``hi`` if the
    loop is stable on the whole interval and ``0`` if it is unstable already for a vanishing gain. Assumes the
    stable set is an interval starting at zero (true for all cases used in the notebooks).
    """
    ok = lambda kf: force_loop_spectral_radius(m, k_e, kf, k_v, Ts, delay, law, k_vi, b_e) < 1.0 - 1e-9
    if ok(hi):
        return hi
    if not ok(hi * 1e-9):
        return 0.0
    a, b = hi * 1e-9, hi
    for _ in range(80):
        mid = 0.5 * (a + b)
        a, b = (mid, b) if ok(mid) else (a, mid)
    return a


def contact_force_admittance(k_a, k_in, k_e, x_d, x_w):
    """Steady contact force of the admittance scheme: virtual spring ``k_a``, inner position loop ``k_in``, wall ``k_e``
    are three springs in series."""
    return series_stiffness(k_a, k_in, k_e) * max(0.0, x_d - x_w)


def admittance_loop_spectral_radius(m, k_e, m_a, b_a, k_a, kp, kd, Ts=1e-3, delay=0, b_e=0.0):
    """Spectral radius of the sampled admittance loop in contact (stable iff < 1).

    Robot ``m x'' = u - k_e x - b_e x'`` with inner position loop ``u = kp (x_c - x) + kd (v_c - v)`` (ZOH, ``Ts``);
    virtual body ``m_a x_c'' + b_a x_c' + k_a x_c = -f_meas`` (target ``x_d = 0``), ``f_meas = k_e x + b_e x'``
    delayed by ``delay`` samples, integrated by semi-implicit Euler exactly as in :class:`Admittance1D`
    (without the model feed-forward). State: ``(x, v, x_c, v_c, f[k-1], ..., f[k-delay])``.
    """
    A = np.array([[0.0, 1.0], [-k_e / m, -b_e / m]])
    Ad, Bd = _zoh(A, np.array([[0.0], [1.0 / m]]), Ts)
    n = 4 + delay
    E = np.eye(n)
    f_now = k_e * E[0] + b_e * E[1]
    f_meas = f_now if delay == 0 else E[n - 1]
    ac = (-f_meas - b_a * E[3] - k_a * E[2]) / m_a                 # virtual acceleration at sample k (row on the state)
    vc_next = E[3] + Ts * ac
    xc_next = E[2] + Ts * vc_next
    u = kp * (xc_next - E[0]) + kd * (vc_next - E[1])
    Acl = np.zeros((n, n))
    for i in range(2):
        Acl[i] = Ad[i, 0] * E[0] + Ad[i, 1] * E[1] + Bd[i, 0] * u
    Acl[2], Acl[3] = xc_next, vc_next
    _history_shift(Acl, 4, delay, f_now)
    return float(np.max(np.abs(np.linalg.eigvals(Acl))))


# ============================================================================
# 5. Planar 2R arm, task-space laws and contact simulator
# ============================================================================
@dataclass
class Planar2R:
    """Planar two-link arm in a vertical plane (``y`` up) with gravity ``g`` (set ``g=0`` for a horizontal plane).

    ``lc1, lc2`` are the distances from each joint to the link's centre of mass; ``I1, I2`` the link inertias
    about the centre of mass; ``cj`` is viscous joint friction (N m s/rad).
    """
    l1: float = 0.4
    l2: float = 0.4
    m1: float = 2.0
    m2: float = 1.5
    lc1: float = 0.2
    lc2: float = 0.2
    I1: float = 0.02
    I2: float = 0.015
    g: float = 9.81
    cj: float = 0.2

    # --- kinematics ---------------------------------------------------------
    def fk(self, q):
        q1, q12 = q[0], q[0] + q[1]
        return np.array([self.l1 * np.cos(q1) + self.l2 * np.cos(q12), self.l1 * np.sin(q1) + self.l2 * np.sin(q12)])

    def jac(self, q):
        s1, c1, s12, c12 = np.sin(q[0]), np.cos(q[0]), np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        return np.array([[-self.l1 * s1 - self.l2 * s12, -self.l2 * s12],
                         [self.l1 * c1 + self.l2 * c12, self.l2 * c12]])

    def jac_dot(self, q, qd):
        s1, c1, s12, c12 = np.sin(q[0]), np.cos(q[0]), np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        w1, w12 = qd[0], qd[0] + qd[1]
        return np.array([[-self.l1 * c1 * w1 - self.l2 * c12 * w12, -self.l2 * c12 * w12],
                         [-self.l1 * s1 * w1 - self.l2 * s12 * w12, -self.l2 * s12 * w12]])

    def ik(self, p, elbow=+1):
        """Closed-form inverse kinematics; ``elbow=+1`` gives ``q2 > 0``."""
        x, y = p
        c2 = (x * x + y * y - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        if abs(c2) > 1 + 1e-12:
            raise ValueError("target out of reach")
        q2 = elbow * np.arccos(np.clip(c2, -1, 1))
        q1 = np.arctan2(y, x) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2))
        return np.array([q1, q2])

    # --- dynamics -----------------------------------------------------------
    def M(self, q):
        c2 = np.cos(q[1])
        a = self.m1 * self.lc1 ** 2 + self.m2 * (self.l1 ** 2 + self.lc2 ** 2) + self.I1 + self.I2
        b = self.m2 * self.l1 * self.lc2
        d = self.m2 * self.lc2 ** 2 + self.I2
        return np.array([[a + 2 * b * c2, d + b * c2], [d + b * c2, d]])

    def C(self, q, qd):
        h = self.m2 * self.l1 * self.lc2 * np.sin(q[1])
        return np.array([[-h * qd[1], -h * (qd[0] + qd[1])], [h * qd[0], 0.0]])

    def G(self, q):
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.cos(q[0]) + self.m2 * self.lc2 * self.g * np.cos(q[0] + q[1])
        g2 = self.m2 * self.lc2 * self.g * np.cos(q[0] + q[1])
        return np.array([g1, g2])

    def bias(self, q, qd):
        return self.C(q, qd) @ qd + self.G(q)

    def forward_dynamics(self, q, qd, tau, F_ext=None):
        """``q'' = M^-1 (tau + J^T F_ext - C qd - G - cj qd)``."""
        F_ext = np.zeros(2) if F_ext is None else F_ext
        rhs = tau + self.jac(q).T @ F_ext - self.bias(q, qd) - self.cj * qd
        return np.linalg.solve(self.M(q), rhs)

    # --- operational-space quantities ----------------------------------------
    def task_inertia(self, q):
        J = self.jac(q)
        return np.linalg.inv(J @ np.linalg.solve(self.M(q), J.T))

    def task_bias(self, q, qd):
        """``mu`` (Coriolis/centrifugal) and ``p`` (gravity) of ``Lambda x'' + mu + p = F + F_ext``."""
        J, M = self.jac(q), self.M(q)
        Lam = self.task_inertia(q)
        mu = Lam @ J @ np.linalg.solve(M, self.C(q, qd) @ qd) - Lam @ self.jac_dot(q, qd) @ qd
        p = Lam @ J @ np.linalg.solve(M, self.G(q))
        return Lam, mu, p


def impedance_wrench(Lam, mu, p, e, e_dot, a_d, M_d, B_d, K_d, F_ext=None):
    """Task-space impedance law. ``e = x - x_d``.

    With ``F_ext`` (measured force of the environment on the robot) the target dynamics
    ``M_d (e'' ) + B_d e' + K_d e = F_ext`` are enforced exactly (full impedance control):

        F = Lam [a_d + M_d^-1 (F_ext - B_d e' - K_d e)] + mu + p - F_ext.

    With ``F_ext=None`` and ``M_d=None`` the apparent inertia stays ``Lam`` and the law reduces to
    ``F = Lam a_d + mu + p - B_d e' - K_d e``.
    """
    if F_ext is None:
        return Lam @ a_d + mu + p - B_d @ e_dot - K_d @ e
    acc = a_d + np.linalg.solve(M_d, F_ext - B_d @ e_dot - K_d @ e)
    return Lam @ acc + mu + p - F_ext


def hybrid_wrench(S, Lam, mu, p, a_pos, f_cmd):
    """Operational-space hybrid law: ``F = S Lam S a_pos + (I - S) f_cmd + mu + p``.

    ``S`` is the diagonal selection matrix (1 = position-controlled axes), ``a_pos`` the commanded acceleration
    of the position-controlled axes (``a_d + K_v e' + K_p e`` form) and ``f_cmd`` the force that the force
    controller wants the robot to exert on the environment (already including its feed-forward).
    Projecting ``Lam`` with ``S`` on both sides keeps the force axes free of inertial coupling, so the
    force-controlled components of ``F`` come from ``f_cmd`` alone (plus the compensation ``mu + p``);
    for a rigid constraint along the force axes this is exactly the dynamics of the free axes.
    """
    I = np.eye(S.shape[0])
    return S @ Lam @ S @ a_pos + (I - S) @ f_cmd + mu + p


def hybrid_torque_joint_space(arm, q, qd, S, e_x, e_xd, f_cmd, Kp, Kv):
    """Craig-style hybrid law in joint space (position part through ``J^-1``, force part through ``J^T``).

    ``tau = M (Kp J^-1 S e_x + Kv J^-1 S e_xd) + C qd + G + J^T (I - S) f_cmd`` with ``e_x = x_d - x``,
    ``e_xd = v_d - v`` (task-space errors) and scalar or diagonal gains ``Kp, Kv`` (joint-space servo gains).
    """
    J = arm.jac(q)
    Ji = np.linalg.inv(J)
    I = np.eye(2)
    qdd_cmd = Kp * (Ji @ (S @ e_x)) + Kv * (Ji @ (S @ e_xd))
    return arm.M(q) @ qdd_cmd + arm.bias(q, qd) + J.T @ ((I - S) @ f_cmd)


# --- sampled controllers for the planar 2R arm (used with :func:`simulate_2r`) ------------------------------
def line_trajectory(p0, p1, duration, t0=0.0):
    """Minimum-jerk straight-line trajectory ``t -> (x_d, v_d, a_d)`` from ``p0`` to ``p1``."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0

    def traj(t):
        s = np.clip((t - t0) / duration, 0.0, 1.0)
        pos = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        vel = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / duration if 0 < s < 1 else 0.0
        acc = (60 * s - 180 * s ** 2 + 120 * s ** 3) / duration ** 2 if 0 < s < 1 else 0.0
        return p0 + d * pos, d * vel, d * acc
    return traj


@dataclass
class OperationalSpace2R:
    """Sampled task-space controller for :class:`Planar2R`.

    ``mode``:

    * ``"position"``  : ``F = Lam (a_d + Kd e' + Kp e) + mu + p``  (stiff motion control, S = I);
    * ``"impedance"`` : full impedance control with inertia shaping (needs the force measurement) when ``M_d`` is
      given, otherwise the force-free variant ``F = Lam a_d + mu + p - B_d e' - K_d e``; here ``Kp`` is the
      stiffness ``K_d`` and ``Kd`` the damping ``B_d``;
    * ``"hybrid"``    : :func:`hybrid_wrench` with selection matrix ``S``, force target ``f_d`` (force that the robot
      exerts on the environment) and a PI force law ``f_cmd = f_d + Kf (f_d - f) + Ki int (f_d - f)``.

    ``traj(t)`` returns ``(x_d, v_d, a_d)``; ``Fe`` handed to :meth:`update` is the environment force on the tool.
    """
    arm: Planar2R
    traj: Callable
    mode: str = "position"
    Kp: np.ndarray = field(default_factory=lambda: np.diag([2500.0, 2500.0]))
    Kd: np.ndarray = field(default_factory=lambda: np.diag([100.0, 100.0]))
    M_d: np.ndarray | None = None
    S: np.ndarray = field(default_factory=lambda: np.diag([1.0, 0.0]))
    f_d: np.ndarray = field(default_factory=lambda: np.zeros(2))
    Kf: float = 0.0
    Ki: float = 0.0
    approach_v: float = 0.0          # >0: guarded move along ``approach_axis`` until |F| > f_thresh (hybrid mode)
    approach_axis: int = 1
    approach_dir: float = -1.0
    f_thresh: float = 1.0
    i_max: float = 1e9

    def reset(self):
        self.I = np.zeros(2)
        self.in_contact = self.approach_v <= 0.0

    def update(self, t, q, qd, Fe, Ts):
        arm = self.arm
        if not hasattr(self, "I"):
            self.reset()
        J = arm.jac(q)
        x, v = arm.fk(q), J @ qd
        x_d, v_d, a_d = self.traj(t)
        e, ed = x - x_d, v - v_d
        Lam, mu, p = arm.task_bias(q, qd)
        if self.mode == "position":
            F = Lam @ (a_d - self.Kd @ ed - self.Kp @ e) + mu + p
        elif self.mode == "impedance":
            F = impedance_wrench(Lam, mu, p, e, ed, a_d, self.M_d, self.Kd, self.Kp,
                                 F_ext=Fe if self.M_d is not None else None)
        elif self.mode == "hybrid":
            if not self.in_contact and abs(Fe[self.approach_axis]) > self.f_thresh:
                self.in_contact = True                       # guarded move ends: switch the axis to force control
            if self.in_contact:
                f = -Fe                                      # force exerted by the robot on the environment
                err = (np.eye(2) - self.S) @ (self.f_d - f)
                self.I = np.clip(self.I + Ts * err, -self.i_max, self.i_max)
                f_cmd = self.f_d + self.Kf * err + self.Ki * self.I
                a_pos = a_d - self.Kd @ ed - self.Kp @ e
                F = hybrid_wrench(self.S, Lam, mu, p, a_pos, f_cmd)
            else:                                            # free-space approach: velocity control on the axis
                ax = self.approach_axis
                e, ed, a_d = e.copy(), ed.copy(), a_d.copy()
                e[ax], a_d[ax] = 0.0, 0.0
                ed[ax] = v[ax] - self.approach_dir * self.approach_v
                F = Lam @ (a_d - self.Kd @ ed - self.Kp @ e) + mu + p
        else:
            raise ValueError(self.mode)
        return J.T @ F


def simulate_2r(arm: Planar2R, controller, surface: Surface | None, q0, qd0=None, T=2.0, dt=5e-4, Ts=1e-3,
                F_ext_extra=None, tau_max=None):
    """Simulate a planar 2R arm touching ``surface`` under a sampled controller.

    ``controller.update(t, q, qd, F_meas, Ts)`` returns the joint torque ``tau`` (held between samples);
    ``F_meas`` is the force of the environment on the tool (what a wrist sensor would report, in the base
    frame). ``F_ext_extra(t)`` optionally adds a known external force on the tool. Semi-implicit Euler integration.
    Returns a dict of arrays: ``t, q, qd, p, v, F (env), tau``.
    """
    q = np.array(q0, float)
    qd = np.zeros(2) if qd0 is None else np.array(qd0, float)
    n_sub, n_steps = int(round(Ts / dt)), int(round(T / dt))
    controller.reset()
    tau = np.zeros(2)
    rec = {k: np.empty((n_steps, 2)) for k in ("q", "qd", "p", "v", "F", "tau")}
    rec["t"] = np.arange(n_steps) * dt
    for k in range(n_steps):
        J = arm.jac(q)
        p, v = arm.fk(q), J @ qd
        F = surface.force_on_tool(p, v) if surface is not None else np.zeros(2)
        if F_ext_extra is not None:
            F = F + F_ext_extra(k * dt)
        if k % n_sub == 0:
            tau = controller.update(k * dt, q, qd, F, Ts)
            if tau_max is not None:
                tau = np.clip(tau, -tau_max, tau_max)
        qdd = arm.forward_dynamics(q, qd, tau, F)
        qd = qd + dt * qdd
        q = q + dt * qd
        rec["q"][k], rec["qd"][k], rec["p"][k], rec["v"][k], rec["F"][k], rec["tau"][k] = q, qd, p, v, F, tau
    return rec


# ============================================================================
# 6. Stiffness and compliance
# ============================================================================
def cartesian_stiffness(J, K_q):
    """``K_c = J^-T K_q J^-1`` for a square, non-singular Jacobian (small displacements, ``dJ`` neglected)."""
    Ji = np.linalg.inv(J)
    return Ji.T @ K_q @ Ji


def cartesian_compliance(J, K_q):
    """``C_c = J K_q^-1 J^T`` (valid also for non-square ``J``; the inverse of ``K_c`` when ``J`` is square)."""
    return J @ np.linalg.solve(K_q, J.T)


def ellipse_points(A, center=(0.0, 0.0), n=200, level=1.0):
    """Points of the ellipse ``{ d : d^T A d = level }`` around ``center`` (``A`` symmetric positive definite)."""
    w, V = np.linalg.eigh(A)
    th = np.linspace(0, 2 * np.pi, n)
    circ = np.vstack([np.cos(th), np.sin(th)]) * np.sqrt(level)
    pts = V @ (circ / np.sqrt(w)[:, None])
    return pts.T + np.asarray(center)


def force_ellipse_points(J, tau_max, center=(0.0, 0.0), n=200):
    """Force ellipse ``{ F : |J^T F| <= tau_max }`` boundary (unit joint-torque norm scaled by ``tau_max``)."""
    A = J @ J.T / tau_max ** 2
    return ellipse_points(A, center, n)


# ============================================================================
# 7. Series elastic actuation
# ============================================================================
def simulate_sea_impact(v0, m_motor, m_load, k_s, wall: Wall, T=0.2, dt=1e-5, series=True):
    """Impact of an actuator-driven body with a wall, motor free-wheeling (zero command).

    * ``series=True``: motor mass ``m_motor`` connected to the output mass ``m_load`` through a spring ``k_s``
      (series elastic actuator); the output touches the wall.
    * ``series=False``: rigid transmission, a single mass ``m_motor + m_load``.

    Both start at the wall surface (``x = x_w``) moving with ``v0`` into it. Returns ``(t, f)`` (contact force)
    and the peak force.
    """
    n = int(round(T / dt))
    t = np.arange(n) * dt
    f = np.empty(n)
    if series:
        xm, vm, xl, vl = wall.x_w - 1e-6, v0, wall.x_w - 1e-6, v0
        for k in range(n):
            fs = k_s * (xm - xl)
            fc = wall.force(xl, vl)
            vm += dt * (-fs) / m_motor
            vl += dt * (fs - fc) / m_load
            xm += dt * vm
            xl += dt * vl
            f[k] = fc
    else:
        x, v = wall.x_w - 1e-6, v0
        M = m_motor + m_load
        for k in range(n):
            fc = wall.force(x, v)
            v += dt * (-fc) / M
            x += dt * v
            f[k] = fc
    return t, f, float(f.max())


def sea_force_loop_spectral_radius(m_motor, m_load, k_s, k_e, k_f, k_v, Ts=1e-3, delay=0, b_e=0.0):
    """Spectral radius of a sampled force loop around a series elastic actuator in contact (stable iff < 1).

    Motor mass ``m_motor`` (force input ``u``), spring ``k_s`` between motor and output mass ``m_load``, wall of
    stiffness ``k_e`` and damping ``b_e``. Control ``u[k] = -(k_f f_s[k-delay] + k_v v_m[k])`` where the *spring force*
    ``f_s = k_s (x_m - x_l)`` is the measured quantity (the SEA's built-in force sensor). The state is
    ``(x_m, v_m, x_l, v_l, f_s[k-1], ..., f_s[k-delay])``; the input is held for one sample (exact ZOH).
    """
    A = np.zeros((4, 4))
    A[0, 1] = 1.0
    A[2, 3] = 1.0
    A[1, 0], A[1, 2] = -k_s / m_motor, k_s / m_motor
    A[3, 0], A[3, 2], A[3, 3] = k_s / m_load, -(k_s + k_e) / m_load, -b_e / m_load
    Ad, Bd = _zoh(A, np.array([[0.0], [1.0 / m_motor], [0.0], [0.0]]), Ts)
    n = 4 + delay
    E = np.eye(n)
    fs_now = k_s * (E[0] - E[2])
    fs_meas = fs_now if delay == 0 else E[n - 1]
    u = -(k_f * fs_meas + k_v * E[1])
    Acl = np.zeros((n, n))
    for i in range(4):
        Acl[i, :4] = Ad[i]
        Acl[i] += Bd[i, 0] * u
    _history_shift(Acl, 4, delay, fs_now)
    return float(np.max(np.abs(np.linalg.eigvals(Acl))))

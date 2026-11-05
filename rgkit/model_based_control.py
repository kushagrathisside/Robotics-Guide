"""Dynamics, simulators and model-based controllers for RMC Chapter 3.

The module contains everything that the notebooks of *Robot Motion Control*, Chapter 3 (Lyapunov stability and
model-based control) need and that is worth testing separately from the prose:

* :class:`Planar2R` - the textbook planar two-link arm in a **vertical** plane, written in terms of five lumped
  dynamic parameters ``(alpha, beta, delta, gamma1, gamma2)``: inertia matrix ``M(q)``, Coriolis matrix ``C(q, qd)``
  (Christoffel form, so that ``Mdot - 2 C`` is skew-symmetric), gravity vector ``g(q)``, energy, forward dynamics.
* :func:`simulate` - a fixed-step RK4 simulator for :class:`Planar2R` with zero-order-hold sampling, pure delay,
  torque saturation, measurement noise and unmodelled viscous/Coulomb friction.
* Controllers with a common signature ``controller(t, q, qd) -> tau``: :func:`pd_control`,
  :func:`pd_gravity_control`, :func:`independent_joint_control`, :class:`ComputedTorque`.
* Reference trajectories (:func:`quintic_reference`, :func:`sine_reference`, :func:`constant_reference`) returning
  ``ref(t) -> (q_d, qd_d, qdd_d)``.
* Lyapunov helpers for the PD controller (:func:`pd_lyapunov`) and gain helpers (:func:`gains_from_wn_zeta`).
* A thin bridge to **MuJoCo** (:class:`MuJoCoDynamics`, :func:`mass_matrix`, :func:`configure_torque_control`,
  :func:`describe_flags`, :func:`simulate_mujoco`) so that exactly the same controllers can drive a real robot
  model, using MuJoCo's own ``M(q)`` and ``qfrc_bias`` as the controller's model.

Conventions
-----------
Angles in radians. The arm moves in the plane ``x`` (right) / ``y`` (up); gravity acts along ``-y``;
``q1`` is the angle of link 1 from the ``+x`` axis, ``q2`` the angle of link 2 relative to link 1 (both
counter-clockwise), so that ``q = (0, 0)`` is the fully stretched, horizontal arm. All controller models expose
``mass(q)``, ``bias(q, qd)`` (the vector ``C(q, qd) qd + g(q)``) and ``gravity(q)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "Planar2R", "Sim", "simulate", "rk4_step",
    "constant_reference", "quintic_reference", "sine_reference",
    "pd_control", "pd_gravity_control", "independent_joint_control", "ComputedTorque",
    "pd_lyapunov", "gains_from_wn_zeta", "settling_time", "rms", "workspace_inertia_range",
    "MuJoCoDynamics", "mass_matrix", "configure_torque_control", "describe_flags", "simulate_mujoco",
]


# ---------------------------------------------------------------------------------------------------------------
#  Planar two-link arm
# ---------------------------------------------------------------------------------------------------------------
class Planar2R:
    """Planar 2R arm in a vertical plane with five lumped dynamic parameters.

    With link masses ``m_i``, lengths ``l_i``, centre-of-mass distances ``lc_i`` along the link and moments of
    inertia ``I_i`` about the centre of mass,

    ``alpha = I1 + I2 + m1 lc1^2 + m2 (l1^2 + lc2^2)``, ``beta = m2 l1 lc2``, ``delta = I2 + m2 lc2^2``,
    ``gamma1 = m1 lc1 + m2 l1``, ``gamma2 = m2 lc2``, and

    ``M = [[alpha + 2 beta c2, delta + beta c2], [delta + beta c2, delta]]``,
    ``C = [[-beta s2 qd2, -beta s2 (qd1 + qd2)], [beta s2 qd1, 0]]``,
    ``g = g0 [gamma1 c1 + gamma2 c12, gamma2 c12]``.

    The model is *linear in the parameters*: ``tau = Y(q, qd, qdd) @ pi`` with ``pi = (alpha, beta, delta,
    gamma1, gamma2)`` (see :meth:`regressor`), the property exploited by adaptive control.

    Unmodelled friction, used only by the simulator, is stored in ``viscous`` and ``coulomb`` (per joint).
    """

    def __init__(self, alpha, beta, delta, gamma1, gamma2, g0=9.81, l1=1.0, l2=1.0,
                 viscous=(0.0, 0.0), coulomb=(0.0, 0.0)):
        self.alpha, self.beta, self.delta = float(alpha), float(beta), float(delta)
        self.gamma1, self.gamma2 = float(gamma1), float(gamma2)
        self.g0, self.l1, self.l2 = float(g0), float(l1), float(l2)
        self.viscous = np.asarray(viscous, float)
        self.coulomb = np.asarray(coulomb, float)
        self.nq = 2

    # -- construction ------------------------------------------------------------------------------------------
    @classmethod
    def from_links(cls, m1, m2, l1, l2, lc1=None, lc2=None, I1=None, I2=None, g0=9.81, **kw):
        """Build from physical link data. Defaults: uniform rods (``lc = l/2``, ``I = m l^2 / 12``)."""
        lc1 = l1 / 2 if lc1 is None else lc1
        lc2 = l2 / 2 if lc2 is None else lc2
        I1 = m1 * l1 ** 2 / 12 if I1 is None else I1
        I2 = m2 * l2 ** 2 / 12 if I2 is None else I2
        return cls(alpha=I1 + I2 + m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2), beta=m2 * l1 * lc2,
                   delta=I2 + m2 * lc2 ** 2, gamma1=m1 * lc1 + m2 * l1, gamma2=m2 * lc2,
                   g0=g0, l1=l1, l2=l2, **kw)

    @property
    def pi(self):
        """Parameter vector ``(alpha, beta, delta, gamma1, gamma2)``."""
        return np.array([self.alpha, self.beta, self.delta, self.gamma1, self.gamma2])

    def copy(self, **changes):
        d = dict(alpha=self.alpha, beta=self.beta, delta=self.delta, gamma1=self.gamma1, gamma2=self.gamma2,
                 g0=self.g0, l1=self.l1, l2=self.l2, viscous=self.viscous.copy(), coulomb=self.coulomb.copy())
        d.update(changes)
        return Planar2R(**d)

    def with_payload(self, mass):
        """Model of the arm carrying a point mass ``mass`` at the tip of link 2 (at distance ``l2``)."""
        return self.copy(alpha=self.alpha + mass * (self.l1 ** 2 + self.l2 ** 2),
                         beta=self.beta + mass * self.l1 * self.l2,
                         delta=self.delta + mass * self.l2 ** 2,
                         gamma1=self.gamma1 + mass * self.l1, gamma2=self.gamma2 + mass * self.l2)

    def scaled(self, mass_factor=1.0, gravity_factor=None):
        """All masses/inertias multiplied by ``mass_factor`` (``gravity_factor`` defaults to the same factor)."""
        gf = mass_factor if gravity_factor is None else gravity_factor
        return self.copy(alpha=self.alpha * mass_factor, beta=self.beta * mass_factor,
                         delta=self.delta * mass_factor, gamma1=self.gamma1 * gf, gamma2=self.gamma2 * gf)

    def without_friction(self):
        return self.copy(viscous=(0.0, 0.0), coulomb=(0.0, 0.0))

    # -- rigid-body dynamics -----------------------------------------------------------------------------------
    def mass(self, q):
        c2 = math.cos(q[1])
        m12 = self.delta + self.beta * c2
        return np.array([[self.alpha + 2 * self.beta * c2, m12], [m12, self.delta]])

    M = mass

    def coriolis_matrix(self, q, qd):
        """Christoffel-symbol matrix ``C(q, qd)``; ``Mdot - 2 C`` is skew-symmetric."""
        h = -self.beta * math.sin(q[1])
        return np.array([[h * qd[1], h * (qd[0] + qd[1])], [-h * qd[0], 0.0]])

    C = coriolis_matrix

    def coriolis(self, q, qd):
        """The vector ``C(q, qd) qd``."""
        h = -self.beta * math.sin(q[1])
        return np.array([h * qd[1] * (2 * qd[0] + qd[1]), -h * qd[0] ** 2])

    def mass_dot(self, q, qd):
        """Time derivative ``Mdot`` of the inertia matrix along a motion."""
        s = -self.beta * math.sin(q[1]) * qd[1]
        return np.array([[2 * s, s], [s, 0.0]])

    def gravity(self, q):
        c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
        return self.g0 * np.array([self.gamma1 * c1 + self.gamma2 * c12, self.gamma2 * c12])

    def gravity_jacobian(self, q):
        """``dg/dq`` (symmetric: g is the gradient of the potential energy)."""
        s1, s12 = math.sin(q[0]), math.sin(q[0] + q[1])
        a, b = -self.g0 * (self.gamma1 * s1 + self.gamma2 * s12), -self.g0 * self.gamma2 * s12
        return np.array([[a, b], [b, b]])

    def bias(self, q, qd):
        return self.coriolis(q, qd) + self.gravity(q)

    def kinetic_energy(self, q, qd):
        qd = np.asarray(qd, float)
        return 0.5 * qd @ self.mass(q) @ qd

    def potential_energy(self, q):
        """Potential energy with the zero level on the horizontal through the shoulder (``y = 0``)."""
        return self.g0 * (self.gamma1 * math.sin(q[0]) + self.gamma2 * math.sin(q[0] + q[1]))

    def energy(self, q, qd):
        return self.kinetic_energy(q, qd) + self.potential_energy(q)

    def friction(self, qd, eps=1e-2):
        """Unmodelled joint friction ``b qd + fc tanh(qd / eps)`` (smooth Coulomb law)."""
        qd = np.asarray(qd, float)
        return self.viscous * qd + self.coulomb * np.tanh(qd / eps)

    def acceleration(self, q, qd, tau, friction=True):
        """Forward dynamics ``qdd = M^-1 (tau - C qd - g - friction)``."""
        rhs = np.asarray(tau, float) - self.bias(q, qd)
        if friction:
            rhs = rhs - self.friction(qd)
        return np.linalg.solve(self.mass(q), rhs)

    def inverse_dynamics(self, q, qd, qdd):
        """``tau = M qdd + C qd + g`` (no friction)."""
        return self.mass(q) @ np.asarray(qdd, float) + self.bias(q, qd)

    def regressor(self, q, qd, qdd):
        """``Y`` with ``tau = Y @ pi`` for ``pi = (alpha, beta, delta, gamma1, gamma2)``."""
        c1, c2, c12, s2 = math.cos(q[0]), math.cos(q[1]), math.cos(q[0] + q[1]), math.sin(q[1])
        a1, a2 = qdd[0], qdd[1]
        v1, v2 = qd[0], qd[1]
        g0 = self.g0
        return np.array([
            [a1, 2 * c2 * a1 + c2 * a2 - s2 * (2 * v1 * v2 + v2 ** 2), a2, g0 * c1, g0 * c12],
            [0.0, c2 * a1 + s2 * v1 ** 2, a1 + a2, 0.0, g0 * c12],
        ])

    # -- kinematics ---------------------------------------------------------------------------------------------
    def forward_kinematics(self, q):
        """Positions of the elbow and of the tip (the end of link 2), each a 2-vector."""
        elbow = np.array([self.l1 * math.cos(q[0]), self.l1 * math.sin(q[0])])
        tip = elbow + self.l2 * np.array([math.cos(q[0] + q[1]), math.sin(q[0] + q[1])])
        return elbow, tip

    def jacobian(self, q):
        s1, c1, s12, c12 = math.sin(q[0]), math.cos(q[0]), math.sin(q[0] + q[1]), math.cos(q[0] + q[1])
        return np.array([[-self.l1 * s1 - self.l2 * s12, -self.l2 * s12], [self.l1 * c1 + self.l2 * c12, self.l2 * c12]])


# ---------------------------------------------------------------------------------------------------------------
#  Integration and simulation
# ---------------------------------------------------------------------------------------------------------------
def rk4_step(f: Callable, x, u, dt: float):
    """One classical Runge-Kutta step of ``xdot = f(x, u)`` with ``u`` held constant over the step."""
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


@dataclass
class Sim:
    """Result of :func:`simulate` / :func:`simulate_mujoco`. Arrays have one row per output sample."""
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray           # torque actually applied to the plant
    q_ref: np.ndarray
    qd_ref: np.ndarray
    qdd_ref: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def e(self):
        return self.q_ref - self.q

    @property
    def ed(self):
        return self.qd_ref - self.qd


def _as_ref(ref):
    if ref is None:
        return None
    if callable(ref):
        return ref
    q = np.asarray(ref, float)
    return constant_reference(q)


def simulate(plant: Planar2R, controller: Callable, q0, T: float, dt: float = 1e-3, qd0=(0.0, 0.0),
             ref=None, ctrl_dt: Optional[float] = None, delay: float = 0.0, tau_max=None,
             noise_q: float = 0.0, noise_qd: float = 0.0, seed: int = 0, friction: bool = True) -> Sim:
    """Simulate ``plant`` under ``controller(t, q, qd) -> tau`` with RK4 at step ``dt``.

    ``ctrl_dt`` (default ``dt``) is the controller sampling period: the torque is recomputed every ``ctrl_dt``
    seconds from the *measured* state and held (zero-order hold). ``delay`` (a multiple of ``dt``) postpones
    the application of each command. ``tau_max`` saturates each joint torque symmetrically. ``noise_q``,
    ``noise_qd`` are standard deviations of Gaussian measurement noise (rad, rad/s). ``friction=False`` ignores the
    plant's friction parameters. ``ref`` is only used for logging the reference.
    """
    n = int(round(T / dt))
    hold = max(1, int(round((ctrl_dt or dt) / dt)))
    nd = int(round(delay / dt))
    rng = np.random.default_rng(seed)
    ref = _as_ref(ref)
    q, qd = np.array(q0, float), np.array(qd0, float)
    tmax = None if tau_max is None else np.broadcast_to(np.asarray(tau_max, float), (2,))
    t_arr = np.arange(n + 1) * dt
    Q, QD, TAU = np.zeros((n + 1, 2)), np.zeros((n + 1, 2)), np.zeros((n + 1, 2))
    R = np.zeros((3, n + 1, 2))
    cmd_hist = []
    tau_cmd = np.zeros(2)

    def f(x, u):
        return np.concatenate([x[2:], plant.acceleration(x[:2], x[2:], u, friction=friction)])

    x = np.concatenate([q, qd])
    for k in range(n + 1):
        t = t_arr[k]
        if k % hold == 0:
            qm = x[:2] + (noise_q * rng.standard_normal(2) if noise_q else 0.0)
            qdm = x[2:] + (noise_qd * rng.standard_normal(2) if noise_qd else 0.0)
            tau_cmd = np.asarray(controller(t, qm, qdm), float)
            if tmax is not None:
                tau_cmd = np.clip(tau_cmd, -tmax, tmax)
        cmd_hist.append(tau_cmd)
        tau_app = cmd_hist[max(0, k - nd)] if nd else tau_cmd
        Q[k], QD[k], TAU[k] = x[:2], x[2:], tau_app
        if ref is not None:
            R[0, k], R[1, k], R[2, k] = ref(t)
        if k < n:
            x = rk4_step(f, x, tau_app, dt)
    return Sim(t_arr, Q, QD, TAU, R[0], R[1], R[2])


# ---------------------------------------------------------------------------------------------------------------
#  References
# ---------------------------------------------------------------------------------------------------------------
def constant_reference(q_d):
    """Set-point reference ``q_d(t) = const``."""
    q_d = np.asarray(q_d, float)
    zero = np.zeros_like(q_d)
    return lambda t: (q_d, zero, zero)


def quintic_reference(q_start, q_end, T):
    """Point-to-point quintic polynomial with zero velocity and acceleration at both ends (duration ``T``)."""
    q_start, q_end = np.asarray(q_start, float), np.asarray(q_end, float)
    dq = q_end - q_start

    def ref(t):
        s = min(max(t / T, 0.0), 1.0)
        p, dp, ddp = 10 * s**3 - 15 * s**4 + 6 * s**5, (30 * s**2 - 60 * s**3 + 30 * s**4) / T, \
            (60 * s - 180 * s**2 + 120 * s**3) / T**2
        if t >= T or t <= 0.0:
            dp = ddp = 0.0
        return q_start + p * dq, dp * dq, ddp * dq
    return ref


def sine_reference(q_center, amplitude, omega, phase=None):
    """``q_d(t) = q_center + A sin(omega t + phase)`` (per joint ``amplitude``, ``omega``, ``phase``)."""
    qc, A = np.asarray(q_center, float), np.asarray(amplitude, float)
    w = np.broadcast_to(np.asarray(omega, float), qc.shape)
    ph = np.zeros_like(qc) if phase is None else np.asarray(phase, float)

    def ref(t):
        a = w * t + ph
        return qc + A * np.sin(a), A * w * np.cos(a), -A * w**2 * np.sin(a)
    return ref


# ---------------------------------------------------------------------------------------------------------------
#  Controllers  (all have the signature  controller(t, q, qd) -> tau)
# ---------------------------------------------------------------------------------------------------------------
def _diag(K, n):
    K = np.asarray(K, float)
    return np.diag(np.broadcast_to(K, (n,))) if K.ndim <= 1 else K


def pd_control(Kp, Kd, ref):
    """Joint-space PD law ``tau = Kp (q_d - q) + Kd (qd_d - qd)`` (no model)."""
    ref = _as_ref(ref)

    def ctrl(t, q, qd):
        qr, qdr, _ = ref(t)
        Kp_, Kd_ = _diag(Kp, len(q)), _diag(Kd, len(q))
        return Kp_ @ (qr - q) + Kd_ @ (qdr - qd)
    ctrl.name = "PD"
    return ctrl


def pd_gravity_control(model, Kp, Kd, ref, compensate="current"):
    """PD law plus gravity compensation.

    ``compensate="current"``: ``tau = Kp e + Kd ed + g(q)`` (Takegaki-Arimoto);
    ``compensate="desired"``: ``tau = Kp e + Kd ed + g(q_d)`` (a constant feed-forward for set-points).
    """
    ref = _as_ref(ref)

    def ctrl(t, q, qd):
        qr, qdr, _ = ref(t)
        Kp_, Kd_ = _diag(Kp, len(q)), _diag(Kd, len(q))
        g = model.gravity(q) if compensate == "current" else model.gravity(qr)
        return Kp_ @ (qr - q) + Kd_ @ (qdr - qd) + g
    ctrl.name = "PD+g" if compensate == "current" else "PD+g(q_d)"
    return ctrl


def independent_joint_control(Jbar, Kp, Kv, ref, model=None):
    """Decentralised control-law partitioning with a *constant* diagonal inertia ``Jbar`` per joint.

    ``tau_i = Jbar_i (qdd_d,i + Kv_i ed_i + Kp_i e_i)`` (+ ``g_i(q)`` if a ``model`` is supplied). Coupling and the
    configuration dependence of ``M`` are left to the feedback to reject.
    """
    ref = _as_ref(ref)
    Jbar = np.asarray(Jbar, float)

    def ctrl(t, q, qd):
        qr, qdr, qddr = ref(t)
        n = len(q)
        a = qddr + _diag(Kv, n) @ (qdr - qd) + _diag(Kp, n) @ (qr - q)
        tau = Jbar * a
        return tau + model.gravity(q) if model is not None else tau
    ctrl.name = "IJ+g" if model is not None else "IJ"
    return ctrl


class ComputedTorque:
    """Computed-torque (inverse-dynamics) controller.

    ``tau = M(q) a + C(q, qd) qd + g(q)`` with ``a = qdd_d + Kv ed + Kp e (+ Ki * integral of e)``, where the model
    ``M, C, g`` is an *estimate* that may differ from the plant. ``model`` needs ``mass(q)`` and ``bias(q, qd)``.
    ``Ki`` adds integral action; the integral is accumulated from the call times, so create a fresh controller
    (or call :meth:`reset`) for every simulation.
    """

    def __init__(self, model, Kp, Kv, ref, Ki=None, name="CT"):
        self.model, self.Kp, self.Kv, self.Ki, self.ref, self.name = model, Kp, Kv, Ki, _as_ref(ref), name
        self.reset()

    def reset(self):
        self._int, self._tlast = None, None

    def acceleration_command(self, t, q, qd):
        qr, qdr, qddr = self.ref(t)
        n = len(q)
        e, ed = qr - q, qdr - qd
        a = qddr + _diag(self.Kv, n) @ ed + _diag(self.Kp, n) @ e
        if self.Ki is not None:
            if self._int is None:
                self._int = np.zeros(n)
            elif t > self._tlast:
                self._int = self._int + e * (t - self._tlast)
            self._tlast = t
            a = a + _diag(self.Ki, n) @ self._int
        return a

    def __call__(self, t, q, qd):
        a = self.acceleration_command(t, q, qd)
        return self.model.mass(q) @ a + self.model.bias(q, qd)


# ---------------------------------------------------------------------------------------------------------------
#  Analysis helpers
# ---------------------------------------------------------------------------------------------------------------
def pd_lyapunov(model, q, qd, q_des, Kp):
    """``V = 1/2 qd' M(q) qd + 1/2 e' Kp e`` with ``e = q_des - q`` (energy-like Lyapunov function of PD+g)."""
    q, qd = np.asarray(q, float), np.asarray(qd, float)
    e = np.asarray(q_des, float) - q
    return 0.5 * qd @ model.mass(q) @ qd + 0.5 * e @ _diag(Kp, len(q)) @ e


def gains_from_wn_zeta(wn, zeta):
    """``Kp = wn^2``, ``Kv = 2 zeta wn`` for the unit-inertia error dynamics ``e'' + Kv e' + Kp e = 0``."""
    wn, zeta = np.asarray(wn, float), np.asarray(zeta, float)
    return wn**2, 2 * zeta * wn


def settling_time(t, y, y_final, tol=0.02, y_scale=None):
    """First time after which ``|y - y_final| <= tol * y_scale`` for good (``y_scale`` defaults to ``|y[0]-y_final|``)."""
    y = np.asarray(y, float)
    scale = abs(y[0] - y_final) if y_scale is None else y_scale
    outside = np.where(np.abs(y - y_final) > tol * scale)[0]
    if len(outside) == 0:
        return float(t[0])
    return float(t[min(outside[-1] + 1, len(t) - 1)])


def rms(x):
    """Root-mean-square of an array (over all entries)."""
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x**2)))


def workspace_inertia_range(model, n=181):
    """Min and max over ``q2`` of the smallest/largest eigenvalue of ``M`` (``M`` depends on ``q2`` only)."""
    lam = np.array([np.linalg.eigvalsh(model.mass([0.0, q2])) for q2 in np.linspace(-np.pi, np.pi, n)])
    return lam[:, 0].min(), lam[:, 1].max()


# ---------------------------------------------------------------------------------------------------------------
#  MuJoCo bridge
# ---------------------------------------------------------------------------------------------------------------
def mass_matrix(model, data):
    """Dense joint-space inertia matrix ``M(q)`` of a MuJoCo model (version independent; needs ``mj_forward``)."""
    import mujoco
    M = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, M)            # recent releases
    except TypeError:
        mujoco.mj_fullM(model, M, data.qM)         # earlier 3.x releases
    return M


class MuJoCoDynamics:
    """MuJoCo's own rigid-body dynamics presented as a controller *model*: ``mass``, ``bias``, ``gravity``.

    A private ``MjData`` is used, so the plant's data is untouched. ``bias(q, qd)`` is ``qfrc_bias`` (the Coriolis,
    centrifugal and gravity forces, **excluding** joint damping and friction); ``include_damping=True`` adds the
    linear joint damping ``dof_damping * qd``, which MuJoCo treats as a *passive* force.
    """

    def __init__(self, model, include_damping=False):
        import mujoco
        self._mj = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.nq = model.nv
        self.include_damping = include_damping

    def _set(self, q, qd):
        self.data.qpos[:] = q
        self.data.qvel[:] = qd
        self._mj.mj_forward(self.model, self.data)

    def mass(self, q):
        self._set(q, np.zeros(self.model.nv))
        return mass_matrix(self.model, self.data)

    def bias(self, q, qd):
        self._set(q, qd)
        b = self.data.qfrc_bias.copy()
        return b + self.model.dof_damping * np.asarray(qd) if self.include_damping else b

    def gravity(self, q):
        self._set(q, np.zeros(self.model.nv))
        return self.data.qfrc_bias.copy()

    def mass_and_bias(self, q, qd):
        self._set(q, qd)
        b = self.data.qfrc_bias.copy()
        if self.include_damping:
            b = b + self.model.dof_damping * np.asarray(qd)
        return mass_matrix(self.model, self.data), b

    def inverse_dynamics(self, q, qd, qdd):
        """``qfrc_inverse`` from ``mj_inverse``: the force that produces ``qdd`` (includes damping and friction terms)."""
        self.data.qpos[:], self.data.qvel[:], self.data.qacc[:] = q, qd, qdd
        self._mj.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse.copy()


def configure_torque_control(model, contacts=False, energy=False):
    """Prepare a MuJoCo model for torque control through ``data.qfrc_applied``.

    Switches off the built-in actuators (``mjDSBL_ACTUATION``) so that only the torque we write acts, optionally
    switches off contact (``mjDSBL_CONTACT``; self-collision of the arm is irrelevant here) and optionally switches
    on energy computation (``mjENBL_ENERGY``). Gravity and the model's damping, friction and armature are *not*
    changed. Returns a dict with the previous flag words so that they can be restored.
    """
    import mujoco
    old = dict(disable=int(model.opt.disableflags), enable=int(model.opt.enableflags))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    if not contacts:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    if energy:
        model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    return old


def describe_flags(model) -> dict:
    """Return (and print-friendly) a dict of everything that affects a torque-controlled simulation."""
    import mujoco
    dis = int(model.opt.disableflags)
    ena = int(model.opt.enableflags)
    names_dis = [n for n in ("CONSTRAINT", "EQUALITY", "FRICTIONLOSS", "LIMIT", "CONTACT", "SPRING", "DAMPER",
                             "GRAVITY", "CLAMPCTRL", "WARMSTART", "FILTERPARENT", "ACTUATION", "REFSAFE", "SENSOR",
                             "MIDPHASE", "EULERDAMP", "AUTORESET", "NATIVECCD", "ISLAND")
                 if hasattr(mujoco.mjtDisableBit, "mjDSBL_" + n) and dis & int(getattr(mujoco.mjtDisableBit, "mjDSBL_" + n))]
    names_ena = [n for n in ("OVERRIDE", "ENERGY", "FWDINV", "INVDISCRETE", "MULTICCD", "ISLAND")
                 if hasattr(mujoco.mjtEnableBit, "mjENBL_" + n) and ena & int(getattr(mujoco.mjtEnableBit, "mjENBL_" + n))]
    integrators = {int(mujoco.mjtIntegrator.mjINT_EULER): "Euler", int(mujoco.mjtIntegrator.mjINT_RK4): "RK4",
                   int(mujoco.mjtIntegrator.mjINT_IMPLICIT): "implicit", int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST): "implicitfast"}
    return dict(timestep=float(model.opt.timestep), integrator=integrators.get(int(model.opt.integrator), "?"),
                gravity=tuple(np.round(model.opt.gravity, 3)), disabled=names_dis or ["(none)"],
                enabled=names_ena or ["(none)"], dof_damping=tuple(np.round(model.dof_damping, 4)),
                dof_armature=tuple(np.round(model.dof_armature, 4)),
                dof_frictionloss=tuple(np.round(model.dof_frictionloss, 4)), n_actuators=int(model.nu))


def simulate_mujoco(model, data, controller: Callable, T: float, q0, qd0=None, ref=None, ctrl_every: int = 1,
                    log_extra: Optional[Callable] = None, delay_steps: int = 0, tau_max=None) -> Sim:
    """Run ``controller(t, q, qd) -> tau`` on a MuJoCo model through ``data.qfrc_applied``.

    The controller is evaluated every ``ctrl_every`` physics steps and held (zero-order hold); ``delay_steps`` physics
    steps of pure delay can be added. ``log_extra(model, data)`` may return a dict of scalars/arrays that are
    stacked into ``Sim.extra``. The caller is responsible for the model flags (see :func:`configure_torque_control`).
    """
    import mujoco
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q0
    if qd0 is not None:
        data.qvel[:] = qd0
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    n = int(round(T / dt))
    nv = model.nv
    ref = _as_ref(ref)
    t_arr = np.arange(n + 1) * dt
    Q, QD, TAU = np.zeros((n + 1, nv)), np.zeros((n + 1, nv)), np.zeros((n + 1, nv))
    R = np.zeros((3, n + 1, nv))
    extras, hist = [], []
    tau_cmd = np.zeros(nv)
    for k in range(n + 1):
        t = data.time
        if k % ctrl_every == 0:
            tau_cmd = np.asarray(controller(t, data.qpos.copy(), data.qvel.copy()), float)
            if tau_max is not None:
                tau_cmd = np.clip(tau_cmd, -np.asarray(tau_max), np.asarray(tau_max))
        hist.append(tau_cmd)
        tau_app = hist[max(0, k - delay_steps)]
        Q[k], QD[k], TAU[k] = data.qpos, data.qvel, tau_app
        if ref is not None:
            R[0, k], R[1, k], R[2, k] = ref(t)
        if log_extra is not None:
            extras.append(log_extra(model, data))
        data.qfrc_applied[:] = tau_app
        if k < n:
            mujoco.mj_step(model, data)
    data.qfrc_applied[:] = 0.0
    out = Sim(t_arr, Q, QD, TAU, R[0], R[1], R[2])
    if extras:
        out.extra["log"] = np.array(extras)
    return out

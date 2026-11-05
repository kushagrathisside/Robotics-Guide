"""Independent-joint control: models, controllers and simulation helpers (RMC Chapter 2).

The module collects everything the notebooks of ``RMC/02-independent-joint-control`` need, so that the
notebooks can stay short and readable.  It has four parts.

1. **Single-joint model** — gearing and DC-motor formulas (:func:`effective_inertia`,
   :func:`effective_damping`), the second-order facts needed to tune a PD or PID joint
   (:func:`second_order_params`, :func:`pd_gains`, :func:`pid_gains_from_poles`, :func:`step_metrics`),
   reference trajectories (:func:`cubic`, :func:`quintic`), a nonlinear one-link plant
   (:class:`Joint`) and a discrete PID(F) controller bank with torque saturation and anti-windup
   (:class:`PID`).  :func:`simulate` closes the loop with a zero-order hold, optional computation delay
   and optional encoder quantisation.
2. **Multi-joint coupling** — the planar two-link arm (:class:`Planar2R`) with the inertia matrix, Coriolis
   terms and gravity derived in the notebooks, plus gearing and rotor inertia.
3. **Flexibility and sampling** — the two-mass (flexible-joint) model (:class:`TwoMass`) and the exact
   closed-loop poles of a sampled PD loop (:func:`sampled_pd_poles`).
4. **MuJoCo helpers** (imported lazily) — run an explicit joint controller on a MuJoCo arm by writing
   ``data.qfrc_applied`` (:func:`mj_run`), read ``M(q)`` and ``g(q)`` from the simulator.

Conventions
-----------
* Gear ratio ``r > 1`` is a *reduction*: motor angle = ``r`` × joint angle, joint torque = ``r`` × motor
  torque.  Motor-side effective inertia is ``Jm + Jl / r**2``; load-side it is ``Jl + r**2 * Jm``.
* Gravity of a one-link joint is ``tau_g = mgl * cos(q)`` with ``q`` measured from the horizontal; for
  the planar 2R arm both angles are measured counter-clockwise in the vertical x-z plane (``g = 0``
  switches gravity off, which is the horizontal-plane arm).
* All angles are in radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "effective_inertia", "effective_damping", "reflected_to_load", "second_order_params", "pd_gains",
    "overshoot", "peak_time", "settling_time", "step_metrics", "pid_poles", "pid_gains_from_poles",
    "ki_limit", "cubic", "quintic", "Joint", "PID", "SimResult", "simulate", "Planar2R", "TwoMass",
    "sampled_pd_poles", "mj_mass_matrix", "mj_gravity", "mj_silence_actuators", "mj_run",
    "bd_box", "bd_sum", "bd_arrow", "bd_text", "save_gif",
]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1. Gearing, DC motor and second-order formulas
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def effective_inertia(Jm, Jl, r):
    """Inertia seen at the **motor shaft**: ``Jm + Jl / r**2`` (``r`` is the reduction ratio)."""
    return Jm + Jl / r**2


def reflected_to_load(Jm, Jl, r):
    """Inertia seen at the **joint (load) shaft**: ``Jl + r**2 * Jm``."""
    return Jl + r**2 * Jm


def effective_damping(Bm, Bl, r, Kt=0.0, Kb=0.0, R=np.inf):
    """Viscous damping at the motor shaft: ``Bm + Bl / r**2 + Kt * Kb / R``.

    The last term is the electrical damping of a *voltage-driven* DC motor (back-EMF); it vanishes for
    a current (torque) drive, which is the case when ``R = inf``.
    """
    return Bm + Bl / r**2 + (Kt * Kb / R if np.isfinite(R) else 0.0)


def second_order_params(J, B, kp, kd=0.0):
    """Natural frequency and damping ratio of ``J q'' + (B + kd) q' + kp q = ...``: ``(wn, zeta)``."""
    wn = np.sqrt(kp / J)
    zeta = (B + kd) / (2.0 * np.sqrt(kp * J))
    return wn, zeta


def pd_gains(J, B, wn, zeta):
    """PD gains that place the closed-loop poles of ``J q'' + B q' = u`` at ``(wn, zeta)``.

    ``kp = J wn**2`` and ``kd = 2 zeta wn J - B`` (``kd`` is the *extra* damping beyond ``B``).
    """
    return J * wn**2, 2.0 * zeta * wn * J - B


def overshoot(zeta):
    """Percentage overshoot of the unit step response of a standard second-order system (0 <= zeta < 1)."""
    zeta = np.asarray(zeta, float)
    out = np.zeros_like(zeta)
    m = (zeta >= 0) & (zeta < 1)
    out[m] = 100.0 * np.exp(-np.pi * zeta[m] / np.sqrt(1.0 - zeta[m] ** 2))
    return float(out) if out.ndim == 0 else out


def peak_time(wn, zeta):
    """Time of the first peak of the step response, ``pi / (wn sqrt(1 - zeta**2))`` (underdamped)."""
    return np.pi / (wn * np.sqrt(1.0 - zeta**2))


def settling_time(wn, zeta, tol=0.02):
    """Approximate settling time ``-ln(tol) / (zeta wn)`` (the usual envelope estimate; 2 % -> ~4/(zeta wn))."""
    return -np.log(tol) / (zeta * wn)


def step_metrics(t, y, y_final=None, y0=None, tol=0.02):
    """Measure a step response: overshoot [%], 10-90 % rise time, settling time (``tol`` band), final error.

    ``y_final`` defaults to the commanded final value ``y[-1]``; pass the *reference* to also obtain the
    steady-state error ``y[-1] - y_final``.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    y0 = y[0] if y0 is None else y0
    yf = y[-1] if y_final is None else y_final
    amp = yf - y0
    s = (y - y0) / amp
    os_pct = max(0.0, (s.max() - 1.0) * 100.0)
    try:
        t10, t90 = t[np.argmax(s >= 0.1)], t[np.argmax(s >= 0.9)]
        rise = t90 - t10 if (s >= 0.9).any() else np.nan
    except ValueError:
        rise = np.nan
    out = np.abs(s - 1.0) > tol
    ts = t[-1] if out[-1] else (t[np.where(out)[0][-1] + 1] if out.any() else 0.0)
    return dict(overshoot=os_pct, rise=rise, settling=ts, final_error=float(y[-1] - yf))


def pid_poles(J, B, kp, ki, kd):
    """Closed-loop poles of ``J q'' + B q' = kp e + ki int(e) + kd e'`` (a unit-feedback loop)."""
    return np.roots([J, B + kd, kp, ki])


def pid_gains_from_poles(J, B, poles):
    """PID gains placing the three closed-loop poles of ``J q'' + B q' = u`` at ``poles``.

    Matching ``J s^3 + (B+kd) s^2 + kp s + ki`` with ``J (s-p1)(s-p2)(s-p3)`` gives
    ``kd = -J sum(p) - B``, ``kp = J sum_{i<j} p_i p_j``, ``ki = -J p1 p2 p3``.
    """
    p = np.asarray(poles, complex)
    c = np.poly(p).real                      # [1, a2, a1, a0]
    return J * c[2], J * c[3], J * c[1] - B   # kp, ki, kd


def ki_limit(J, B, kp, kd):
    """Routh limit: the PID loop is stable iff ``0 < ki < kp (B + kd) / J``."""
    return kp * (B + kd) / J


def cubic(t, q0, q1, T):
    """Cubic rest-to-rest polynomial: returns ``(q, qd, qdd)`` (arrays like ``t``; clamped outside ``[0, T]``)."""
    t = np.asarray(t, float)
    tt = np.clip(t, 0.0, T)
    s = tt / T
    d = q1 - q0
    q = q0 + d * (3 * s**2 - 2 * s**3)
    qd = d * (6 * s - 6 * s**2) / T
    qdd = d * (6 - 12 * s) / T**2
    inside = (t >= 0) & (t <= T)
    return q, np.where(inside, qd, 0.0), np.where(inside, qdd, 0.0)


def quintic(t, q0, q1, T):
    """Quintic rest-to-rest polynomial (zero velocity and acceleration at both ends): ``(q, qd, qdd)``.

    ``q(t) = q0 + (q1 - q0)(10 s^3 - 15 s^4 + 6 s^5)``, ``s = t/T``; peak velocity ``1.875 (q1-q0)/T``.
    """
    t = np.asarray(t, float)
    tt = np.clip(t, 0.0, T)
    s = tt / T
    d = np.asarray(q1, float) - np.asarray(q0, float)
    q = q0 + d * (10 * s**3 - 15 * s**4 + 6 * s**5)
    inside = (t >= 0) & (t <= T)
    qd = np.where(inside, d * (30 * s**2 - 60 * s**3 + 30 * s**4) / T, 0.0)
    qdd = np.where(inside, d * (60 * s - 180 * s**2 + 120 * s**3) / T**2, 0.0)
    return q, qd, qdd


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2. One-link plant, PID controller, simulation loop
# ─────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Joint:
    """A single joint ``J q'' + B q' + tau_c sign(q') + mgl cos(q) + d(t) = sat(u)``.

    Parameters
    ----------
    J : effective inertia [kg m^2]; ``B`` viscous damping [N m s/rad]; ``mgl`` gravity amplitude [N m]
    coulomb : Coulomb friction torque [N m] (smoothed with ``tanh``); ``u_max`` : actuator torque limit
    [N m]; ``disturbance`` : optional callable ``d(t)`` [N m].
    """

    J: float
    B: float = 0.0
    mgl: float = 0.0
    coulomb: float = 0.0
    u_max: float = np.inf
    disturbance: callable = None
    n: int = field(default=1, init=False)

    def gravity(self, q):
        return self.mgl * np.cos(q)

    def deriv(self, t, x, u):
        q, qd = float(x[0]), float(x[1])
        u = float(np.ravel(u)[0])
        u = self.u_max if u > self.u_max else (-self.u_max if u < -self.u_max else u)
        d = self.disturbance(t) if self.disturbance else 0.0
        fric = self.coulomb * math.tanh(qd / 1e-3) if self.coulomb else 0.0
        return np.array([qd, (u - self.B * qd - fric - self.mgl * math.cos(q) - d) / self.J])

    def measure(self, x):
        return np.atleast_1d(x[0]), np.atleast_1d(x[1])

    def applied(self, u):
        return np.clip(u, -self.u_max, self.u_max)


class PID:
    """Discrete PID controller bank (one gain per joint) with feed-forward, saturation and anti-windup.

    Control law, evaluated once per sample of length ``dt``::

        u = kp e + kd ed + I + J_ff qdd_ref + B_ff qd_ref + gravity(q)           (then clipped)

    with ``e = q_ref - q`` and ``ed = qd_ref - qd`` (the derivative acts on the *velocity error*, so a
    step in the set-point causes no derivative kick).  ``I`` is the integral *torque*, updated by forward
    Euler.  ``antiwindup`` selects

    * ``"none"``     : ``I += ki e dt`` always (the integrator winds up while the actuator saturates);
    * ``"clamp"``    : conditional integration — freeze ``I`` while the output is saturated and the error
      would push it further into saturation;
    * ``"backcalc"`` : back-calculation, ``I += (ki e + kaw (u_sat - u)) dt``.

    ``gravity`` is an optional callable ``g(q) -> torque`` added as a feed-forward, evaluated at the
    measured angle (``gravity_at="meas"``) or at the reference (``"ref"``).
    """

    def __init__(self, kp, ki=0.0, kd=0.0, dt=1e-3, u_min=-np.inf, u_max=np.inf, antiwindup="clamp",
                 kaw=None, J_ff=0.0, B_ff=0.0, gravity=None, gravity_at="meas", i_limit=np.inf):
        self.kp, self.ki, self.kd = (np.atleast_1d(np.asarray(v, float)) for v in (kp, ki, kd))
        self.dt, self.antiwindup = dt, antiwindup
        self.u_min, self.u_max = np.atleast_1d(np.asarray(u_min, float)), np.atleast_1d(np.asarray(u_max, float))
        self.kaw = np.atleast_1d(np.asarray(kaw if kaw is not None else 0.0, float))
        self.J_ff, self.B_ff = np.atleast_1d(np.asarray(J_ff, float)), np.atleast_1d(np.asarray(B_ff, float))
        self.gravity, self.gravity_at, self.i_limit = gravity, gravity_at, i_limit
        self.reset()

    def reset(self):
        n = max(len(self.kp), len(self.ki), len(self.kd))
        self.I = np.zeros(n)
        self.u_unsat = np.zeros(n)

    def __call__(self, q, qd, q_ref, qd_ref=0.0, qdd_ref=0.0):
        q, qd = np.atleast_1d(q), np.atleast_1d(qd)
        q_ref, qd_ref, qdd_ref = (np.atleast_1d(np.asarray(v, float)) for v in (q_ref, qd_ref, qdd_ref))
        e, ed = q_ref - q, qd_ref - qd
        u_ff = self.J_ff * qdd_ref + self.B_ff * qd_ref
        if self.gravity is not None:
            u_ff = u_ff + self.gravity(q if self.gravity_at == "meas" else q_ref)
        u = self.kp * e + self.kd * ed + self.I + u_ff
        self.u_unsat = u
        u_sat = np.clip(u, self.u_min, self.u_max)
        if self.antiwindup == "none":
            self.I = self.I + self.ki * e * self.dt
        elif self.antiwindup == "clamp":
            winding = ((u > self.u_max) & (e > 0)) | ((u < self.u_min) & (e < 0))
            self.I = self.I + np.where(winding, 0.0, self.ki * e * self.dt)
        elif self.antiwindup == "backcalc":
            self.I = self.I + (self.ki * e + self.kaw * (u_sat - u)) * self.dt
        else:
            raise ValueError(self.antiwindup)
        self.I = np.clip(self.I, -self.i_limit, self.i_limit)
        return u_sat


@dataclass
class SimResult:
    """Time histories returned by :func:`simulate` (arrays are squeezed for a single joint)."""

    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    u: np.ndarray        # torque actually applied by the actuator
    u_cmd: np.ndarray    # torque requested by the controller (before the actuator limit)
    q_ref: np.ndarray
    qd_ref: np.ndarray
    integral: np.ndarray = None

    @property
    def e(self):
        return self.q_ref - self.q


def simulate(plant, ctrl, ref, T, dt=1e-3, substeps=4, x0=None, delay=0, enc_res=None, vel_from_diff=False):
    """Close the loop around ``plant`` with the sampled controller ``ctrl``.

    Parameters
    ----------
    plant : object with ``deriv(t, x, u)`` and ``measure(x) -> (q, qd)`` (:class:`Joint`, :class:`Planar2R`, ...)
    ctrl : callable ``ctrl(q, qd, q_ref, qd_ref, qdd_ref) -> u`` (e.g. a :class:`PID`)
    ref : callable ``ref(t) -> (q_ref, qd_ref, qdd_ref)``; or a constant (set-point) number/array
    T, dt : duration and **sampling period** of the controller; the plant is integrated with RK4 and
        ``substeps`` sub-steps per sample while the command is held (zero-order hold)
    delay : computation/communication delay in whole samples (command computed at sample ``k`` is applied at ``k+delay``)
    enc_res : encoder resolution [rad]; the measured angle is rounded to a multiple of it
    vel_from_diff : estimate the velocity by a backward difference of the measured angle
    """
    if not callable(ref):
        const = np.atleast_1d(np.asarray(ref, float))
        ref = lambda t, c=const: (c, np.zeros_like(c), np.zeros_like(c))  # noqa: E731
    n_steps = int(round(T / dt))
    x = np.array(x0 if x0 is not None else np.zeros(2 * plant.n), float)
    h = dt / substeps
    queue = [np.zeros(plant.n)] * delay
    rec = {k: [] for k in ("t", "q", "qd", "u", "u_cmd", "qr", "qdr", "I")}
    q_prev = None
    if hasattr(ctrl, "reset"):
        ctrl.reset()
    for k in range(n_steps):
        t = k * dt
        q, qd = plant.measure(x)
        if enc_res:
            q = np.round(q / enc_res) * enc_res
        if vel_from_diff:
            qd = np.zeros_like(q) if q_prev is None else (q - q_prev) / dt
        q_prev = q.copy()
        qr, qdr, qddr = (np.atleast_1d(np.asarray(v, float)) for v in ref(t))
        u_cmd = np.atleast_1d(ctrl(q, qd, qr, qdr, qddr))
        queue.append(u_cmd)
        u_hold = queue.pop(0)
        rec["t"].append(t); rec["q"].append(q); rec["qd"].append(qd); rec["u_cmd"].append(u_cmd)
        rec["u"].append(plant.applied(u_hold)); rec["qr"].append(qr); rec["qdr"].append(qdr)
        rec["I"].append(getattr(ctrl, "I", np.zeros(1)).copy())
        for _ in range(substeps):                             # RK4 with held input
            tt = t + _ * h
            k1 = plant.deriv(tt, x, u_hold)
            k2 = plant.deriv(tt + h / 2, x + h / 2 * k1, u_hold)
            k3 = plant.deriv(tt + h / 2, x + h / 2 * k2, u_hold)
            k4 = plant.deriv(tt + h, x + h * k3, u_hold)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    sq = (lambda a: a[:, 0]) if plant.n == 1 else (lambda a: a)
    arr = {k: np.array(v) for k, v in rec.items()}
    return SimResult(arr["t"], sq(arr["q"]), sq(arr["qd"]), sq(arr["u"]), sq(arr["u_cmd"]), sq(arr["qr"]),
                     sq(arr["qdr"]), sq(arr["I"]))


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 3. The planar two-link arm
# ─────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Planar2R:
    """Planar 2R arm moving in the vertical x-z plane (``g > 0``) or the horizontal plane (``g = 0``).

    Link ``i`` has mass ``m_i``, length ``l_i``, centre of mass at distance ``lc_i`` from its proximal joint
    and moment of inertia ``I_i`` about the centre of mass (axis normal to the plane).  Both joint angles
    are measured counter-clockwise; ``q1`` from the horizontal ``x`` axis, ``q2`` relative to link 1.
    ``Jm``, ``r`` and ``Bm`` are the rotor inertia, reduction ratio and rotor viscous friction of each
    motor; they add ``r_i**2 Jm_i`` to the diagonal of the inertia matrix as seen at the joint.

    Equations of motion ``M(q) qdd + C(q, qd) qd + g(q) = tau`` with (``c2 = cos q2``)::

        M11 = I1 + I2 + m1 lc1^2 + m2 (l1^2 + lc2^2 + 2 l1 lc2 c2),   M12 = I2 + m2 (lc2^2 + l1 lc2 c2),
        M22 = I2 + m2 lc2^2,
        C = h [[-qd2, -(qd1+qd2)], [qd1, 0]],   h = m2 l1 lc2 sin q2,
        g1 = (m1 lc1 + m2 l1) g cos q1 + m2 lc2 g cos(q1+q2),   g2 = m2 lc2 g cos(q1+q2).
    """

    m1: float = 2.0
    m2: float = 1.5
    l1: float = 1.0
    l2: float = 0.8
    lc1: float = 0.5
    lc2: float = 0.4
    I1: float = 2.0 / 12
    I2: float = 1.5 * 0.8**2 / 12
    g: float = 9.81
    Jm: tuple = (0.0, 0.0)
    r: tuple = (1.0, 1.0)
    Bm: tuple = (0.0, 0.0)
    tau_max: tuple = (np.inf, np.inf)
    n: int = field(default=2, init=False)

    # -- rigid-body terms ------------------------------------------------------------------------
    def M_link(self, q):
        c2 = np.cos(q[1])
        a = self.m2 * self.l1 * self.lc2
        M11 = self.I1 + self.I2 + self.m1 * self.lc1**2 + self.m2 * (self.l1**2 + self.lc2**2) + 2 * a * c2
        M12 = self.I2 + self.m2 * self.lc2**2 + a * c2
        M22 = self.I2 + self.m2 * self.lc2**2
        return np.array([[M11, M12], [M12, M22]])

    def M(self, q):
        """Inertia matrix at the joints, including the reflected rotor inertia ``r_i^2 Jm_i``."""
        return self.M_link(q) + np.diag(np.asarray(self.r, float) ** 2 * np.asarray(self.Jm, float))

    def C(self, q, qd):
        h = self.m2 * self.l1 * self.lc2 * np.sin(q[1])
        return h * np.array([[-qd[1], -(qd[0] + qd[1])], [qd[0], 0.0]])

    def gvec(self, q):
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.cos(q[0]) + \
            self.m2 * self.lc2 * self.g * np.cos(q[0] + q[1])
        g2 = self.m2 * self.lc2 * self.g * np.cos(q[0] + q[1])
        return np.array([g1, g2])

    def friction(self, qd):
        return np.asarray(self.r, float) ** 2 * np.asarray(self.Bm, float) * qd

    def qdd(self, q, qd, tau):
        return np.linalg.solve(self.M(q), tau - self.friction(qd) - self.C(q, qd) @ qd - self.gvec(q))

    def tau_inverse(self, q, qd, qdd):
        """Inverse dynamics: the joint torque that produces ``qdd`` (the 'model-based portion')."""
        return self.M(q) @ qdd + self.friction(qd) + self.C(q, qd) @ qd + self.gvec(q)

    # -- geometry --------------------------------------------------------------------------------
    def fk(self, q):
        """Positions of the elbow and the tip in the x-z plane."""
        e = np.array([self.l1 * np.cos(q[0]), self.l1 * np.sin(q[0])])
        return e, e + self.l2 * np.array([np.cos(q[0] + q[1]), np.sin(q[0] + q[1])])

    # -- plant interface for simulate() ------------------------------------------------------------
    def deriv(self, t, x, tau):
        q1, q2, v1, v2 = (float(v) for v in x)
        tmax = self.tau_max
        t1 = min(max(float(tau[0]), -tmax[0]), tmax[0])
        t2 = min(max(float(tau[1]), -tmax[1]), tmax[1])
        c2, s2, c1, c12 = math.cos(q2), math.sin(q2), math.cos(q1), math.cos(q1 + q2)
        a = self.m2 * self.l1 * self.lc2
        r1, r2 = self.r[0] ** 2, self.r[1] ** 2
        M11 = self.I1 + self.I2 + self.m1 * self.lc1**2 + self.m2 * (self.l1**2 + self.lc2**2) + 2 * a * c2 \
            + r1 * self.Jm[0]
        M12 = self.I2 + self.m2 * self.lc2**2 + a * c2
        M22 = self.I2 + self.m2 * self.lc2**2 + r2 * self.Jm[1]
        h = a * s2
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * c1 + self.m2 * self.lc2 * self.g * c12
        g2 = self.m2 * self.lc2 * self.g * c12
        b1 = t1 - r1 * self.Bm[0] * v1 + h * (2 * v1 * v2 + v2 * v2) - g1
        b2 = t2 - r2 * self.Bm[1] * v2 - h * v1 * v1 - g2
        det = M11 * M22 - M12 * M12
        return np.array([v1, v2, (M22 * b1 - M12 * b2) / det, (M11 * b2 - M12 * b1) / det])

    def measure(self, x):
        return x[:2].copy(), x[2:].copy()

    def applied(self, u):
        return np.clip(u, -np.asarray(self.tau_max), np.asarray(self.tau_max))


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4. Flexible joint and sampled-data analysis
# ─────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class TwoMass:
    """Motor inertia ``Jm`` and load inertia ``Jl`` joined by a spring ``k`` and damper ``c`` (flexible joint).

    ``Jm qm'' = u - Bm qm' - k (qm - ql) - c (qm' - ql')`` and
    ``Jl ql'' = k (qm - ql) + c (qm' - ql') - Bl ql' - d``.  State ``x = (qm, ql, qm', ql')``.
    """

    Jm: float
    Jl: float
    k: float
    c: float = 0.0
    Bm: float = 0.0
    Bl: float = 0.0
    disturbance: callable = None
    side: str = "motor"           # which angle the controller measures: "motor" (collocated) or "load"
    n: int = field(default=1, init=False)

    def A(self):
        Jm, Jl, k, c, Bm, Bl = self.Jm, self.Jl, self.k, self.c, self.Bm, self.Bl
        return np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                         [-k / Jm, k / Jm, -(Bm + c) / Jm, c / Jm],
                         [k / Jl, -k / Jl, c / Jl, -(Bl + c) / Jl]])

    def Bu(self):
        return np.array([0.0, 0.0, 1.0 / self.Jm, 0.0])

    def resonance(self):
        """Undamped resonance ``sqrt(k (Jm + Jl) / (Jm Jl))`` [rad/s] (the poles of the free system)."""
        return np.sqrt(self.k * (self.Jm + self.Jl) / (self.Jm * self.Jl))

    def antiresonance(self):
        """Antiresonance ``sqrt(k / Jl)`` [rad/s]: zero of the motor-angle transfer function."""
        return np.sqrt(self.k / self.Jl)

    def frf(self, w, output="motor"):
        """Frequency response ``qm(jw)/u(jw)`` or ``ql(jw)/u(jw)``."""
        s = 1j * np.asarray(w, float)
        Jm, Jl, k, c, Bm, Bl = self.Jm, self.Jl, self.k, self.c, self.Bm, self.Bl
        a = Jm * s**2 + (Bm + c) * s + k
        b = Jl * s**2 + (Bl + c) * s + k
        det = a * b - (c * s + k) ** 2
        return (b / det) if output == "motor" else ((c * s + k) / det)

    def deriv(self, t, x, u):
        d = self.disturbance(t) if self.disturbance else 0.0
        dx = self.A() @ x + self.Bu() * u
        dx[3] += -d / self.Jl
        return dx

    def measure(self, x):
        i = 0 if self.side == "motor" else 1
        return np.atleast_1d(x[i]), np.atleast_1d(x[2 + i])

    def applied(self, u):
        return u


def sampled_pd_poles(J, kp, kd, Ts, B=0.0, delay=0):
    """Poles (in the z-plane) of ``J q'' + B q' = u`` under ``u_k = -kp q_k - kd qd_k`` with a zero-order hold.

    ``delay`` extra samples of computation delay are included.  The loop is stable iff every pole has
    ``|z| < 1``; the continuous-time poles ``s`` correspond to ``z = exp(s Ts)``.
    """
    from scipy.linalg import expm

    A = np.array([[0.0, 1.0], [0.0, -B / J]])
    Bc = np.array([[0.0], [1.0 / J]])
    M = np.zeros((3, 3)); M[:2, :2] = A * Ts; M[:2, 2:] = Bc * Ts
    E = expm(M)
    Ad, Bd = E[:2, :2], E[:2, 2:]
    K = np.array([[kp, kd]])
    if delay == 0:
        return np.linalg.eigvals(Ad - Bd @ K)
    N = 2 + delay
    F = np.zeros((N, N))
    F[:2, :2] = Ad
    F[:2, 2 + delay - 1:2 + delay] = Bd             # oldest queued command drives the plant
    F[2, :2] = -K                                   # newest command enters the queue
    for i in range(1, delay):
        F[2 + i, 2 + i - 1] = 1.0
    return np.linalg.eigvals(F)


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 5. Block-diagram drawing (matplotlib) for the control-loop figures
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def bd_box(ax, x, y, text, w=1.6, h=0.8, fc="#eaf2fb", ec="#3182bd", fontsize=10):
    """Draw a labelled block centred at ``(x, y)``; returns its connection points ``dict(l, r, t, b)``."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=4)
    return dict(l=(x - w / 2, y), r=(x + w / 2, y), t=(x, y + h / 2), b=(x, y - h / 2), c=(x, y))


def bd_sum(ax, x, y, signs=("+", "-"), r=0.2, fontsize=9):
    """Draw a summing junction; ``signs`` = (sign at the left input, sign at the bottom input)."""
    from matplotlib.patches import Circle

    ax.add_patch(Circle((x, y), r, fc="white", ec="#444", lw=1.4, zorder=3))
    ax.text(x - r - 0.12, y + 0.14, signs[0], fontsize=fontsize, ha="center", va="center")
    ax.text(x + 0.14, y - r - 0.12, signs[1], fontsize=fontsize, ha="center", va="center")
    return dict(l=(x - r, y), r=(x + r, y), t=(x, y + r), b=(x, y - r), c=(x, y))


def bd_arrow(ax, pts, label=None, label_at=0, dy=0.14, fontsize=9, color="#444", head=True):
    """Draw a polyline through ``pts`` ending in an arrow head; ``label`` sits above the ``label_at``-th segment."""
    pts = [tuple(p) for p in pts]
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.4, zorder=2)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>" if head else "-", color=color, lw=1.4, shrinkA=0, shrinkB=0),
                zorder=2)
    if label:
        a, b = pts[label_at], pts[label_at + 1]
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, label, ha="center", va="bottom", fontsize=fontsize)


def bd_text(ax, x, y, text, fontsize=9, **kw):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, **kw)


def save_gif(frames, path, fps=8, max_width=480, colors=64):
    """Write RGB frames (``numpy`` arrays) to an animated GIF — only when ``RG_SAVE_ASSETS=1`` (the builder sets it).

    Returns the path if it was written, else ``None``.  Frames are shrunk to ``max_width`` pixels and quantised
    so that README previews stay small.
    """
    import os
    from pathlib import Path

    if os.environ.get("RG_SAVE_ASSETS", "0") != "1":
        return None
    from PIL import Image

    imgs = []
    for f in frames:
        im = Image.fromarray(np.asarray(f, dtype=np.uint8))
        if im.width > max_width:
            im = im.resize((max_width, int(im.height * max_width / im.width)), Image.LANCZOS)
        imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=colors))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 6. MuJoCo helpers (the import of ``mujoco`` is deferred so that the rest of the module has no dependency)
# ─────────────────────────────────────────────────────────────────────────────────────────────────
def _mj():
    import mujoco
    return mujoco


def mj_mass_matrix(model, data):
    """Dense joint-space inertia matrix ``M(q)`` of the current state (call ``mj_forward`` first)."""
    mujoco = _mj()
    M = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, M)
    except TypeError:
        mujoco.mj_fullM(model, M, data.qM)
    return M


def mj_gravity(model, data, q, n=None):
    """Gravity torque ``g(q)`` of the first ``n`` joints: the bias force with zero joint velocity."""
    mujoco = _mj()
    n = model.nv if n is None else n
    q_save, v_save = data.qpos.copy(), data.qvel.copy()
    data.qpos[:n] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    g = data.qfrc_bias[:n].copy()
    data.qpos[:], data.qvel[:] = q_save, v_save
    mujoco.mj_forward(model, data)
    return g


def mj_silence_actuators(model, idx):
    """Zero the gain and bias of actuators ``idx`` so that they exert no force; returns a function that restores them."""
    saved = (model.actuator_gainprm[idx].copy(), model.actuator_biasprm[idx].copy())
    model.actuator_gainprm[idx] = 0.0
    model.actuator_biasprm[idx] = 0.0

    def restore():
        model.actuator_gainprm[idx], model.actuator_biasprm[idx] = saved
    return restore


def mj_run(model, data, controller, ref, T, q0, n=None, ctrl_every=1, record_every=1, on_step=None,
           u_limit=None, delay=0):
    """Run an explicit joint controller on a MuJoCo arm by writing ``data.qfrc_applied``.

    Parameters
    ----------
    controller : callable ``controller(q, qd, q_ref, qd_ref, qdd_ref) -> torque`` (e.g. a :class:`PID`) of
        the first ``n`` degrees of freedom
    ref : callable ``ref(t) -> (q_ref, qd_ref, qdd_ref)`` or a constant set-point
    q0 : initial joint angles (rest); T : duration [s]
    ctrl_every : the controller is evaluated every ``ctrl_every`` physics steps (zero-order hold in between),
        i.e. the sampling period is ``ctrl_every * model.opt.timestep``
    u_limit : optional symmetric torque limit(s) applied to the torque actually written (actuator saturation)
    delay : extra delay in controller samples
    on_step : optional callback ``on_step(k, model, data)`` called after every physics step (for rendering)

    The caller is responsible for silencing the built-in actuators of the arm joints
    (:func:`mj_silence_actuators`) so that only the explicit torque acts.
    Returns a dict with ``t, q, qd, u, q_ref`` (recorded every ``record_every`` steps).
    """
    mujoco = _mj()
    n = model.nv if n is None else n
    if not callable(ref):
        const = np.asarray(ref, float)
        ref = lambda t, c=const: (c, np.zeros_like(c), np.zeros_like(c))  # noqa: E731
    mujoco.mj_resetData(model, data)
    data.qpos[:n] = q0
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    steps = int(round(T / dt))
    if hasattr(controller, "reset"):
        controller.reset()
    rec = {k: [] for k in ("t", "q", "qd", "u", "q_ref")}
    u = np.zeros(n)
    queue = [np.zeros(n)] * delay
    lim = None if u_limit is None else np.broadcast_to(np.asarray(u_limit, float), (n,))
    for k in range(steps):
        t = data.time
        if k % ctrl_every == 0:
            qr, qdr, qddr = ref(t)
            cmd = np.asarray(controller(data.qpos[:n].copy(), data.qvel[:n].copy(), qr, qdr, qddr), float)
            queue.append(cmd)
            u = queue.pop(0)
            if lim is not None:
                u = np.clip(u, -lim, lim)
        data.qfrc_applied[:n] = u
        if k % record_every == 0:
            rec["t"].append(t); rec["q"].append(data.qpos[:n].copy()); rec["qd"].append(data.qvel[:n].copy())
            rec["u"].append(u.copy()); rec["q_ref"].append(np.asarray(ref(t)[0], float).copy())
        mujoco.mj_step(model, data)
        if on_step is not None:
            on_step(k, model, data)
    data.qfrc_applied[:] = 0.0
    return {k: np.array(v) for k, v in rec.items()}

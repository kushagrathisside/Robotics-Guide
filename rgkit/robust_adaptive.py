"""Robust, adaptive and operational-space control of manipulators (RMC Chapter 4).

This module is a small, tested toolbox for the notebooks of
``RMC/04-robust-adaptive-and-operational-space-control``.  It is deliberately
self-contained: it needs only NumPy (MuJoCo is imported lazily by the helpers
that talk to the simulator).

Conventions
-----------
* Tracking error ``e = q - q_d`` (as in Slotine & Li and Spong et al.).
* The manipulator equation is ``M(q) q'' + C(q, q') q' + g(q) = tau`` with the
  Christoffel choice of ``C``, so that ``M' - 2C`` is skew-symmetric.
* The planar arms move in a **vertical** plane: ``x`` to the right, ``y`` up,
  gravity ``(0, -g)``.  Joint angles are measured counter-clockwise from the
  ``x`` axis; ``q_2`` is relative to link 1.

Contents
--------
:class:`Arm2R`
    Planar two-link arm written directly in the five *lumped parameters*
    ``theta`` of the linear-in-parameters form ``tau = Y(q, q', q'', ) theta``.
:class:`PlanarArm`
    A general planar ``n``-link arm (inertia from the centre-of-mass
    Jacobians), used for the redundant 3R arm.
Reference trajectories
    :func:`min_jerk`, :func:`sinusoids`, :func:`task_line`.
Simulation
    :func:`simulate` (RK4 plant, zero-order-hold controller with its own state).
Controllers
    :func:`ct_controller`, :func:`robust_ct_controller`, :func:`sliding_controller`,
    :func:`adaptive_controller` (Slotine-Li).
Operational space
    :func:`task_inertia`, :func:`task_dynamics`, :func:`osc_force`, :func:`jt_pd_force`,
    :func:`null_projector`, :func:`dyn_null_projector`, :func:`dyn_pinv`, :func:`damped_pinv`,
    :func:`manipulability`.
MuJoCo helpers
    :func:`mj_arm` (Panda/UR5e for torque control), :func:`mj_dynamics`, :func:`mj_add_point_mass`,
    :func:`mj_sl_terms` (Slotine-Li feed-forward by polarisation), :func:`mj_planar_xml`, :func:`mj_render`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "Arm2R", "PlanarArm", "payload_theta", "min_jerk", "sinusoids", "task_line", "Sim", "simulate",
    "ct_controller", "robust_ct_controller", "lyapunov_P", "sliding_controller", "sliding_variable",
    "adaptive_controller", "sat", "task_inertia", "task_dynamics", "osc_force", "jt_pd_force",
    "null_projector", "dyn_pinv", "dyn_null_projector", "damped_pinv", "manipulability",
    "mj_dynamics", "mj_add_point_mass", "mj_arm", "mj_panda7", "mj_planar_xml", "mj_sl_terms", "mj_render",
    "PANDA_TCP",
]

G0 = 9.81


# ───────────────────────────── planar two-link arm in lumped parameters ──────────────────────────────

def payload_theta(mp: float, l1: float, l2: float) -> np.ndarray:
    """Change of the lumped parameter vector caused by a point mass ``mp`` at the tip of a 2R arm."""
    return mp * np.array([l1 ** 2 + l2 ** 2, l2 ** 2, l1 * l2, l1, l2])


class Arm2R:
    """Planar 2R arm in a vertical plane, parametrised by five lumped parameters.

    With link masses ``m_i``, centre-of-mass distances ``lc_i`` from the joint, and centroidal
    inertias ``I_i`` (about the axis normal to the plane), the lumped parameters are::

        theta1 = m1 lc1^2 + I1 + m2 l1^2 + m2 lc2^2 + I2
        theta2 = m2 lc2^2 + I2
        theta3 = m2 l1 lc2
        theta4 = m1 lc1 + m2 l1
        theta5 = m2 lc2

    and ``M = [[th1 + 2 th3 c2, th2 + th3 c2], [th2 + th3 c2, th2]]``, ``h = -th3 s2``,
    ``C = [[h q2', h (q1' + q2')], [-h q1', 0]]``, ``g = g (th4 c1 + th5 c12, th5 c12)``.
    """

    def __init__(self, theta, l1: float = 1.0, l2: float = 0.8, g: float = G0):
        self.theta = np.asarray(theta, dtype=float).copy()
        self.l1, self.l2, self.g = float(l1), float(l2), float(g)

    # -- construction -------------------------------------------------------------------------
    @staticmethod
    def theta_from_physical(m1, m2, l1, lc1, lc2, I1, I2) -> np.ndarray:
        return np.array([m1 * lc1 ** 2 + I1 + m2 * l1 ** 2 + m2 * lc2 ** 2 + I2,
                         m2 * lc2 ** 2 + I2, m2 * l1 * lc2, m1 * lc1 + m2 * l1, m2 * lc2])

    @classmethod
    def from_physical(cls, m1, m2, l1, l2, lc1, lc2, I1, I2, g: float = G0) -> "Arm2R":
        return cls(cls.theta_from_physical(m1, m2, l1, lc1, lc2, I1, I2), l1, l2, g)

    def with_theta(self, theta) -> "Arm2R":
        return Arm2R(theta, self.l1, self.l2, self.g)

    def with_payload(self, mp: float) -> "Arm2R":
        """The same arm carrying a point mass ``mp`` at the tip."""
        return self.with_theta(self.theta + payload_theta(mp, self.l1, self.l2))

    # -- dynamics -----------------------------------------------------------------------------
    def M(self, q) -> np.ndarray:
        t1, t2, t3 = self.theta[:3]
        c2 = math.cos(q[1])
        m12 = t2 + t3 * c2
        return np.array([[t1 + 2.0 * t3 * c2, m12], [m12, t2]])

    def C(self, q, qd) -> np.ndarray:
        h = -self.theta[2] * math.sin(q[1])
        return np.array([[h * qd[1], h * (qd[0] + qd[1])], [-h * qd[0], 0.0]])

    def coriolis(self, q, qd) -> np.ndarray:
        """The velocity-product vector ``C(q, qd) qd``."""
        h = -self.theta[2] * math.sin(q[1])
        return np.array([h * qd[1] * (2.0 * qd[0] + qd[1]), -h * qd[0] ** 2])

    def gravity(self, q) -> np.ndarray:
        t4, t5 = self.theta[3:]
        c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
        return self.g * np.array([t4 * c1 + t5 * c12, t5 * c12])

    def bias(self, q, qd) -> np.ndarray:
        return self.coriolis(q, qd) + self.gravity(q)

    def tau(self, q, qd, qdd) -> np.ndarray:
        """Inverse dynamics."""
        return self.M(q) @ np.asarray(qdd) + self.bias(q, qd)

    def accel(self, q, qd, tau) -> np.ndarray:
        """Forward dynamics ``q'' = M^{-1} (tau - C qd - g)`` (closed-form 2x2 solve, for fast simulation)."""
        t1, t2, t3, t4, t5 = self.theta
        c2, s2 = math.cos(q[1]), math.sin(q[1])
        c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
        m11, m12, m22 = t1 + 2.0 * t3 * c2, t2 + t3 * c2, t2
        h = -t3 * s2
        r1 = tau[0] - h * qd[1] * (2.0 * qd[0] + qd[1]) - self.g * (t4 * c1 + t5 * c12)
        r2 = tau[1] + h * qd[0] ** 2 - self.g * t5 * c12
        det = m11 * m22 - m12 * m12
        return np.array([(m22 * r1 - m12 * r2) / det, (m11 * r2 - m12 * r1) / det])

    def regressor(self, q, qd, qdr, qddr) -> np.ndarray:
        """2x5 regressor ``Y`` with ``M(q) qddr + C(q, qd) qdr + g(q) = Y theta`` (Slotine-Li form)."""
        s2, c2 = math.sin(q[1]), math.cos(q[1])
        c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
        g = self.g
        return np.array([
            [qddr[0], qddr[1], 2.0 * c2 * qddr[0] + c2 * qddr[1] - s2 * (qd[1] * qdr[0] + (qd[0] + qd[1]) * qdr[1]),
             g * c1, g * c12],
            [0.0, qddr[0] + qddr[1], c2 * qddr[0] + s2 * qd[0] * qdr[0], 0.0, g * c12]])

    # -- kinematics ---------------------------------------------------------------------------
    def elbow(self, q) -> np.ndarray:
        return self.l1 * np.array([math.cos(q[0]), math.sin(q[0])])

    def fk(self, q) -> np.ndarray:
        s = q[0] + q[1]
        return np.array([self.l1 * math.cos(q[0]) + self.l2 * math.cos(s),
                         self.l1 * math.sin(q[0]) + self.l2 * math.sin(s)])

    def jac(self, q) -> np.ndarray:
        s = q[0] + q[1]
        l1, l2 = self.l1, self.l2
        return np.array([[-l1 * math.sin(q[0]) - l2 * math.sin(s), -l2 * math.sin(s)],
                         [l1 * math.cos(q[0]) + l2 * math.cos(s), l2 * math.cos(s)]])

    def jac_dot(self, q, qd) -> np.ndarray:
        s, sd = q[0] + q[1], qd[0] + qd[1]
        l1, l2 = self.l1, self.l2
        return np.array([[-l1 * math.cos(q[0]) * qd[0] - l2 * math.cos(s) * sd, -l2 * math.cos(s) * sd],
                         [-l1 * math.sin(q[0]) * qd[0] - l2 * math.sin(s) * sd, -l2 * math.sin(s) * sd]])

    def ik(self, x, elbow_up: bool = True):
        """Closed-form inverse kinematics of the tip position (``elbow_up``: q2 > 0)."""
        x = np.asarray(x, float)
        c2 = (x @ x - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        if abs(c2) > 1.0 + 1e-12:
            raise ValueError("target outside the workspace")
        q2 = math.acos(max(-1.0, min(1.0, c2))) * (1 if elbow_up else -1)
        q1 = math.atan2(x[1], x[0]) - math.atan2(self.l2 * math.sin(q2), self.l1 + self.l2 * math.cos(q2))
        return np.array([q1, q2])


# ─────────────────────────────── general planar n-link arm ────────────────────────────────

class PlanarArm:
    """Planar ``n``-link serial arm in a vertical plane (revolute joints, point-mass-plus-inertia links).

    ``lengths[i]`` is the length of link ``i``, ``masses[i]`` its mass, ``lc[i]`` the distance of its centre
    of mass from joint ``i`` along the link, ``inertias[i]`` its centroidal moment of inertia.
    The inertia matrix is ``M = sum_i m_i Jv_i^T Jv_i + I_i w_i w_i^T`` and the velocity-product vector is
    obtained from the centripetal accelerations of the centres of mass (Newton-Euler form).
    """

    def __init__(self, lengths, masses, lc, inertias, g: float = G0):
        self.l = np.asarray(lengths, float)
        self.m = np.asarray(masses, float)
        self.lc = np.asarray(lc, float)
        self.I = np.asarray(inertias, float)
        self.g = float(g)
        self.n = len(self.l)

    @staticmethod
    def _u(a):
        return np.array([math.cos(a), math.sin(a)])

    @staticmethod
    def _up(a):
        return np.array([-math.sin(a), math.cos(a)])

    def _phi(self, q):
        return np.cumsum(np.asarray(q, float))

    def joints(self, q) -> np.ndarray:
        """Joint positions and tip: an ``(n+1, 2)`` array."""
        phi = self._phi(q)
        pts = [np.zeros(2)]
        for i in range(self.n):
            pts.append(pts[-1] + self.l[i] * self._u(phi[i]))
        return np.array(pts)

    def fk(self, q) -> np.ndarray:
        return self.joints(q)[-1]

    def jac(self, q) -> np.ndarray:
        """2 x n Jacobian of the tip position."""
        phi = self._phi(q)
        J = np.zeros((2, self.n))
        for k in range(self.n):
            J[:, k] = sum(self.l[j] * self._up(phi[j]) for j in range(k, self.n))
        return J

    def jac_dot(self, q, qd) -> np.ndarray:
        phi = self._phi(q)
        phid = np.cumsum(np.asarray(qd, float))
        Jd = np.zeros((2, self.n))
        for k in range(self.n):
            Jd[:, k] = -sum(self.l[j] * phid[j] * self._u(phi[j]) for j in range(k, self.n))
        return Jd

    def com_jacobians(self, q):
        """List of 2 x n Jacobians of the link centres of mass."""
        phi = self._phi(q)
        out = []
        for i in range(self.n):
            J = np.zeros((2, self.n))
            for k in range(i + 1):
                J[:, k] = sum(self.l[j] * self._up(phi[j]) for j in range(k, i)) + self.lc[i] * self._up(phi[i])
            out.append(J)
        return out

    def M(self, q) -> np.ndarray:
        M = np.zeros((self.n, self.n))
        for i, J in enumerate(self.com_jacobians(q)):
            w = np.zeros(self.n)
            w[: i + 1] = 1.0
            M += self.m[i] * J.T @ J + self.I[i] * np.outer(w, w)
        return M

    def coriolis(self, q, qd) -> np.ndarray:
        """Velocity-product vector ``C(q, qd) qd`` (centripetal accelerations of the link centres of mass)."""
        phi = self._phi(q)
        phid = np.cumsum(np.asarray(qd, float))
        c = np.zeros(self.n)
        for i, J in enumerate(self.com_jacobians(q)):
            a = -sum(self.l[j] * phid[j] ** 2 * self._u(phi[j]) for j in range(i)) \
                - self.lc[i] * phid[i] ** 2 * self._u(phi[i])
            c += self.m[i] * J.T @ a
        return c

    def gravity(self, q) -> np.ndarray:
        tau = np.zeros(self.n)
        for i, J in enumerate(self.com_jacobians(q)):
            tau += self.m[i] * self.g * J[1, :]
        return tau

    def bias(self, q, qd) -> np.ndarray:
        return self.coriolis(q, qd) + self.gravity(q)

    def tau(self, q, qd, qdd) -> np.ndarray:
        return self.M(q) @ np.asarray(qdd) + self.bias(q, qd)

    def accel(self, q, qd, tau) -> np.ndarray:
        return np.linalg.solve(self.M(q), np.asarray(tau) - self.bias(q, qd))

    def C(self, q, qd, h: float = 1e-6) -> np.ndarray:
        """Christoffel matrix by central differences of ``M`` (accurate to about 1e-9)."""
        q = np.asarray(q, float)
        dM = []
        for k in range(self.n):
            e = np.zeros(self.n)
            e[k] = h
            dM.append((self.M(q + e) - self.M(q - e)) / (2 * h))        # dM[k][i, j] = dM_ij / dq_k
        C = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                C[i, j] = sum(0.5 * (dM[k][i, j] + dM[j][i, k] - dM[i][j, k]) * qd[k] for k in range(self.n))
        return C


# ─────────────────────────────────────── references ───────────────────────────────────────

def min_jerk(q0, q1, T: float, t0: float = 0.0) -> Callable:
    """Minimum-jerk (quintic) point-to-point reference ``t -> (q, q', q'')`` from ``q0`` to ``q1`` in ``T`` s."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    dq = q1 - q0

    def ref(t):
        s = min(max((t - t0) / T, 0.0), 1.0)
        p = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        dp = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
        ddp = (60 * s - 180 * s ** 2 + 120 * s ** 3) / T ** 2
        return q0 + p * dq, dp * dq, ddp * dq

    return ref


def sinusoids(center, amp, omega, phase=None) -> Callable:
    """Joint-wise sinusoidal reference ``q_i = c_i + A_i sin(w_i t + p_i)``; ``omega`` may be a vector."""
    center, amp, omega = (np.asarray(v, float) for v in (center, amp, omega))
    phase = np.zeros_like(center) if phase is None else np.asarray(phase, float)

    def ref(t):
        a = omega * t + phase
        return center + amp * np.sin(a), amp * omega * np.cos(a), -amp * omega ** 2 * np.sin(a)

    return ref


def task_line(x0, x1, T: float, t0: float = 0.0) -> Callable:
    """Straight line in task space with a minimum-jerk time law (same signature as :func:`min_jerk`)."""
    return min_jerk(x0, x1, T, t0)


# ─────────────────────────────────────── simulation ───────────────────────────────────────

@dataclass
class Sim:
    """Result of :func:`simulate`: time ``t`` (N), ``q``, ``qd``, ``tau`` (N x n) and controller state ``z``."""
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray
    z: np.ndarray | None


def simulate(accel: Callable, controller: Callable, q0, qd0, T: float, dt: float = 1e-3, z0=None,
             hold: int = 1) -> Sim:
    """Integrate ``q'' = accel(q, qd, tau)`` with RK4 while ``controller`` acts as a zero-order-hold law.

    ``controller(t, q, qd, z) -> (tau, zdot)`` is evaluated every ``hold`` integration steps (so the control
    period is ``hold * dt``); ``tau`` is held constant in between and the controller state ``z`` (for example the
    parameter estimate of an adaptive law) is advanced by explicit Euler over the control period.
    Returns the sampled signals at the integration steps.
    """
    n = len(q0)
    N = int(round(T / dt))
    q, qd = np.array(q0, float), np.array(qd0, float)
    z = None if z0 is None else np.array(z0, float)
    t_arr = np.arange(N + 1) * dt
    Q, QD, TAU = np.zeros((N + 1, n)), np.zeros((N + 1, n)), np.zeros((N + 1, n))
    Z = None if z is None else np.zeros((N + 1, len(z)))
    tau = np.zeros(n)

    def f(q_, qd_, tau_):
        return qd_, accel(q_, qd_, tau_)

    for k in range(N + 1):
        Q[k], QD[k] = q, qd
        if k % hold == 0:
            tau, zdot = controller(t_arr[k], q, qd, z)
            tau = np.asarray(tau, float)
            if z is not None and zdot is not None:
                z = z + hold * dt * np.asarray(zdot, float)
        TAU[k] = tau
        if Z is not None:
            Z[k] = z
        if k == N:
            break
        k1q, k1v = f(q, qd, tau)
        k2q, k2v = f(q + 0.5 * dt * k1q, qd + 0.5 * dt * k1v, tau)
        k3q, k3v = f(q + 0.5 * dt * k2q, qd + 0.5 * dt * k2v, tau)
        k4q, k4v = f(q + dt * k3q, qd + dt * k3v, tau)
        q = q + dt / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
        qd = qd + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return Sim(t_arr, Q, QD, TAU, Z)


# ─────────────────────────────────────── controllers ───────────────────────────────────────

def _mat(K, n):
    K = np.asarray(K, float)
    return K * np.eye(n) if K.ndim < 2 else K


def sat(x, phi=1.0):
    """Saturation function ``clip(x / phi, -1, 1)``; ``phi = 0`` gives the sign function."""
    x = np.asarray(x, float)
    return np.sign(x) if np.all(np.asarray(phi) == 0) else np.clip(x / phi, -1.0, 1.0)


def ct_controller(model, ref: Callable, Kp, Kd) -> Callable:
    """Computed torque ``tau = M^(q'' _d - Kd e' - Kp e) + C^ q' + g^`` using the *model* ``model``."""
    def ctrl(t, q, qd, z):
        qd_, dqd_, ddqd_ = ref(t)
        n = len(q)
        a = ddqd_ - _mat(Kd, n) @ (qd - dqd_) - _mat(Kp, n) @ (q - qd_)
        return model.M(q) @ a + model.bias(q, qd), None
    return ctrl


def lyapunov_P(Kp, Kd, Q=None) -> np.ndarray:
    """Solution ``P`` of ``A^T P + P A = -Q`` for ``A = [[0, I], [-Kp, -Kd]]`` (``Q = I`` by default)."""
    from scipy.linalg import solve_continuous_lyapunov
    Kp, Kd = np.atleast_2d(Kp), np.atleast_2d(Kd)
    n = Kp.shape[0]
    A = np.block([[np.zeros((n, n)), np.eye(n)], [-Kp, -Kd]])
    Q = np.eye(2 * n) if Q is None else Q
    return solve_continuous_lyapunov(A.T, -Q)


def robust_ct_controller(model, ref: Callable, Kp, Kd, rho: float, eps: float, P=None) -> Callable:
    """Computed torque with the Lyapunov-redesign term ``da = -rho w/|w|`` (smoothed inside ``|w| < eps``).

    ``w = B^T P z`` with ``z = (e, e')``.  ``eps = 0`` gives the discontinuous (ideal) law.
    """
    Kp, Kd = np.atleast_2d(Kp), np.atleast_2d(Kd)
    n = Kp.shape[0]
    P = lyapunov_P(Kp, Kd) if P is None else P

    def ctrl(t, q, qd, z):
        qd_, dqd_, ddqd_ = ref(t)
        e, ed = q - qd_, qd - dqd_
        w = P[n:, :] @ np.concatenate([e, ed])           # B^T P z
        nw = np.linalg.norm(w)
        da = np.zeros(n) if nw == 0 else (-rho * w / nw if (nw > eps or eps == 0) else -rho * w / eps)
        a = ddqd_ - Kd @ ed - Kp @ e + da
        return model.M(q) @ a + model.bias(q, qd), None
    return ctrl


def sliding_variable(q, qd, ref_vals, lam):
    """``s = e' + Lambda e`` and the reference signals ``(q'_r, q''_r)`` from reference values ``(q_d, q'_d, q''_d)``."""
    q_d, dq_d, ddq_d = ref_vals
    lam = _mat(lam, len(q))
    e, ed = q - q_d, qd - dq_d
    return ed + lam @ e, dq_d - lam @ e, ddq_d - lam @ ed


def sliding_controller(model: Arm2R, ref: Callable, lam, KD, dtheta, eta, phi=0.0) -> Callable:
    """Slotine-Li sliding controller for the 2R arm.

    ``tau = Y theta^ - KD s - K sat(s / phi)`` with the state-dependent switching gains
    ``K_i = sum_j |Y_ij| dtheta_j + eta_i`` (``dtheta`` bounds the parameter errors).  ``phi`` is the boundary-layer
    thickness (scalar or per joint); ``phi = 0`` gives the discontinuous law with ``sgn``.
    """
    dtheta = np.asarray(dtheta, float)

    def ctrl(t, q, qd, z):
        s, qdr, qddr = sliding_variable(q, qd, ref(t), lam)
        Y = model.regressor(q, qd, qdr, qddr)
        K = np.abs(Y) @ dtheta + eta
        return Y @ model.theta - _mat(KD, 2) @ s - K * sat(s, phi), None
    return ctrl


def adaptive_controller(model: Arm2R, ref: Callable, lam, KD, Gamma, sigma: float = 0.0,
                        theta_min=None, theta_max=None) -> Callable:
    """Slotine-Li adaptive controller; the controller state ``z`` is the estimate ``theta^`` (start it at ``z0``).

    ``tau = Y theta^ - KD s``, ``theta^' = -Gamma Y^T s - sigma theta^`` (``sigma > 0``: leakage), and optional
    projection onto the box ``[theta_min, theta_max]``.  ``model`` supplies the geometry (``l1``, ``l2``, ``g``).
    """
    Gamma = _mat(Gamma, 5)

    def ctrl(t, q, qd, z):
        s, qdr, qddr = sliding_variable(q, qd, ref(t), lam)
        Y = model.regressor(q, qd, qdr, qddr)
        zdot = -Gamma @ Y.T @ s - sigma * z
        if theta_min is not None:
            zdot = np.where((z <= theta_min) & (zdot < 0), 0.0, zdot)
        if theta_max is not None:
            zdot = np.where((z >= theta_max) & (zdot > 0), 0.0, zdot)
        return Y @ z - _mat(KD, 2) @ s, zdot
    return ctrl


# ───────────────────────────────── operational-space tools ─────────────────────────────────

def task_inertia(M, J) -> np.ndarray:
    """Operational-space inertia ``Lambda = (J M^{-1} J^T)^{-1}`` (``J`` of full row rank)."""
    M, J = np.asarray(M, float), np.asarray(J, float)
    return np.linalg.inv(J @ np.linalg.solve(M, J.T))


def task_dynamics(M, c, g, J, Jd_qd):
    """Khatib's task-space terms ``(Lambda, mu, p)`` with ``Lambda x'' + mu + p = F`` for ``tau = J^T F``.

    ``c`` is the velocity-product vector ``C qd``, ``g`` the gravity torque, ``Jd_qd = J' q'``.
    ``mu = Lambda (J M^{-1} c - J' q')`` and ``p = Lambda J M^{-1} g``.
    """
    M, J = np.asarray(M, float), np.asarray(J, float)
    Lam = task_inertia(M, J)
    JMi = J @ np.linalg.inv(M)
    return Lam, Lam @ (JMi @ c - Jd_qd), Lam @ (JMi @ g)


def osc_force(Lam, mu, p, xddot_cmd) -> np.ndarray:
    """Operational-space inverse dynamics ``F = Lambda x''_cmd + mu + p``."""
    return Lam @ xddot_cmd + mu + p


def jt_pd_force(e, xd, Kp, Kd) -> np.ndarray:
    """Jacobian-transpose PD force ``F = -Kp e - Kd x'`` (with ``e = x - x_d``)."""
    return -np.asarray(Kp) @ e - np.asarray(Kd) @ xd


def null_projector(J) -> np.ndarray:
    """Kinematic null-space projector ``N = I - J^+ J``."""
    J = np.asarray(J, float)
    return np.eye(J.shape[1]) - np.linalg.pinv(J) @ J


def dyn_pinv(M, J) -> np.ndarray:
    """Dynamically consistent generalised inverse ``Jbar = M^{-1} J^T Lambda`` (so that ``J Jbar = I``)."""
    M, J = np.asarray(M, float), np.asarray(J, float)
    return np.linalg.solve(M, J.T) @ task_inertia(M, J)


def dyn_null_projector(M, J) -> np.ndarray:
    """Transpose of the dynamically consistent null-space projector, ``Nbar^T = I - J^T Jbar^T``.

    Joint torques ``Nbar^T tau_0`` produce no task-space acceleration.
    """
    J = np.asarray(J, float)
    return np.eye(J.shape[1]) - J.T @ dyn_pinv(M, J).T


def damped_pinv(J, lam: float) -> np.ndarray:
    """Damped least-squares inverse ``J^T (J J^T + lam^2 I)^{-1}``."""
    J = np.asarray(J, float)
    return J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(J.shape[0]))


def manipulability(J) -> float:
    """Yoshikawa's measure ``sqrt(det(J J^T))``."""
    J = np.asarray(J, float)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


# ──────────────────────────────────── MuJoCo helpers ────────────────────────────────────

#: Tool-centre-point offset of the Franka hand along its z axis (m).
PANDA_TCP = 0.1034


def mj_dynamics(model, data, site: str | None = None, body: str | None = None, offset=(0.0, 0.0, 0.0)):
    """Dynamics and kinematics of a MuJoCo arm at the current ``qpos, qvel`` (calls ``mj_forward``).

    Returns a dict with ``M`` (dense inertia matrix, armature included), ``bias`` (``qfrc_bias`` =
    Coriolis + gravity), ``x`` (3,), ``J`` (3 x nv, translational Jacobian), ``Jd`` (3 x nv, ``dJ/dt``), ``Jd_qd`` (3,; ``J_dot q_dot``) and ``xd``
    for either a ``site`` or the point ``offset`` (in the frame of ``body``).
    """
    import mujoco
    mujoco.mj_forward(model, data)
    nv = model.nv
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(model, data, M)
    if site is not None:
        sid = model.site(site).id
        bid = int(model.site_bodyid[sid])
        point = data.site_xpos[sid].copy()
    else:
        bid = model.body(body).id
        point = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ np.asarray(offset, float)
    J, Jd = np.zeros((3, nv)), np.zeros((3, nv))
    mujoco.mj_jac(model, data, J, None, point, bid)
    mujoco.mj_jacDot(model, data, Jd, None, point, bid)
    return dict(M=M, bias=data.qfrc_bias.copy(), x=point.copy(), J=J, Jd=Jd, Jd_qd=Jd @ data.qvel, xd=J @ data.qvel)


def mj_add_point_mass(model, body: str, mass: float, point_local) -> None:
    """Add a point mass to a body of a compiled model (in place): updates mass, centre of mass and inertia.

    ``point_local`` is the position of the added mass in the body's own frame.  The combined inertia tensor is
    re-diagonalised so that ``body_mass``, ``body_ipos``, ``body_inertia`` and ``body_iquat`` stay consistent.
    """
    import mujoco
    b = model.body(body).id
    m0, c0 = model.body_mass[b], model.body_ipos[b].copy()
    R0 = np.zeros(9)
    mujoco.mju_quat2Mat(R0, model.body_iquat[b])
    R0 = R0.reshape(3, 3)
    I0 = R0 @ np.diag(model.body_inertia[b]) @ R0.T                 # inertia about the old CoM, body axes
    p = np.asarray(point_local, float)
    m1 = m0 + mass
    c1 = (m0 * c0 + mass * p) / m1

    def shift(m, c_from, c_to):                                     # parallel-axis term for a point/body
        d = c_from - c_to
        return m * (d @ d * np.eye(3) - np.outer(d, d))

    I1 = I0 + shift(m0, c0, c1) + shift(mass, p, c1)
    w, V = np.linalg.eigh(I1)
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, V.reshape(-1))
    model.body_mass[b], model.body_ipos[b] = m1, c1
    model.body_inertia[b], model.body_iquat[b] = w, quat
    # propagate to subtree quantities used by MuJoCo (total mass of the ancestors)
    mujoco.mj_setConst(model, mujoco.MjData(model))


def mj_planar_xml(lengths, masses, lc, inertias, g: float = G0, timestep: float = 1e-3, scene: str = "",
                  floor_z: float | None = None) -> str:
    """MJCF string of a planar ``n``-link arm in the vertical ``x-y`` plane with explicit inertial parameters.

    Hinge axes are ``+z``, link ``i`` extends along its own ``x`` axis, the tool site is called ``tip`` and the
    camera ``front`` looks at the plane along ``-z`` (``x`` to the right, ``y`` up).  Joint damping, armature and
    friction are zero, so the dynamics are those of :class:`PlanarArm` exactly.  ``scene`` is extra MJCF placed
    directly under ``<mujoco>`` (for example ``rgkit.mujoco_utils.scene_assets()``).
    """
    n = len(lengths)
    cols = ["0.84 0.35 0.15 1", "0.12 0.47 0.71 1", "0.17 0.63 0.17 1", "0.58 0.40 0.74 1"]
    body = ""
    for i in reversed(range(n)):
        pos = "0 0 0" if i == 0 else f"{lengths[i - 1]} 0 0"
        inner = body
        site = f'<site name="tip" pos="{lengths[i]} 0 0" size="0.025" rgba="1 0.5 0 1"/>' if i == n - 1 else ""
        body = (f'<body name="link{i + 1}" pos="{pos}">\n'
                f'  <joint name="q{i + 1}" type="hinge" axis="0 0 1"/>\n'
                f'  <inertial pos="{lc[i]} 0 0" mass="{masses[i]}" diaginertia="{inertias[i] / 2} {inertias[i] / 2} {inertias[i]}"/>\n'
                f'  <geom type="capsule" fromto="0 0 0 {lengths[i]} 0 0" size="0.03" rgba="{cols[i % 4]}" '
                f'contype="0" conaffinity="0"/>\n  {site}\n{inner}</body>\n')
    reach = float(sum(lengths))
    floor = "" if floor_z is None else f'<geom type="plane" pos="0 0 {floor_z}" size="3 3 0.1" rgba="0.9 0.9 0.92 1"/>'
    return (f'<mujoco model="planar{n}r">\n  <option gravity="0 {-g} 0" timestep="{timestep}" integrator="RK4"/>\n'
            f'  {scene}\n  <worldbody>\n    <light pos="0 0 3" dir="0 0 -1"/>\n'
            f'    <camera name="front" pos="0 {0.0} {2.2 * reach + 1.0}" xyaxes="1 0 0 0 1 0" fovy="50"/>\n'
            f'    {floor}\n{body}  </worldbody>\n</mujoco>')


def mj_arm(name: str = "panda", zero_damping: bool = True, tcp: float | None = None):
    """A Menagerie arm prepared for torque control: ``"panda"`` (7 joints, fingers removed) or ``"ur5e"``.

    Actuators, tendons, equality constraints and keyframes are removed, so joints are driven through
    ``data.qfrc_applied``.  Joint damping is set to zero by default so that the rigid-body model
    ``M q'' + c + g = tau`` is exact (armature, i.e. rotor inertia, is kept).  A site ``tcp`` is added: on the
    Panda hand ``tcp`` metres along its z axis (default :data:`PANDA_TCP`), on the UR5e at its flange
    (``attachment_site``) shifted by ``tcp`` if given.  Returns ``(model, data)`` at the manufacturer's home posture.
    """
    import mujoco
    from . import robots
    base = {"panda": "panda.xml", "ur5e": "ur5e.xml"}[name]
    spec = mujoco.MjSpec.from_file(str(robots.fetch(name) / base))
    for k in list(spec.keys):
        spec.delete(k)
    for e in list(spec.equalities):
        spec.delete(e)
    for a in list(spec.actuators):
        spec.delete(a)
    for t in list(spec.tendons):
        spec.delete(t)
    if name == "panda":
        for b in ("left_finger", "right_finger"):
            spec.delete(spec.body(b))
        spec.body("hand").add_site(name="tcp", pos=[0, 0, PANDA_TCP if tcp is None else tcp], size=[0.01] * 3)
        home = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
    else:
        spec.body("wrist_3_link").add_site(name="tcp", pos=[0, 0.1 if tcp is None else tcp, 0], size=[0.01] * 3)
        home = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
    model = spec.compile()
    if zero_damping:
        model.dof_damping[:] = 0.0
    model.opt.timestep = 1e-3
    data = mujoco.MjData(model)
    data.qpos[:] = home
    mujoco.mj_forward(model, data)
    return model, data


def mj_panda7(zero_damping: bool = True, tcp: float = PANDA_TCP):
    """The Franka Panda reduced to its 7 arm joints; shorthand for ``mj_arm("panda", ...)``."""
    return mj_arm("panda", zero_damping, tcp)


def _mj_bias_at(model, data, v):
    import mujoco
    saved = data.qvel.copy()
    data.qvel[:] = v
    mujoco.mj_comVel(model, data)                       # cvel and cdof_dot depend on qvel
    out = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, out)
    data.qvel[:] = saved
    mujoco.mj_comVel(model, data)
    return out


def mj_sl_terms(model, data, qdr, qddr, M=None) -> np.ndarray:
    """``M(q) qddr + C(q, qd) qdr + g(q)`` of a MuJoCo model at the current state (the Slotine-Li feed-forward).

    MuJoCo exposes only the velocity-product vector ``C(q, v) v``.  Because the Christoffel form ``C(q, v) w`` is a
    symmetric bilinear form in ``(v, w)``, polarisation gives
    ``C(q, qd) qdr = [c(qd + qdr) - c(qd - qdr)] / 4`` with ``c(u) = C(q, u) u`` obtained from ``mj_rne`` at velocity ``u``.
    Call ``mj_forward`` first (kinematics and ``qM`` must be current).
    """
    import mujoco
    qdr, qddr = np.asarray(qdr, float), np.asarray(qddr, float)
    if M is None:
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, M)
    g = _mj_bias_at(model, data, np.zeros(model.nv))
    v = data.qvel.copy()
    cq = 0.25 * ((_mj_bias_at(model, data, v + qdr) - g) - (_mj_bias_at(model, data, v - qdr) - g))
    return M @ qddr + cq + g


def mj_render(model, data, camera=None, markers=(), width=480, height=360, lines=()):
    """Render ``data`` offscreen with optional decorations.

    ``markers`` is a sequence of ``(position, radius, rgba)`` spheres, ``lines`` a sequence of
    ``(points (N, 3), width, rgba)`` polylines drawn as thin capsules.  ``camera`` is a dict as in
    :func:`rgkit.mujoco_utils.render` or a camera name.  Renderers are cached per model and size.
    """
    import mujoco
    cache = mj_render.__dict__.setdefault("_cache", {})
    key = (id(model), width, height)
    if key not in cache:
        cache[key] = mujoco.Renderer(model, height=height, width=width)
    r = cache[key]
    if isinstance(camera, dict):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        for k, v in camera.items():
            if k == "lookat":
                cam.lookat[:] = v
            else:
                setattr(cam, k, v)
        camera = cam
    r.update_scene(data, camera=-1 if camera is None else camera)
    sc = r.scene
    for pos, rad, rgba in markers:
        if sc.ngeom >= sc.maxgeom:
            break
        mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, 0, 0]),
                            np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        sc.ngeom += 1
    for pts, w, rgba in lines:
        pts = np.asarray(pts, float)
        for a, b in zip(pts[:-1], pts[1:]):
            if sc.ngeom >= sc.maxgeom:
                break
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                np.zeros(9), np.asarray(rgba, np.float32))
            mujoco.mjv_connector(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, w, a, b)
            sc.ngeom += 1
    return r.render().copy()

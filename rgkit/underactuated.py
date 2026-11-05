"""Underactuated mechanical systems: pendulum, cart-pole and acrobot.

Everything in this module follows the *manipulator equation*

.. math::  M(q)\\,\\ddot q + h(q,\\dot q) = B\\,u, \\qquad h = C(q,\\dot q)\\dot q - \\tau_g(q),

where :math:`h` is the "bias" vector that MuJoCo calls ``qfrc_bias`` (Coriolis and centrifugal terms
plus gravity).  A system is *underactuated* when ``B`` has fewer independent columns than ``q`` has
entries.

Conventions (identical to the MuJoCo models built in the notebooks)

* **Pendulum**: angle ``theta`` measured from the *downward* vertical; ``theta = pi`` is upright.
  State ``[theta, thetadot]``.
* **Cart-pole**: ``q = [x, theta]``; the pole hangs down at ``theta = 0`` and is upright at ``theta = pi``
  (the pole is a point mass ``mp`` on a massless rod of length ``l``).  State ``[x, theta, xdot, thetadot]``.
* **Acrobot**: ``q = [theta1, theta2]``; ``theta1`` is measured from the downward vertical, ``theta2`` is
  the elbow angle relative to link 1 (0 = straight).  Upright is ``[pi, 0]``.  State
  ``[theta1, theta2, theta1dot, theta2dot]``; the torque acts at the elbow only.

All dynamics functions broadcast: a state of shape ``(n,)`` is one state, a state of shape ``(n, N)``
is a batch of ``N`` states (used for region-of-attraction experiments).

The module contains the models, their linearisations, LQR, partial feedback linearisation
(collocated and non-collocated), the swing-up controllers used in the chapter, an RK4 simulator and
the sympy derivations that the tests use as an independent check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are

__all__ = [
    "wrap_pi", "rk4_step", "simulate", "linearize", "controllability_matrix", "lqr", "hybrid_controller",
    "lqr_basin_fraction",
    "PendulumParams", "pendulum_deriv", "pendulum_energy", "pendulum_energy_shaping", "pendulum_linearized",
    "CartPoleParams", "cartpole_M", "cartpole_h", "cartpole_deriv", "cartpole_energy", "cartpole_pole_energy",
    "cartpole_linearized", "cartpole_swingup",
    "AcrobotParams", "acrobot_M", "acrobot_h", "acrobot_deriv", "acrobot_energy", "acrobot_linearized",
    "acrobot_swingup", "acrobot_position",
    "pfl_collocated", "pfl_noncollocated",
    "pendulum_symbolic", "cartpole_symbolic", "acrobot_symbolic", "lagrange_to_manipulator",
]


# --------------------------------------------------------------------------------------------------
# generic tools
# --------------------------------------------------------------------------------------------------
def wrap_pi(a):
    """Wrap an angle (or array of angles) to ``[-pi, pi)``."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def rk4_step(deriv, x, u, dt):
    """One classical Runge-Kutta step of ``xdot = deriv(x, u)`` with the input held constant."""
    k1 = deriv(x, u)
    k2 = deriv(x + 0.5 * dt * k1, u)
    k3 = deriv(x + 0.5 * dt * k2, u)
    k4 = deriv(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(deriv, controller, x0, dt, T, substeps=1):
    """Simulate ``xdot = deriv(x, u)`` under ``u = controller(t, x)`` (zero-order hold, RK4).

    Parameters
    ----------
    deriv : callable ``(x, u) -> xdot``.
    controller : callable ``(t, x) -> u`` evaluated once per control period ``dt``; may return a scalar,
        or (for a batch ``x`` of shape ``(n, N)``) an array of shape ``(N,)``.
    x0 : initial state, shape ``(n,)`` or ``(n, N)``.
    dt, T : control period and duration.
    substeps : RK4 sub-steps per control period (the input is held).

    Returns ``t`` of shape ``(K+1,)``, ``X`` of shape ``(K+1, *x0.shape)`` and ``U`` of shape ``(K, ...)``.
    """
    x = np.array(x0, dtype=float)
    K = int(round(T / dt))
    t = np.arange(K + 1) * dt
    X = np.empty((K + 1,) + x.shape)
    U = []
    X[0] = x
    for k in range(K):
        u = controller(t[k], x)
        U.append(np.asarray(u, dtype=float))
        for _ in range(substeps):
            x = rk4_step(deriv, x, u, dt / substeps)
        X[k + 1] = x
    return t, X, np.array(U)


def linearize(deriv, x0, u0=0.0, eps=1e-6):
    """Central-difference Jacobians ``A = df/dx`` and ``B = df/du`` of ``xdot = deriv(x, u)`` at ``(x0, u0)``."""
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    A = np.empty((n, n))
    for i in range(n):
        d = np.zeros(n)
        d[i] = eps
        A[:, i] = (deriv(x0 + d, u0) - deriv(x0 - d, u0)) / (2 * eps)
    B = ((deriv(x0, u0 + eps) - deriv(x0, u0 - eps)) / (2 * eps)).reshape(n, 1)
    return A, B


def controllability_matrix(A, B):
    """Kalman controllability matrix ``[B, AB, ..., A^(n-1) B]``."""
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    n = A.shape[0]
    cols, M = [], B
    for _ in range(n):
        cols.append(M)
        M = A @ M
    return np.hstack(cols)


def lqr(A, B, Q, R):
    """Continuous-time LQR: returns the gain ``K`` (``u = -K x``) and the Riccati solution ``P``."""
    A, B, Q = np.atleast_2d(A), np.atleast_2d(B), np.atleast_2d(Q)
    R = np.atleast_2d(R)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P


def hybrid_controller(swingup, K, P, x_eq, level, u_max=np.inf, angle_idx=()):
    """Switch between a swing-up law and a saturated LQR balancer.

    The LQR law ``u = -K (x - x_eq)`` is used while the quadratic cost-to-go ``e^T P e`` of the wrapped
    error ``e`` is below ``level``; otherwise ``swingup(x)`` is used.  ``angle_idx`` lists the entries of the
    state that are angles (wrapped to ``[-pi, pi)`` before the test).
    """
    x_eq = np.asarray(x_eq, dtype=float)

    def ctrl(t, x):
        e = np.array(x, dtype=float) - x_eq
        for i in angle_idx:
            e[i] = wrap_pi(e[i])
        if e @ P @ e < level:
            return float(np.clip(-(K @ e).item(), -u_max, u_max))
        return swingup(x)

    return ctrl


def lqr_basin_fraction(deriv, K, P, x_eq, u_max, levels, n=300, T=8.0, dt=0.005, angle_idx=(), tol=1e-2, seed=1):
    """Monte-Carlo estimate of the fraction of initial states with ``e^T P e = level`` that the saturated LQR
    law brings to ``x_eq`` (within ``tol``) in ``T`` seconds.  Initial errors are uniform in direction.

    This is a *sampling* experiment, not a proof: it can show that a level set is **not** inside the region
    of attraction, never that it is.
    """
    rng = np.random.default_rng(seed)
    x_eq = np.asarray(x_eq, dtype=float)
    L = np.linalg.cholesky(np.linalg.inv(P))
    frac = []
    for level in levels:
        z = rng.normal(size=(P.shape[0], n))
        z *= np.sqrt(level) / np.linalg.norm(z, axis=0)
        x0 = x_eq[:, None] + L @ z

        def ctrl(t, x):
            e = x - x_eq[:, None]
            for i in angle_idx:
                e[i] = wrap_pi(e[i])
            return np.clip(-(K @ e).ravel(), -u_max, u_max)

        _, X, _ = simulate(deriv, ctrl, x0, dt, T)
        ef = X[-1] - x_eq[:, None]
        for i in angle_idx:
            ef[i] = wrap_pi(ef[i])
        frac.append(float(np.mean(np.linalg.norm(ef, axis=0) < tol)))
    return np.array(frac)


def _solve2(a, b, c, d, r1, r2):
    """Solve ``[[a, b], [c, d]] [y1, y2] = [r1, r2]`` for broadcast entries (Cramer's rule)."""
    det = a * d - b * c
    return (d * r1 - b * r2) / det, (a * r2 - c * r1) / det


# --------------------------------------------------------------------------------------------------
# pendulum
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PendulumParams:
    """Point mass ``m`` on a massless rod of length ``l`` with viscous joint friction ``b`` (N m s/rad)."""
    m: float = 1.0
    l: float = 0.5
    b: float = 0.0
    g: float = 9.81


def pendulum_deriv(x, u, p: PendulumParams = PendulumParams()):
    """``m l^2 thetaddot + b thetadot + m g l sin(theta) = u``; state ``[theta, thetadot]``."""
    x = np.asarray(x, dtype=float)
    th, om = x[0], x[1]
    acc = (u - p.b * om - p.m * p.g * p.l * np.sin(th)) / (p.m * p.l ** 2)
    return np.stack([om, acc])


def pendulum_energy(x, p: PendulumParams = PendulumParams()):
    """Mechanical energy ``E = (1/2) m l^2 thetadot^2 - m g l cos(theta)`` (zero at the horizontal level)."""
    x = np.asarray(x, dtype=float)
    return 0.5 * p.m * p.l ** 2 * x[1] ** 2 - p.m * p.g * p.l * np.cos(x[0])


def pendulum_energy_shaping(x, p: PendulumParams = PendulumParams(), k=1.0, u_max=np.inf):
    """Energy-shaping swing-up law ``u = -k thetadot (E - E_d)`` with ``E_d = m g l``, saturated at ``u_max``.

    Along trajectories of the frictionless pendulum :math:`\\dot E = \\dot\\theta\\,u`, so
    :math:`\\dot E = -k\\dot\\theta^2 (E - E_d)`.
    """
    x = np.asarray(x, dtype=float)
    Ed = p.m * p.g * p.l
    return np.clip(-k * x[1] * (pendulum_energy(x, p) - Ed), -u_max, u_max)


def pendulum_linearized(p: PendulumParams = PendulumParams()):
    """``(A, B)`` of the pendulum linearised about the upright equilibrium, in ``phi = theta - pi``."""
    I = p.m * p.l ** 2
    return np.array([[0.0, 1.0], [p.m * p.g * p.l / I, -p.b / I]]), np.array([[0.0], [1.0 / I]])


# --------------------------------------------------------------------------------------------------
# cart-pole
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CartPoleParams:
    """Cart of mass ``mc`` (force ``u`` along the rail) carrying a point mass ``mp`` on a massless rod ``l``."""
    mc: float = 1.0
    mp: float = 0.3
    l: float = 0.5
    g: float = 9.81


def cartpole_M(q, p: CartPoleParams = CartPoleParams()):
    """Mass matrix; ``q = [x, theta]`` (batch ``(2, N)`` allowed). Returns an array ``(2, 2, ...)``."""
    th = np.asarray(q)[1]
    off = p.mp * p.l * np.cos(th)
    return np.array([[(p.mc + p.mp) + 0 * th, off], [off, p.mp * p.l ** 2 + 0 * th]])


def cartpole_h(q, qd, p: CartPoleParams = CartPoleParams()):
    """Bias vector ``h = C qdot - tau_g`` (MuJoCo's ``qfrc_bias``); shape ``(2, ...)``."""
    th, thd = np.asarray(q)[1], np.asarray(qd)[1]
    return np.array([-p.mp * p.l * thd ** 2 * np.sin(th) + 0 * th, p.mp * p.g * p.l * np.sin(th)])


def cartpole_deriv(s, u, p: CartPoleParams = CartPoleParams()):
    """State ``[x, theta, xdot, thetadot]``; ``u`` is the horizontal force on the cart (closed-form accelerations)."""
    s = np.asarray(s, dtype=float)
    th, xd, thd = s[1], s[2], s[3]
    sn, c = np.sin(th), np.cos(th)
    den = p.mc + p.mp * sn ** 2
    xdd = (u + p.mp * sn * (p.l * thd ** 2 + p.g * c)) / den
    thdd = (-c * u - p.mp * p.l * thd ** 2 * c * sn - (p.mc + p.mp) * p.g * sn) / (p.l * den)
    return np.stack([xd, thd, xdd, thdd])


def cartpole_pole_energy(s, p: CartPoleParams = CartPoleParams()):
    """Pole energy ``E = (1/2) mp l^2 thetadot^2 - mp g l cos(theta)``; the target is ``E_d = mp g l``."""
    s = np.asarray(s, dtype=float)
    return 0.5 * p.mp * p.l ** 2 * s[3] ** 2 - p.mp * p.g * p.l * np.cos(s[1])


def cartpole_energy(s, p: CartPoleParams = CartPoleParams()):
    """Total mechanical energy ``T + U`` of the cart-pole (``U = -mp g l cos(theta)``)."""
    s = np.asarray(s, dtype=float)
    th, xd, thd = s[1], s[2], s[3]
    T = 0.5 * (p.mc + p.mp) * xd ** 2 + p.mp * p.l * xd * thd * np.cos(th) + 0.5 * p.mp * p.l ** 2 * thd ** 2
    return T - p.mp * p.g * p.l * np.cos(th)


def cartpole_linearized(p: CartPoleParams = CartPoleParams()):
    """``(A, B)`` about the upright equilibrium, in ``[x, phi, xdot, phidot]`` with ``phi = theta - pi``."""
    g, l, mc, mp = p.g, p.l, p.mc, p.mp
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, mp * g / mc, 0, 0], [0, (mc + mp) * g / (mc * l), 0, 0]], float)
    B = np.array([[0], [0], [1 / mc], [1 / (mc * l)]], float)
    return A, B


def cartpole_swingup(s, p: CartPoleParams = CartPoleParams(), k_E=3.0, k_p=1.0, k_d=1.0, u_max=np.inf):
    """Swing-up by energy shaping through collocated PFL on the cart.

    The desired cart acceleration is ``v = k_E thetadot cos(theta) (E - E_d) - k_p x - k_d xdot``; the force that
    produces it exactly is returned (saturated at ``u_max``).
    """
    s = np.asarray(s, dtype=float)
    x, th, xd, thd = s
    Ed = p.mp * p.g * p.l
    v = k_E * thd * np.cos(th) * (cartpole_pole_energy(s, p) - Ed) - k_p * x - k_d * xd
    M, h = cartpole_M(s[:2], p), cartpole_h(s[:2], s[2:], p)
    return float(np.clip(pfl_collocated(M, h, v, act=0), -u_max, u_max))


# --------------------------------------------------------------------------------------------------
# acrobot
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AcrobotParams:
    """Two uniform links: masses ``m1, m2``; lengths ``l1, l2``; COM distances ``lc1, lc2`` from the joint;
    moments of inertia ``I1, I2`` *about the centres of mass*."""
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    lc1: float = 0.5
    lc2: float = 0.5
    I1: float = 1.0 / 12.0
    I2: float = 1.0 / 12.0
    g: float = 9.81


def acrobot_M(q, p: AcrobotParams = AcrobotParams()):
    """Mass matrix of the acrobot; ``q = [theta1, theta2]``."""
    c2 = np.cos(np.asarray(q)[1])
    a1 = p.I1 + p.m1 * p.lc1 ** 2 + p.m2 * p.l1 ** 2
    a2 = p.I2 + p.m2 * p.lc2 ** 2
    a3 = p.m2 * p.l1 * p.lc2
    m12 = a2 + a3 * c2
    return np.array([[a1 + a2 + 2 * a3 * c2, m12], [m12, a2 + 0 * c2]])


def acrobot_h(q, qd, p: AcrobotParams = AcrobotParams()):
    """Bias ``h = C qdot - tau_g`` of the acrobot."""
    q, qd = np.asarray(q), np.asarray(qd)
    s1, s2, s12 = np.sin(q[0]), np.sin(q[1]), np.sin(q[0] + q[1])
    a3 = p.m2 * p.l1 * p.lc2
    h1 = -a3 * s2 * (2 * qd[0] * qd[1] + qd[1] ** 2) + (p.m1 * p.lc1 + p.m2 * p.l1) * p.g * s1 + p.m2 * p.lc2 * p.g * s12
    h2 = a3 * s2 * qd[0] ** 2 + p.m2 * p.lc2 * p.g * s12
    return np.array([h1, h2])


def acrobot_deriv(s, u, p: AcrobotParams = AcrobotParams()):
    """State ``[theta1, theta2, theta1dot, theta2dot]``; ``u`` is the elbow torque."""
    s = np.asarray(s, dtype=float)
    q, qd = s[:2], s[2:]
    M, h = acrobot_M(q, p), acrobot_h(q, qd, p)
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    qdd1, qdd2 = _solve2(a, b, c, d, -h[0], u - h[1])
    return np.stack([qd[0], qd[1], qdd1, qdd2])


def acrobot_position(q, p: AcrobotParams = AcrobotParams()):
    """Elbow and tip positions ``(x, z)`` (z up) for ``q = [theta1, theta2]``."""
    q = np.asarray(q, dtype=float)
    elbow = np.array([p.l1 * np.sin(q[0]), -p.l1 * np.cos(q[0])])
    tip = elbow + np.array([p.l2 * np.sin(q[0] + q[1]), -p.l2 * np.cos(q[0] + q[1])])
    return elbow, tip


def acrobot_energy(s, p: AcrobotParams = AcrobotParams()):
    """Total mechanical energy; the potential is ``-g (m1 lc1 cos q1 + m2 (l1 cos q1 + lc2 cos(q1+q2)))``."""
    s = np.asarray(s, dtype=float)
    q, qd = s[:2], s[2:]
    M = acrobot_M(q, p)
    T = 0.5 * (M[0, 0] * qd[0] ** 2 + 2 * M[0, 1] * qd[0] * qd[1] + M[1, 1] * qd[1] ** 2)
    U = -p.g * (p.m1 * p.lc1 * np.cos(q[0]) + p.m2 * (p.l1 * np.cos(q[0]) + p.lc2 * np.cos(q[0] + q[1])))
    return T + U


def acrobot_linearized(p: AcrobotParams = AcrobotParams()):
    """``(A, B)`` about the upright equilibrium ``[pi, 0, 0, 0]`` (finite-difference Jacobians of the exact model)."""
    return linearize(lambda x, u: acrobot_deriv(x, u, p), np.array([np.pi, 0, 0, 0]), 0.0)


def acrobot_swingup(s, p: AcrobotParams = AcrobotParams(), alpha=0.9, kp=30.0, kd=8.0, u_max=np.inf):
    """Spong-style swing-up: collocated PFL tracking ``q2_d = (2 alpha / pi) arctan(q1dot)``.

    The elbow is steered to an angle whose sign follows the velocity of link 1, which pumps energy into
    the first link; ``alpha < pi/2`` bounds the elbow excursion.
    """
    s = np.asarray(s, dtype=float)
    q, qd = s[:2], s[2:]
    q2d = 2.0 * alpha / np.pi * np.arctan(qd[0])
    v = kp * (q2d - q[1]) - kd * qd[1]
    u = pfl_collocated(acrobot_M(q, p), acrobot_h(q, qd, p), v, act=1)
    return float(np.clip(u, -u_max, u_max))


# --------------------------------------------------------------------------------------------------
# partial feedback linearisation for two degrees of freedom with one actuator
# --------------------------------------------------------------------------------------------------
def pfl_collocated(M, h, v, act):
    """Collocated PFL: the input that makes the *actuated* acceleration equal ``v``.

    For ``M qdd + h = e_act u``, with ``a = act`` the actuated and ``p = 1 - act`` the passive index:
    ``u = (M_aa - M_ap M_pp^-1 M_pa) v + h_a - M_ap M_pp^-1 h_p``  gives  ``qdd_a = v``.
    """
    a, pas = act, 1 - act
    Mpp_inv = 1.0 / M[pas, pas]
    return (M[a, a] - M[a, pas] * Mpp_inv * M[pas, a]) * v + h[a] - M[a, pas] * Mpp_inv * h[pas]


def pfl_noncollocated(M, h, v, act):
    """Non-collocated PFL: the input that makes the *passive* acceleration equal ``v`` (needs ``M_pa != 0``).

    Row ``p`` gives ``qdd_a = -M_pa^-1 (M_pp v + h_p)``; row ``a`` then gives the required ``u``.
    """
    a, pas = act, 1 - act
    qdd_a = -(M[pas, pas] * v + h[pas]) / M[pas, a]
    return M[a, pas] * v + M[a, a] * qdd_a + h[a]


# --------------------------------------------------------------------------------------------------
# symbolic derivations (sympy) -- used by the notebooks and as an independent check in the tests
# --------------------------------------------------------------------------------------------------
def lagrange_to_manipulator(T, U, q, qd):
    """From kinetic and potential energy to ``M``, ``C`` (Christoffel), ``tau_g`` and ``h = C qd - tau_g``.

    Uses ``M = Hessian_qd(T)``, ``h = (dM/dt) qd - dT/dq + dU/dq`` (Euler-Lagrange) and the Christoffel
    symbols for ``C`` so that ``dM/dt - 2C`` is skew-symmetric.
    """
    import sympy as sp

    n = len(q)
    q, qd = sp.Matrix(q), sp.Matrix(qd)
    M = sp.hessian(T, tuple(qd)).applyfunc(sp.simplify)
    dTdq = sp.Matrix([sp.diff(T, qi) for qi in q])
    dUdq = sp.Matrix([sp.diff(U, qi) for qi in q])
    Mdot_qd = sp.zeros(n, 1)
    for k in range(n):
        Mdot_qd += sp.diff(M, q[k]) * qd[k] * qd
    h = (Mdot_qd - dTdq + dUdq).applyfunc(sp.simplify)
    C = sp.zeros(n, n)
    for k in range(n):
        for j in range(n):
            C[k, j] = sum(sp.Rational(1, 2) * (sp.diff(M[k, j], q[i]) + sp.diff(M[k, i], q[j]) - sp.diff(M[i, j], q[k])) * qd[i]
                          for i in range(n))
    C = C.applyfunc(sp.simplify)
    tau_g = (-dUdq).applyfunc(sp.simplify)
    return dict(q=q, qd=qd, M=M, C=C, tau_g=tau_g, h=h, T=T, U=U)


def pendulum_symbolic():
    """Euler-Lagrange derivation of the pendulum (point mass; ``theta = 0`` hanging down)."""
    import sympy as sp

    m, l, g = sp.symbols("m l g", positive=True)
    th, thd = sp.symbols("theta thetad")
    T = sp.Rational(1, 2) * m * l ** 2 * thd ** 2
    U = -m * g * l * sp.cos(th)
    out = lagrange_to_manipulator(T, U, [th], [thd])
    out["params"] = (m, l, g)
    return out


def cartpole_symbolic():
    """Euler-Lagrange derivation of the cart-pole from the Cartesian position of the pole mass."""
    import sympy as sp

    mc, mp, l, g = sp.symbols("m_c m_p l g", positive=True)
    x, th, xd, thd = sp.symbols("x theta xd thetad")
    px, pz = x + l * sp.sin(th), -l * sp.cos(th)
    pxd = xd + l * sp.cos(th) * thd
    pzd = l * sp.sin(th) * thd
    T = sp.Rational(1, 2) * mc * xd ** 2 + sp.Rational(1, 2) * mp * (pxd ** 2 + pzd ** 2)
    U = mp * g * pz
    out = lagrange_to_manipulator(sp.expand(T), U, [x, th], [xd, thd])
    out["params"] = (mc, mp, l, g)
    return out


def acrobot_symbolic():
    """Euler-Lagrange derivation of the acrobot (links with COM offsets ``lc`` and inertias ``I`` about the COM)."""
    import sympy as sp

    m1, m2, l1, lc1, lc2, I1, I2, g = sp.symbols("m_1 m_2 l_1 l_c1 l_c2 I_1 I_2 g", positive=True)
    q1, q2, d1, d2 = sp.symbols("q1 q2 qd1 qd2")
    c1x, c1z = lc1 * sp.sin(q1), -lc1 * sp.cos(q1)
    c2x, c2z = l1 * sp.sin(q1) + lc2 * sp.sin(q1 + q2), -l1 * sp.cos(q1) - lc2 * sp.cos(q1 + q2)
    q, qd = [q1, q2], [d1, d2]

    def dot(expr):
        return sum(sp.diff(expr, qi) * qdi for qi, qdi in zip(q, qd))

    T = (sp.Rational(1, 2) * m1 * (dot(c1x) ** 2 + dot(c1z) ** 2) + sp.Rational(1, 2) * I1 * d1 ** 2
         + sp.Rational(1, 2) * m2 * (dot(c2x) ** 2 + dot(c2z) ** 2) + sp.Rational(1, 2) * I2 * (d1 + d2) ** 2)
    U = m1 * g * c1z + m2 * g * c2z
    out = lagrange_to_manipulator(sp.expand(T), U, q, qd)
    out["params"] = (m1, m2, l1, lc1, lc2, I1, I2, g)
    return out

"""Digital implementation of joint control: sampling, delay, cascades and real-time scheduling.

Back-end of notebook 8.3 (``RMC/08-sensing-and-implementation``). Pure NumPy/SciPy.

* :func:`alias_frequency` — where a sinusoid appears after sampling.
* :func:`loop_margins`, :func:`delay_margin` — gain-crossover frequency, phase margin and the largest pure
  delay that a loop can tolerate, for any loop transfer function ``L(jw)``.
* :func:`pd_joint_L` — the loop transfer function of a PD-controlled joint ``J s^2 + b s``.
* :func:`simulate_delayed_loop` — exact simulation of a continuous plant under a sampled-data state-feedback law
  whose command is applied after a (random) delay.
* :func:`motor_cascade_gains`, :func:`simulate_cascade` — current / velocity / position loops of a DC-motor drive.
* :func:`liu_layland_bound`, :func:`response_times` — fixed-priority real-time schedulability analysis.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = ["alias_frequency", "pd_joint_L", "loop_margins", "delay_margin", "simulate_delayed_loop",
           "motor_cascade_gains", "simulate_cascade", "liu_layland_bound", "response_times", "utilisation"]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def alias_frequency(f, fs):
    """Apparent frequency (Hz, in ``[0, fs/2]``) of a sinusoid of frequency ``f`` sampled at ``fs``."""
    f = np.asarray(f, dtype=float)
    return np.abs(f - fs * np.round(f / fs))


# ---------------------------------------------------------------------------
# Loop margins and delay
# ---------------------------------------------------------------------------
def pd_joint_L(J: float, b: float, Kp: float, Kd: float):
    """Return ``L(w) = (Kp + Kd j w) / (J (j w)^2 + b j w)``, the loop gain of a PD-controlled rigid joint."""
    def L(w):
        s = 1j * np.asarray(w, dtype=float)
        return (Kp + Kd * s) / (J * s**2 + b * s)
    return L


def loop_margins(L, delay: float = 0.0, w=None):
    """Gain-crossover frequency ``wc`` [rad/s] and phase margin [deg] of ``L(w) exp(-j w delay)``.

    The first crossing of ``|L| = 1`` (from above) is used. Returns ``(wc, pm_deg)``; ``(nan, nan)`` if the gain
    never crosses unity on the frequency grid.
    """
    w = np.logspace(-2, 5, 40_000) if w is None else np.asarray(w, dtype=float)
    Lw = L(w) * np.exp(-1j * w * delay)
    mag = np.abs(Lw)
    below = np.where((mag[:-1] >= 1.0) & (mag[1:] < 1.0))[0]
    if len(below) == 0:
        return np.nan, np.nan
    i = below[0]
    f = (np.log(mag[i]) - 0.0) / (np.log(mag[i]) - np.log(mag[i + 1]))
    wc = float(np.exp(np.log(w[i]) + f * (np.log(w[i + 1]) - np.log(w[i]))))
    ph = np.unwrap(np.angle(Lw))
    phase = np.interp(np.log(wc), np.log(w), ph)
    return wc, float(np.rad2deg(phase) + 180.0)


def delay_margin(L, w=None):
    """Largest pure delay ``Td`` [s] for which the loop stays stable: ``phase margin [rad] / wc``."""
    wc, pm = loop_margins(L, 0.0, w)
    return np.deg2rad(pm) / wc


# ---------------------------------------------------------------------------
# Sampled-data simulation with delay and jitter
# ---------------------------------------------------------------------------
def simulate_delayed_loop(Ac, Bc, K, Ts: float, T: float, x0, delay: float = 0.0, jitter: float = 0.0,
                          substeps: int = 20, x_ref=None, dist=None, rng=None, u_max: float | None = None):
    """Continuous plant ``x' = Ac x + Bc (u + d(t))`` under the sampled law ``u_k = -K (x(kTs) - x_ref(kTs))``.

    The state is sampled at ``t_k = k Ts``; the command ``u_k`` is computed instantaneously but *applied*
    ``delay + jitter * U(0, 1)`` seconds later (quantised to the simulation step ``Ts / substeps``) and held until
    the next command arrives. Between events the plant is propagated exactly (matrix exponential), so the only
    approximations are the quantised delay and the held commands.

    Returns ``(t, X, tk, U)``: the fine time grid, states (n x nx), sample times and the commands applied.
    """
    Ac, Bc, K = np.atleast_2d(Ac).astype(float), np.atleast_2d(Bc).astype(float), np.atleast_2d(K).astype(float)
    if Bc.shape[0] != Ac.shape[0]:
        Bc = Bc.T
    nx = Ac.shape[0]
    rng = rng or np.random.default_rng(0)
    h = Ts / substeps
    M = expm(np.block([[Ac, Bc], [np.zeros((Bc.shape[1], nx + Bc.shape[1]))]]) * h)
    Ad, Bd = M[:nx, :nx], M[:nx, nx:]
    n_steps = int(round(T / h))
    n_samples = int(round(T / Ts))
    d_idx = np.round((delay + jitter * rng.random(n_samples)) / h).astype(int)
    act = np.arange(n_samples) * substeps + d_idx                      # fine-step index at which u_k becomes active
    x = np.asarray(x0, dtype=float).copy()
    X = np.empty((n_steps + 1, nx)); X[0] = x
    U = np.zeros(n_samples)
    u_now, k_next = np.zeros(Bc.shape[1]), 0
    pending = {}
    xr = (lambda t: np.zeros(nx)) if x_ref is None else (x_ref if callable(x_ref) else (lambda t, c=np.asarray(x_ref, float): c))
    for i in range(n_steps):
        if i % substeps == 0 and i // substeps < n_samples:            # sampling instant
            k = i // substeps
            u_k = -K @ (x - xr(i * h))
            if u_max is not None:
                u_k = np.clip(u_k, -u_max, u_max)
            U[k] = u_k[0]
            pending[act[k]] = u_k
        if i in pending:
            u_now = pending.pop(i)
        d = np.zeros(Bc.shape[1]) if dist is None else np.atleast_1d(dist(i * h))
        x = Ad @ x + Bd @ (u_now + d)
        X[i + 1] = x
    return np.arange(n_steps + 1) * h, X, np.arange(n_samples) * Ts, U


# ---------------------------------------------------------------------------
# Cascade control of a DC-motor drive
# ---------------------------------------------------------------------------
def motor_cascade_gains(R: float, L: float, Kt: float, J: float, b: float, w_i: float, w_v: float, w_p: float):
    """PI/PI/P gains of the cascade by pole-zero cancellation.

    Current loop (plant ``1/(L s + R)``): ``Kp_i = w_i L``, ``Ki_i = w_i R`` gives the closed loop ``w_i / (s + w_i)``.
    Velocity loop (plant ``Kt/(J s + b)``, current loop treated as unity): ``Kp_v = w_v J / Kt``, ``Ki_v = w_v b / Kt``
    gives ``w_v / (s + w_v)``. Position loop (plant ``1/s``): ``Kp_p = w_p``.
    """
    return dict(Kp_i=w_i * L, Ki_i=w_i * R, Kp_v=w_v * J / Kt, Ki_v=w_v * b / Kt, Kp_p=w_p)


def simulate_cascade(motor: dict, gains: dict, theta_ref, T: float, dt: float = 2e-5, V_max: float = np.inf,
                     I_max: float = np.inf, w_max: float = np.inf, rates=(1, 10, 50), tau_load=None):
    """Simulate position -> velocity -> current -> voltage cascade of a DC motor (explicit Euler, step ``dt``).

    ``motor`` has keys ``R, L, Kt, Ke, J, b``. ``theta_ref`` is a function of time. The loops run every
    ``rates = (n_i, n_v, n_p)`` steps (multi-rate), the integrators use conditional integration (anti-windup)
    and the set-points are saturated at ``w_max`` (speed), ``I_max`` (current) and ``V_max`` (voltage).
    Returns a dict with ``t, theta, omega, i, V, i_ref, w_ref``.
    """
    R, L, Kt, Ke, J, b = (motor[k] for k in ("R", "L", "Kt", "Ke", "J", "b"))
    n = int(round(T / dt))
    out = {k: np.zeros(n) for k in ("theta", "omega", "i", "V", "i_ref", "w_ref")}
    th = om = i = 0.0
    xi_v = xi_i = 0.0
    w_ref = i_ref = V = 0.0
    n_i, n_v, n_p = rates
    for k in range(n):
        t = k * dt
        if k % n_p == 0:
            w_ref = float(np.clip(gains["Kp_p"] * (theta_ref(t) - th), -w_max, w_max))
        if k % n_v == 0:
            e_v = w_ref - om
            u = gains["Kp_v"] * e_v + xi_v
            u_sat = float(np.clip(u, -I_max, I_max))
            if u == u_sat or e_v * u < 0:
                xi_v += gains["Ki_v"] * e_v * dt * n_v
            i_ref = float(np.clip(gains["Kp_v"] * e_v + xi_v, -I_max, I_max))
        if k % n_i == 0:
            e_i = i_ref - i
            u = gains["Kp_i"] * e_i + xi_i
            V = float(np.clip(u, -V_max, V_max))
            if u == V or e_i * u < 0:
                xi_i += gains["Ki_i"] * e_i * dt * n_i
        load = 0.0 if tau_load is None else tau_load(t)
        di = (V - R * i - Ke * om) / L
        dom = (Kt * i - b * om - load) / J
        i += di * dt
        om += dom * dt
        th += om * dt
        for key, val in (("theta", th), ("omega", om), ("i", i), ("V", V), ("i_ref", i_ref), ("w_ref", w_ref)):
            out[key][k] = val
    out["t"] = np.arange(n) * dt
    return out


# ---------------------------------------------------------------------------
# Real-time scheduling
# ---------------------------------------------------------------------------
def utilisation(C, T):
    """Total processor utilisation ``sum C_i / T_i``."""
    return float(np.sum(np.asarray(C, dtype=float) / np.asarray(T, dtype=float)))


def liu_layland_bound(n: int) -> float:
    """Rate-monotonic utilisation bound ``n (2^{1/n} - 1)`` (sufficient, not necessary)."""
    return n * (2.0 ** (1.0 / n) - 1.0)


def response_times(C, T, D=None, blocking=None):
    """Worst-case response times of independent periodic tasks under fixed-priority preemptive scheduling.

    Tasks are listed from highest to lowest priority (for rate-monotonic priorities sort by period). ``C`` are the
    worst-case execution times, ``T`` the periods, ``D`` the deadlines (default ``T``) and ``blocking`` an optional
    extra delay per task (for example a non-preemptible section). Solves ``R_i = C_i + B_i + sum_{j<i} ceil(R_i/T_j) C_j``
    by fixed-point iteration; returns ``inf`` for a task whose response exceeds its deadline.
    """
    C, T = np.asarray(C, dtype=float), np.asarray(T, dtype=float)
    D = T if D is None else np.asarray(D, dtype=float)
    B = np.zeros(len(C)) if blocking is None else np.asarray(blocking, dtype=float)
    R = np.empty(len(C))
    for i in range(len(C)):
        r = C[i] + B[i]
        while True:
            r_new = C[i] + B[i] + np.sum(np.ceil(r / T[:i]) * C[:i])
            if r_new > D[i] + 1e-12:
                r = np.inf
                break
            if abs(r_new - r) < 1e-12:
                r = r_new
                break
            r = r_new
        R[i] = r
    return R

"""Models of robot sensors and the estimators that clean up their signals.

This module backs Chapter 8 of the Robot Motion Control notebooks
(``RMC/08-sensing-and-implementation``). It is pure NumPy/SciPy: nothing here
needs a simulator, so every function can be unit-tested against a closed form.

Contents
--------
* **Encoders** — :class:`IncrementalEncoder`, :func:`quadrature_signals`,
  :func:`quadrature_decode`, Gray code (:func:`gray_encode`, :func:`gray_decode`)
  and :func:`read_absolute` (a track-by-track model of an absolute disc).
* **Velocity from position** — :func:`velocity_M` (finite difference),
  :func:`velocity_T` (pulse period), :func:`velocity_MT` (both),
  :func:`lowpass_derivative`, :func:`alpha_beta_filter`, :func:`kalman_filter`
  and the steady-state constant-velocity Kalman gains :func:`cv_kalman_gains`.
* **Force/torque** — :func:`bridge_output`, :func:`quarter_bridge`,
  :func:`calibrate_wrench` and :func:`estimate_payload` (tool weight and centre of
  mass from a static wrench sensor).
* **Inertial sensing** — :func:`simulate_gyro`, :func:`tilt_from_accel`,
  :func:`complementary_filter`, :func:`kalman_angle_bias`, :func:`allan_deviation`.

Conventions
-----------
Angles are radians, times seconds. "Counts" are integers. Random generators are
always passed in (``rng = np.random.default_rng(seed)``) so results are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

__all__ = [
    "quantize", "IncrementalEncoder", "quadrature_signals", "quadrature_decode",
    "gray_encode", "gray_decode", "read_absolute",
    "edge_times", "velocity_M", "velocity_T", "velocity_MT",
    "lowpass_derivative", "alpha_beta_filter", "alpha_beta_gains", "kalman_filter",
    "cv_kalman_model", "cv_kalman_gains",
    "bridge_output", "quarter_bridge", "calibrate_wrench", "apply_calibration", "estimate_payload",
    "simulate_gyro", "tilt_from_accel", "complementary_filter", "kalman_angle_bias",
    "allan_deviation",
]


# ---------------------------------------------------------------------------
# Quantisation and encoders
# ---------------------------------------------------------------------------
def quantize(x, step: float, mode: str = "floor"):
    """Quantise ``x`` to an integer multiple of ``step``.

    ``mode="floor"`` models a counter (the reading is the last level crossed);
    ``mode="round"`` models a mid-tread converter whose error is zero-mean.
    """
    x = np.asarray(x, dtype=float)
    if mode == "floor":
        return step * np.floor(x / step)
    if mode == "round":
        return step * np.floor(x / step + 0.5)
    raise ValueError("mode must be 'floor' or 'round'")


@dataclass(frozen=True)
class IncrementalEncoder:
    """An incremental (quadrature) optical encoder.

    ``lines`` is the number of lines (slots) per revolution on channel A.
    ``decoding`` is 1, 2 or 4: how many of the four edges per line are counted.
    """

    lines: int
    decoding: int = 4

    @property
    def counts_per_rev(self) -> int:
        return self.lines * self.decoding

    @property
    def resolution(self) -> float:
        """Angle represented by one count, in radians."""
        return 2.0 * np.pi / self.counts_per_rev

    def counts(self, theta):
        """Integer count after the shaft has turned to angle ``theta`` (counter started at 0)."""
        return np.floor(np.asarray(theta, dtype=float) / self.resolution).astype(np.int64)

    def angle(self, counts):
        """Angle reported by the controller for a given count (the low edge of the bin)."""
        return np.asarray(counts, dtype=float) * self.resolution


def quadrature_signals(theta, lines: int):
    """Channels A and B (0/1 square waves in quadrature) for shaft angle ``theta``.

    One electrical period spans ``2*pi/lines`` of shaft rotation. B lags A by a quarter
    period when ``theta`` increases, so the state sequence is the Gray cycle
    (A,B) = 00 -> 10 -> 11 -> 01 -> 00 for forward rotation.
    """
    ph = np.mod(np.asarray(theta, dtype=float) * lines / (2.0 * np.pi), 1.0)   # electrical phase in [0, 1)
    A = ((ph >= 0.25) & (ph < 0.75)).astype(int)
    B = (ph >= 0.5).astype(int)
    return A, B


# state = 2*A + B ; forward Gray sequence 0 (00) -> 2 (10) -> 3 (11) -> 1 (01) -> 0
_FORWARD = {0: 2, 2: 3, 3: 1, 1: 0}


def quadrature_decode(A, B):
    """Cumulative x4 count from sampled channels A and B (a Gray-code state machine).

    A transition to the next state of the forward cycle adds one, to the previous state
    subtracts one, no change adds zero, and a two-state jump (a missed sample) is
    flagged by adding zero — the loss that a too-slow sampling rate would cause.
    """
    s = 2 * np.asarray(A, dtype=int) + np.asarray(B, dtype=int)
    counts = np.zeros(len(s), dtype=np.int64)
    for k in range(1, len(s)):
        a, b = s[k - 1], s[k]
        step = 0
        if a != b:
            if _FORWARD[a] == b:
                step = 1
            elif _FORWARD[b] == a:
                step = -1
        counts[k] = counts[k - 1] + step
    return counts


def gray_encode(n):
    """Binary-reflected Gray code of the non-negative integer(s) ``n``."""
    n = np.asarray(n, dtype=np.int64)
    return n ^ (n >> 1)


def gray_decode(g):
    """Inverse of :func:`gray_encode`."""
    g = np.asarray(g, dtype=np.int64)
    n = g.copy()
    shift = 1
    while np.any(g >> shift):
        n = n ^ (g >> shift)
        shift += 1
    return n


def read_absolute(theta, bits: int, code: str = "gray", offsets=None):
    """Reading of an absolute single-turn disc with ``bits`` tracks.

    Track ``b`` (``b = 0`` the finest) is read by its own head, displaced by
    ``offsets[b]`` *least-significant-bit widths* from the ideal position — the
    mechanical misalignment that makes plain binary discs unreliable. Returns the
    integer reading (decoded back to binary if ``code="gray"``).
    """
    theta = np.asarray(theta, dtype=float)
    n_codes = 2 ** bits
    q = 2.0 * np.pi / n_codes
    offsets = np.zeros(bits) if offsets is None else np.asarray(offsets, dtype=float)
    word = np.zeros(theta.shape, dtype=np.int64)
    for b in range(bits):
        n_b = np.floor(np.mod(theta / q + offsets[b], n_codes)).astype(np.int64)
        pattern = n_b if code == "binary" else gray_encode(n_b)
        word |= ((pattern >> b) & 1) << b
    return word if code == "binary" else gray_decode(word)


# ---------------------------------------------------------------------------
# Velocity from a quantised position
# ---------------------------------------------------------------------------
def edge_times(t, theta, res: float):
    """Times at which a monotonically increasing ``theta(t)`` crosses a multiple of ``res``.

    ``t`` is a (fine) time grid and ``theta`` the true angle on it; crossings are located
    by linear interpolation, which is exact for constant velocity between grid points.
    """
    t = np.asarray(t, dtype=float)
    lev = np.asarray(theta, dtype=float) / res
    k0, k1 = int(np.floor(lev[0])) + 1, int(np.floor(lev[-1]))
    if k1 < k0:
        return np.empty(0)
    levels = np.arange(k0, k1 + 1)
    return np.interp(levels, lev, t)


def velocity_M(counts, res: float, Ts: float):
    """M method (finite difference): ``(n_k - n_{k-1}) * res / Ts`` — quantum ``res/Ts``."""
    n = np.asarray(counts, dtype=float)
    v = np.zeros_like(n)
    v[1:] = np.diff(n) * res / Ts
    return v


def _clock(t, f_clk):
    return t if f_clk is None else np.floor(np.asarray(t) * f_clk) / f_clk


def velocity_T(edges, t_samples, res: float, f_clk: float | None = None):
    """T method: ``res`` divided by the clock-measured interval between the last two edges.

    Returns zero until two edges have been seen. ``f_clk`` is the timer frequency in Hz
    (``None`` = ideal clock).
    """
    edges = np.asarray(edges, dtype=float)
    t_samples = np.asarray(t_samples, dtype=float)
    idx = np.searchsorted(edges, t_samples, side="right")          # number of edges <= t
    v = np.zeros(len(t_samples))
    ok = idx >= 2
    dt = _clock(edges[idx[ok] - 1], f_clk) - _clock(edges[idx[ok] - 2], f_clk)
    v[ok] = np.where(dt > 0, res / np.where(dt > 0, dt, 1.0), 0.0)
    return v


def velocity_MT(edges, t_samples, res: float, f_clk: float | None = None):
    """M/T method: counts between the last edges before consecutive samples, divided by their time.

    ``v_k = (n_k - n_{k-1}) res / (tau_k - tau_{k-1})`` where ``tau_k`` is the time of the last edge
    at or before sample ``k``. It has the T method's resolution at low speed and the M method's
    regularity at high speed. If no edge occurred in a sample period the previous estimate is
    held for one period and then decays to zero (the shaft is declared stopped).
    """
    edges = np.asarray(edges, dtype=float)
    t_samples = np.asarray(t_samples, dtype=float)
    n = np.searchsorted(edges, t_samples, side="right")
    v = np.zeros(len(t_samples))
    for k in range(1, len(t_samples)):
        if n[k] >= 1 and n[k - 1] >= 1 and n[k] > n[k - 1]:
            tau1, tau0 = _clock(edges[n[k] - 1], f_clk), _clock(edges[n[k - 1] - 1], f_clk)
            v[k] = (n[k] - n[k - 1]) * res / (tau1 - tau0)
        elif n[k] >= 1:                         # no new edge: speed cannot exceed res / (time since last edge)
            elapsed = t_samples[k] - edges[n[k] - 1]
            v[k] = min(v[k - 1], res / elapsed) if elapsed > 0 else v[k - 1]
    return v


def lowpass_derivative(x, Ts: float, tau: float):
    """Filtered derivative ``s / (tau s + 1)`` (backward Euler): ``y_k = (tau y_{k-1} + x_k - x_{k-1}) / (tau + Ts)``."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    for k in range(1, len(x)):
        y[k] = (tau * y[k - 1] + x[k] - x[k - 1]) / (tau + Ts)
    return y


def alpha_beta_gains(omega: float, Ts: float):
    """alpha, beta of a critically damped (double-pole) alpha-beta filter with observer bandwidth ``omega`` [rad/s].

    Both error poles sit at ``p = exp(-omega Ts)``: ``alpha = 1 - p^2``, ``beta = (1 - p)^2``.
    """
    p = np.exp(-omega * Ts)
    return 1.0 - p * p, (1.0 - p) ** 2


def alpha_beta_filter(z, Ts: float, alpha: float, beta: float, x0: float | None = None, v0: float = 0.0):
    """Constant-velocity alpha-beta tracker; returns ``(position_estimate, velocity_estimate)``.

    Predict ``x^- = x + Ts v``; correct with the innovation ``r = z - x^-``:
    ``x = x^- + alpha r``, ``v = v + (beta / Ts) r``.
    """
    z = np.asarray(z, dtype=float)
    x, v = (z[0] if x0 is None else x0), v0
    xs, vs = np.empty_like(z), np.empty_like(z)
    xs[0], vs[0] = x, v
    for k in range(1, len(z)):
        xp = x + Ts * v
        r = z[k] - xp
        x = xp + alpha * r
        v = v + beta * r / Ts
        xs[k], vs[k] = x, v
    return xs, vs


def kalman_filter(F, H, Q, R, z, x0, P0):
    """Standard linear Kalman filter. ``z`` has shape (N, m); returns states (N, n) and covariances (N, n, n).

    ``x_{k+1} = F x_k + w``, ``w ~ N(0, Q)``;  ``z_k = H x_k + v``, ``v ~ N(0, R)``.
    The first measurement is used to *update* the prior ``(x0, P0)``.
    """
    F, H, Q, R = (np.atleast_2d(np.asarray(a, dtype=float)) for a in (F, H, Q, R))
    z = np.asarray(z, dtype=float)
    z = z.reshape(len(z), -1)
    x, P = np.asarray(x0, dtype=float).copy(), np.asarray(P0, dtype=float).copy()
    xs, Ps = np.empty((len(z), len(x))), np.empty((len(z), len(x), len(x)))
    I = np.eye(len(x))
    for k in range(len(z)):
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (z[k] - H @ x)
        P = (I - K @ H) @ P
        xs[k], Ps[k] = x, P
        x = F @ x
        P = F @ P @ F.T + Q
    return xs, Ps


def cv_kalman_model(Ts: float, q: float, r: float):
    """Constant-velocity model driven by white acceleration noise of spectral density ``q`` [rad^2/s^3].

    Returns ``(F, H, Q, R)`` for the state ``[position, velocity]`` and a position measurement of
    variance ``r``. ``Q = q [[Ts^3/3, Ts^2/2], [Ts^2/2, Ts]]`` is the exact discretisation.
    """
    F = np.array([[1.0, Ts], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q * np.array([[Ts**3 / 3.0, Ts**2 / 2.0], [Ts**2 / 2.0, Ts]])
    return F, H, Q, np.array([[r]])


def cv_kalman_gains(Ts: float, q: float, r: float):
    """Steady-state Kalman gain ``K = [K_pos, K_vel]`` of :func:`cv_kalman_model` (a-priori form converted
    to the *update* gain used by :func:`kalman_filter`) and the steady-state a-posteriori covariance."""
    F, H, Q, R = cv_kalman_model(Ts, q, r)
    Pm = solve_discrete_are(F.T, H.T, Q, R)          # a-priori covariance
    S = H @ Pm @ H.T + R
    K = Pm @ H.T @ np.linalg.inv(S)
    P = (np.eye(2) - K @ H) @ Pm
    return K.ravel(), P


# ---------------------------------------------------------------------------
# Strain gauges and force/torque sensors
# ---------------------------------------------------------------------------
def bridge_output(Vex: float, R1, R2, R3, R4):
    """Wheatstone-bridge output ``Vex * (R3/(R3+R4) - R2/(R1+R2))``.

    The excitation is applied across the series pairs (R1, R2) and (R4, R3); the output is
    taken between the two mid-points.
    """
    return Vex * (R3 / (R3 + R4) - R2 / (R1 + R2))


def quarter_bridge(Vex: float, strain, gauge_factor: float = 2.0, R0: float = 350.0):
    """Output of a quarter bridge (one active gauge in arm R4, three fixed resistors of value ``R0``).

    Exact expression; to first order ``-GF * strain * Vex / 4`` (the sign is a matter of which arm is active).
    """
    strain = np.asarray(strain, dtype=float)
    return bridge_output(Vex, R0, R0, R0, R0 * (1.0 + gauge_factor * strain))


def calibrate_wrench(V, W, offset: bool = True):
    """Least-squares calibration of a six-axis force/torque sensor.

    ``V`` (n x 6) are raw bridge readings, ``W`` (n x 6) the applied wrenches. Fits
    ``W = C V + w0`` and returns ``(C, w0)``; the matrix ``C`` is the calibration matrix and ``w0`` the
    zero-load offset. At least 7 independent loads are needed with an offset (6 without).
    """
    V, W = np.asarray(V, dtype=float), np.asarray(W, dtype=float)
    X = np.hstack([V, np.ones((len(V), 1))]) if offset else V
    coef, *_ = np.linalg.lstsq(X, W, rcond=None)
    if offset:
        return coef[:-1].T, coef[-1]
    return coef.T, np.zeros(W.shape[1])


def apply_calibration(C, w0, V):
    """Wrench(es) for raw readings ``V``: ``C V + w0`` (rows are samples)."""
    return np.asarray(V, dtype=float) @ np.asarray(C).T + np.asarray(w0)


def _skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def estimate_payload(R_list, F_list, T_list, g=(0.0, 0.0, -9.81)):
    """Tool mass, centre of mass and sensor offsets from a *static* wrench sensor.

    The sensor reports the wrench exerted by its support on the tool side (the MuJoCo ``force`` and
    ``torque`` sensors use this sign). With ``R_i`` the orientation of the sensor frame in the world at
    pose ``i``, ``g_i = R_i^T g`` and the tool mass ``m`` with centre of mass ``c`` (sensor frame):

    ``F_i = -m g_i + f0``,   ``T_i = c x (-m g_i) + t0 = [g_i]_x (m c) + t0``.

    Both equations are linear in ``(m, m c, f0, t0)``; the 8 unknowns are found by least squares from
    at least three poses whose gravity directions are not all parallel. Returns
    ``(m, c, f0, t0)``.
    """
    g = np.asarray(g, dtype=float)
    rows, rhs = [], []
    for R, F, T in zip(R_list, F_list, T_list):
        gi = np.asarray(R, dtype=float).T @ g
        A_F = np.hstack([-gi.reshape(3, 1), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))])
        A_T = np.hstack([np.zeros((3, 1)), _skew(gi), np.zeros((3, 3)), np.eye(3)])
        rows += [A_F, A_T]
        rhs += [np.asarray(F, dtype=float), np.asarray(T, dtype=float)]
    x, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
    m = x[0]
    return float(m), x[1:4] / m, x[4:7], x[7:10]


# ---------------------------------------------------------------------------
# Inertial sensing
# ---------------------------------------------------------------------------
def simulate_gyro(omega, Ts: float, rng, arw: float = 0.0, bias0: float = 0.0, bias_walk: float = 0.0,
                  scale_error: float = 0.0):
    """Rate-gyro measurement of the true angular rate ``omega`` (one axis).

    ``y_k = (1 + s) omega_k + b_k + n_k`` with white noise of standard deviation ``arw / sqrt(Ts)``
    (``arw`` is the *angle random walk* coefficient in rad/s/sqrt(Hz) = rad/sqrt(s)) and a bias that
    performs a random walk with increments of standard deviation ``bias_walk * sqrt(Ts)``
    (``bias_walk`` in rad/s^2/sqrt(Hz)). Returns ``(y, b)``.
    """
    omega = np.asarray(omega, dtype=float)
    n = arw / np.sqrt(Ts) * rng.standard_normal(len(omega))
    b = bias0 + np.cumsum(np.concatenate([[0.0], bias_walk * np.sqrt(Ts) * rng.standard_normal(len(omega) - 1)]))
    return (1.0 + scale_error) * omega + b + n, b


def tilt_from_accel(a_forward, a_up):
    """Tilt angle ``atan2(a_forward, a_up)`` from the two accelerometer axes in a vertical plane.

    At rest an accelerometer reads the *reaction to gravity* (``+g`` along the world's up direction), so
    for a sensor tilted by ``theta`` about its y axis the reading is ``(g sin(theta), g cos(theta))``.
    """
    return np.arctan2(a_forward, a_up)


def complementary_filter(gyro, angle_acc, Ts: float, tau: float, theta0: float | None = None):
    """First-order complementary filter: gyro integration at high frequency, accelerometer at low.

    ``theta_k = a (theta_{k-1} + Ts gyro_k) + (1 - a) angle_acc_k`` with ``a = tau / (tau + Ts)``. Its transfer
    function from the true angle to the estimate through the accelerometer channel is the low-pass
    ``1/(tau s + 1)`` and through the gyro channel the high-pass ``tau s/(tau s + 1)``; the two add to one.
    """
    gyro, angle_acc = np.asarray(gyro, dtype=float), np.asarray(angle_acc, dtype=float)
    a = tau / (tau + Ts)
    th = np.empty(len(gyro))
    th[0] = angle_acc[0] if theta0 is None else theta0
    for k in range(1, len(gyro)):
        th[k] = a * (th[k - 1] + Ts * gyro[k]) + (1.0 - a) * angle_acc[k]
    return th


def kalman_angle_bias(gyro, angle_acc, Ts: float, arw: float, bias_walk: float, r_acc: float,
                      theta0: float | None = None, p0=(1.0, 1e-2)):
    """Two-state Kalman filter for angle and gyro bias, with the gyro as control input.

    State ``[theta, b]``: ``theta_{k+1} = theta_k + Ts (gyro_k - b_k)``, ``b_{k+1} = b_k + w``; the measurement is
    the accelerometer angle with variance ``r_acc``. Process noise is ``diag(arw^2 Ts, bias_walk^2 Ts)``
    (angle noise from the integrated gyro white noise; bias random walk). Returns ``(theta, bias)``.
    """
    gyro, angle_acc = np.asarray(gyro, dtype=float), np.asarray(angle_acc, dtype=float)
    F = np.array([[1.0, -Ts], [0.0, 1.0]])
    Bu = np.array([Ts, 0.0])
    Q = np.diag([arw**2 * Ts, bias_walk**2 * Ts])
    H = np.array([[1.0, 0.0]])
    x = np.array([angle_acc[0] if theta0 is None else theta0, 0.0])
    P = np.diag(p0).astype(float)
    th, bi = np.empty(len(gyro)), np.empty(len(gyro))
    th[0], bi[0] = x
    for k in range(1, len(gyro)):
        x = F @ x + Bu * gyro[k]                     # predict with the gyro sample of this interval
        P = F @ P @ F.T + Q
        S = (H @ P @ H.T)[0, 0] + r_acc              # update with the accelerometer
        K = (P @ H.T)[:, 0] / S
        x = x + K * (angle_acc[k] - x[0])
        P = (np.eye(2) - np.outer(K, H[0])) @ P
        th[k], bi[k] = x
    return th, bi


def allan_deviation(x, Ts: float, m_list=None):
    """Overlapping Allan deviation of a rate signal ``x`` sampled every ``Ts`` seconds.

    Returns ``(taus, adev)``. For white rate noise of density ``N`` (rad/s/sqrt(Hz)) the curve is
    ``N / sqrt(tau)`` (slope -1/2); a bias random walk gives slope +1/2; the flat minimum is the bias instability.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if m_list is None:
        m_list = np.unique(np.floor(np.logspace(0, np.log10(n // 9), 30)).astype(int))
    c = np.concatenate([[0.0], np.cumsum(x)])
    taus, adev = [], []
    for m in m_list:
        if 2 * m >= n:
            continue
        means = (c[m:] - c[:-m]) / m                 # all overlapping cluster averages
        d = means[m:] - means[:-m]
        taus.append(m * Ts)
        adev.append(np.sqrt(0.5 * np.mean(d**2)))
    return np.array(taus), np.array(adev)

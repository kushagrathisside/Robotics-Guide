"""Trajectory generation: polynomials, via points, trapezoidal and S-curve profiles, Cartesian paths, time scaling.

This module backs Chapter 7 of MFoR (Craig, *Introduction to Robotics*, Ch. 7; Lynch & Park, Ch. 9).
It contains no plotting and no simulator code, so every function can be tested in isolation.

Vocabulary
----------
* A **path** is a curve in joint space or in task space, without any timing.
* A **trajectory** is a path together with a timing law; here every trajectory is a function ``q(t)``
  on a closed interval ``[t0, t1]`` with derivatives up to the jerk.
* Joint coordinates may be scalar or vectors; every trajectory object reports ``n`` (number of
  coordinates) and returns arrays of shape ``t.shape + (n,)`` (or ``t.shape`` for a scalar trajectory).

Conventions
-----------
* Polynomial coefficients are stored in **ascending** powers of the *local* time ``tau = t - t_knot``.
* Outside ``[t0, t1]`` a trajectory holds its end positions and has zero velocity, acceleration and jerk.
* Angles are in radians, times in seconds.
"""

from __future__ import annotations

from math import factorial

import numpy as np

from . import transforms as _tf

__all__ = [
    # base classes and wrappers
    "Trajectory", "PiecewisePoly", "Stack", "JointLine", "TimeScaled", "concatenate",
    # polynomials
    "hermite_coeffs", "cubic", "quintic", "septic", "linear",
    # via points
    "heuristic_velocities", "spline_velocities", "via_cubic", "via_quintic", "via_stop", "chord_times",
    # trapezoidal profiles
    "lspb", "lspb_limits", "min_time_trapezoid", "bang_bang", "lspb_via",
    "scurve", "coordinated_ptp",
    # Cartesian paths
    "rotation_geodesic", "euler_interp", "matrix_lerp", "pose_interp", "CartesianLine",
    "pose_error", "dls_step", "track_poses",
    # time scaling and execution
    "Path", "ScaledPath", "time_optimal_scaling", "required_time_stretch", "peak_values",
    "check_limits", "servo_times", "OnlinePTP",
]


# ---------------------------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------------------------
class Trajectory:
    """Common interface: ``pos``, ``vel``, ``acc``, ``jerk`` and ``sample``.

    Subclasses define ``t0``, ``t1``, ``n``, ``scalar`` and ``_eval(t, d)``, which returns the ``d``-th time
    derivative at times ``t`` (a 1-D array inside ``[t0, t1]``) with shape ``(len(t), n)``.
    """

    t0: float = 0.0
    t1: float = 1.0
    n: int = 1
    scalar: bool = True

    @property
    def duration(self) -> float:
        return float(self.t1 - self.t0)

    def _eval(self, t, d):  # pragma: no cover - abstract
        raise NotImplementedError

    def _deriv(self, t, d):
        t = np.asarray(t, dtype=float)
        flat = t.ravel()
        inside = (flat >= self.t0 - 1e-12) & (flat <= self.t1 + 1e-12)
        out = self._eval(np.clip(flat, self.t0, self.t1), d)
        if d > 0:
            out = out * inside[:, None]
        out = out.reshape(t.shape + (self.n,))
        return out[..., 0] if self.scalar else out

    def pos(self, t):
        """Position ``q(t)``."""
        return self._deriv(t, 0)

    def vel(self, t):
        """Velocity ``dq/dt``."""
        return self._deriv(t, 1)

    def acc(self, t):
        """Acceleration ``d2q/dt2``."""
        return self._deriv(t, 2)

    def jerk(self, t):
        """Jerk ``d3q/dt3`` (zero inside constant-acceleration pieces; impulses at junctions are not represented)."""
        return self._deriv(t, 3)

    def __call__(self, t):
        return self.pos(t), self.vel(t), self.acc(t)

    def sample(self, rate=None, dt=None):
        """Samples at a servo ``rate`` (Hz) or step ``dt`` (s): returns ``t, q, qd, qdd``."""
        dt = 1.0 / rate if dt is None else dt
        t = servo_times(self.duration, dt=dt) + self.t0
        return t, self.pos(t), self.vel(t), self.acc(t)


def _ff(j, d):
    """Falling factorial ``j (j-1) ... (j-d+1)``: the factor produced by differentiating ``tau**j`` ``d`` times."""
    out = 1.0
    for i in range(d):
        out *= (j - i)
    return out


class PiecewisePoly(Trajectory):
    """Piecewise polynomial in local time.

    Parameters
    ----------
    times : knot times ``t_0 < t_1 < ... < t_m`` (length ``m+1``).
    coeffs : array ``(m, n, k)``; ``coeffs[i, j, p]`` multiplies ``(t - t_i)**p`` for coordinate ``j`` on segment ``i``.
    scalar : if True the trajectory has one coordinate and returns plain arrays.
    info : free-form dictionary describing how the trajectory was built (blend times, peak velocity, ...).
    """

    def __init__(self, times, coeffs, scalar=False, info=None):
        self.times = np.asarray(times, dtype=float)
        self.coeffs = np.asarray(coeffs, dtype=float)
        if self.coeffs.ndim != 3 or len(self.times) != self.coeffs.shape[0] + 1:
            raise ValueError("coeffs must have shape (m, n, k) for m+1 knot times")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("knot times must be strictly increasing")
        self.t0, self.t1 = float(self.times[0]), float(self.times[-1])
        self.n = self.coeffs.shape[1]
        self.scalar = bool(scalar)
        self.info = dict(info or {})

    def _eval(self, t, d):
        m, _, k = self.coeffs.shape
        idx = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, m - 1)
        tau = t - self.times[idx]
        c = self.coeffs[idx]                                  # (len(t), n, k)
        out = np.zeros((len(t), self.n))
        for p in range(d, k):
            out += c[:, :, p] * (_ff(p, d) * tau ** (p - d))[:, None]
        return out

    def shifted(self, dt):
        """The same trajectory delayed by ``dt``."""
        return PiecewisePoly(self.times + dt, self.coeffs, self.scalar, self.info)

    def segment_coeffs(self, i):
        """Coefficients (ascending, local time) of segment ``i``: array ``(n, k)``."""
        return self.coeffs[i]


class Stack(Trajectory):
    """Several scalar trajectories side by side (independent joints, possibly with different knots)."""

    def __init__(self, trajs):
        self.trajs = list(trajs)
        if not all(tr.scalar for tr in self.trajs):
            raise ValueError("Stack combines scalar trajectories")
        self.n = len(self.trajs)
        self.scalar = False
        self.t0 = min(tr.t0 for tr in self.trajs)
        self.t1 = max(tr.t1 for tr in self.trajs)

    def _eval(self, t, d):
        return np.stack([tr._deriv(t, d) for tr in self.trajs], axis=-1)


class JointLine(Trajectory):
    """Motion along the straight joint-space segment ``q0 -> q1`` with scalar timing law ``s(t)``, ``s: 0 -> 1``.

    ``q(t) = q0 + s(t) (q1 - q0)``.  Every joint follows the same normalised profile, so all joints start,
    accelerate, cruise and stop together.
    """

    def __init__(self, q0, q1, s):
        self.q0 = np.atleast_1d(np.asarray(q0, dtype=float))
        self.q1 = np.atleast_1d(np.asarray(q1, dtype=float))
        self.s = s
        self.n = self.q0.size
        self.scalar = np.ndim(q0) == 0
        self.t0, self.t1 = s.t0, s.t1

    def _eval(self, t, d):
        sd = self.s._eval(t, d)                               # (len(t), 1)
        dq = self.q1 - self.q0
        return self.q0 + sd * dq if d == 0 else sd * dq


class TimeScaled(Trajectory):
    """``traj`` slowed down (``lam > 1``) or sped up (``lam < 1``) by the factor ``lam``: ``q(t) = q_0((t - t0)/lam)``.

    Position is unchanged, velocity scales as ``1/lam``, acceleration as ``1/lam**2``, jerk as ``1/lam**3``.
    """

    def __init__(self, traj, lam):
        if lam <= 0:
            raise ValueError("time-stretch factor must be positive")
        self.base, self.lam = traj, float(lam)
        self.n, self.scalar = traj.n, traj.scalar
        self.t0, self.t1 = traj.t0, traj.t0 + lam * (traj.t1 - traj.t0)

    def _eval(self, t, d):
        return self.base._eval(self.base.t0 + (t - self.t0) / self.lam, d) / self.lam ** d


def concatenate(trajs):
    """Chain piecewise-polynomial trajectories one after another (each starts where the previous ends in time)."""
    trajs = list(trajs)
    n, scalar = trajs[0].n, trajs[0].scalar
    k = max(tr.coeffs.shape[2] for tr in trajs)
    times, coeffs, t = [0.0], [], 0.0
    for tr in trajs:
        c = np.zeros((tr.coeffs.shape[0], n, k))
        c[:, :, :tr.coeffs.shape[2]] = tr.coeffs
        coeffs.append(c)
        times.extend(t + (tr.times[1:] - tr.times[0]))
        t = times[-1]
    return PiecewisePoly(times, np.concatenate(coeffs, axis=0), scalar)


# ---------------------------------------------------------------------------------------------
# Polynomials: Hermite interpolation of boundary derivatives
# ---------------------------------------------------------------------------------------------
def hermite_coeffs(T, start, end):
    """Coefficients (ascending) of the polynomial of degree ``2m-1`` matching ``m`` derivatives at both ends.

    ``start[d]`` and ``end[d]`` are the ``d``-th derivatives at ``t = 0`` and ``t = T`` (``d = 0 .. m-1``); they
    may be arrays (one entry per coordinate).  ``m = 2`` gives the cubic, ``m = 3`` the quintic, ``m = 4`` the septic.
    The first ``m`` coefficients follow directly from the conditions at ``t = 0``; the other ``m`` come from
    one ``m x m`` linear system.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    m = start.shape[0]
    deg = 2 * m - 1
    c = np.zeros((deg + 1,) + start.shape[1:])
    for k in range(m):
        c[k] = start[k] / factorial(k)
    A = np.zeros((m, m))
    rhs = np.zeros((m,) + start.shape[1:])
    for d in range(m):
        for j in range(m, deg + 1):
            A[d, j - m] = _ff(j, d) * T ** (j - d)
        known = sum(_ff(k, d) * T ** (k - d) * c[k] for k in range(d, m))
        rhs[d] = end[d] - known
    c[m:] = np.linalg.solve(A, rhs.reshape(m, -1)).reshape(rhs.shape)
    return c


def _build(T, bcs0, bcs1):
    scalar = all(np.ndim(b) == 0 for b in list(bcs0) + list(bcs1))
    arrs = np.broadcast_arrays(*[np.atleast_1d(np.asarray(b, dtype=float)) for b in list(bcs0) + list(bcs1)])
    m = len(bcs0)
    start, end = np.stack(arrs[:m]), np.stack(arrs[m:])
    c = hermite_coeffs(T, start, end)                         # (k, n)
    return PiecewisePoly([0.0, T], c.T[None], scalar)


def linear(q0, qf, T):
    """Straight-line interpolation (velocity jumps at both ends: infinite acceleration)."""
    q0, qf = np.asarray(q0, dtype=float), np.asarray(qf, dtype=float)
    scalar = q0.ndim == 0 and qf.ndim == 0
    q0, qf = np.broadcast_arrays(np.atleast_1d(q0), np.atleast_1d(qf))
    c = np.stack([q0, (qf - q0) / T], axis=-1)
    return PiecewisePoly([0.0, T], c[None], scalar)


def cubic(q0, qf, T, v0=0.0, vf=0.0):
    """Cubic from ``q0`` to ``qf`` in time ``T`` with end velocities ``v0``, ``vf`` (Craig's cubic)."""
    return _build(T, [q0, v0], [qf, vf])


def quintic(q0, qf, T, v0=0.0, vf=0.0, a0=0.0, af=0.0):
    """Quintic with prescribed end positions, velocities and accelerations."""
    return _build(T, [q0, v0, a0], [qf, vf, af])


def septic(q0, qf, T, v0=0.0, vf=0.0, a0=0.0, af=0.0, j0=0.0, jf=0.0):
    """Degree-7 polynomial with prescribed end positions, velocities, accelerations and jerks."""
    return _build(T, [q0, v0, a0, j0], [qf, vf, af, jf])


# ---------------------------------------------------------------------------------------------
# Via points
# ---------------------------------------------------------------------------------------------
def _pts(points):
    q = np.asarray(points, dtype=float)
    scalar = q.ndim == 1
    return (q[:, None] if scalar else q), scalar


def chord_times(points, speed=1.0, power=1.0):
    """Knot times proportional to ``|q_{k+1} - q_k|**power`` (``power = 1``: chord length; ``0.5``: centripetal)."""
    q, _ = _pts(points)
    h = np.linalg.norm(np.diff(q, axis=0), axis=1) ** power / speed
    return np.concatenate([[0.0], np.cumsum(h)])


def heuristic_velocities(points, times, v0=0.0, vf=0.0):
    """Craig's heuristic via-point velocities.

    At an interior point the velocity is the *average of the two adjacent chord slopes* if they have the same sign
    and zero if the slope changes sign (the joint reverses direction there).
    """
    q, scalar = _pts(points)
    h = np.diff(np.asarray(times, dtype=float))
    s = np.diff(q, axis=0) / h[:, None]
    v = np.zeros_like(q)
    v[0], v[-1] = v0, vf
    for i in range(1, len(q) - 1):
        v[i] = np.where(s[i - 1] * s[i] > 0, 0.5 * (s[i - 1] + s[i]), 0.0)
    return v[:, 0] if scalar else v


def spline_velocities(points, times, bc="clamped", v0=0.0, vf=0.0):
    """Knot velocities that make a cubic through the points ``C^2`` (continuous acceleration).

    Interior rows of the tridiagonal system (``h_k = t_{k+1} - t_k``, ``s_k`` the chord slope of segment ``k``)::

        h_k v_{k-1} + 2 (h_{k-1} + h_k) v_k + h_{k-1} v_{k+1} = 3 (h_k s_{k-1} + h_{k-1} s_k)

    ``bc = "clamped"`` prescribes the end velocities ``v0, vf``; ``bc = "natural"`` makes the end accelerations zero.
    """
    q, scalar = _pts(points)
    t = np.asarray(times, dtype=float)
    h = np.diff(t)
    s = np.diff(q, axis=0) / h[:, None]
    N = len(q)
    A, b = np.zeros((N, N)), np.zeros((N, q.shape[1]))
    for k in range(1, N - 1):
        A[k, k - 1], A[k, k], A[k, k + 1] = h[k], 2 * (h[k - 1] + h[k]), h[k - 1]
        b[k] = 3 * (h[k] * s[k - 1] + h[k - 1] * s[k])
    if bc == "clamped":
        A[0, 0], A[-1, -1] = 1.0, 1.0
        b[0], b[-1] = v0, vf
    elif bc == "natural":
        A[0, 0], A[0, 1], b[0] = 2.0, 1.0, 3 * s[0]
        A[-1, -2], A[-1, -1], b[-1] = 1.0, 2.0, 3 * s[-1]
    else:
        raise ValueError("bc must be 'clamped' or 'natural'")
    v = np.linalg.solve(A, b)
    return v[:, 0] if scalar else v


def _hermite_segments(q, t, derivs):
    """Piecewise polynomial through ``q`` with given derivative arrays ``derivs`` (list of (N, n) arrays, order 1, 2, ...)."""
    segs = []
    for i in range(len(q) - 1):
        h = t[i + 1] - t[i]
        s0 = np.stack([q[i]] + [d[i] for d in derivs])
        s1 = np.stack([q[i + 1]] + [d[i + 1] for d in derivs])
        segs.append(hermite_coeffs(h, s0, s1).T)              # (n, k)
    return np.stack(segs)


def via_cubic(points, times, velocities="heuristic", v0=0.0, vf=0.0):
    """Cubic through via points with prescribed or computed knot velocities.

    ``velocities`` may be ``"heuristic"`` (Craig), ``"spline"`` (C^2, clamped by ``v0, vf``), ``"natural"``
    (C^2 with zero end accelerations), ``"zero"`` (stop at every point) or an array ``(N,)`` / ``(N, n)``.
    """
    q, scalar = _pts(points)
    t = np.asarray(times, dtype=float)
    if isinstance(velocities, str):
        v = {"heuristic": lambda: heuristic_velocities(q, t, v0, vf),
             "spline": lambda: spline_velocities(q, t, "clamped", v0, vf),
             "natural": lambda: spline_velocities(q, t, "natural"),
             "zero": lambda: np.zeros_like(q)}[velocities]()
    else:
        v = np.asarray(velocities, dtype=float)
    v = v.reshape(q.shape)
    return PiecewisePoly(t, _hermite_segments(q, t, [v]), scalar, info=dict(velocities=v))


def via_quintic(points, times, velocities, accelerations=None):
    """Quintic through via points with prescribed knot velocities and accelerations (zero accelerations by default)."""
    q, scalar = _pts(points)
    t = np.asarray(times, dtype=float)
    v = np.asarray(velocities, dtype=float).reshape(q.shape)
    a = np.zeros_like(q) if accelerations is None else np.asarray(accelerations, dtype=float).reshape(q.shape)
    return PiecewisePoly(t, _hermite_segments(q, t, [v, a]), scalar, info=dict(velocities=v, accelerations=a))


def via_stop(points, times):
    """Rest-to-rest cubic between consecutive points (the joint stops at every via point)."""
    return via_cubic(points, times, velocities="zero")


# ---------------------------------------------------------------------------------------------
# Linear segments with parabolic blends (LSPB), bang-bang, S-curves
# ---------------------------------------------------------------------------------------------
def lspb_limits(D, T):
    """Admissible parameter ranges of a rest-to-rest trapezoidal profile of distance ``D > 0`` in time ``T``.

    Returns ``dict(a_min, v_min, v_max, tb_max)``: acceleration must satisfy ``a >= 4 D / T**2``, the cruise speed
    ``D/T < v <= 2 D/T`` and the blend time ``0 < tb <= T/2``.
    """
    D = abs(D)
    return dict(a_min=4 * D / T ** 2, v_min=D / T, v_max=2 * D / T, tb_max=T / 2)


def lspb(q0, qf, T, *, v=None, a=None, tb=None):
    """Rest-to-rest trapezoidal-velocity profile: parabolic blend, linear segment, parabolic blend.

    Exactly one of ``v`` (cruise speed), ``a`` (blend acceleration) or ``tb`` (blend time) is given.  For several
    joints the blend time is common to all, so every joint starts, cruises and stops together; ``v`` and ``a`` then
    refer to the joint with the largest ``|q_f - q_0|``.  Infeasible parameters raise ``ValueError`` with the
    admissible interval in the message.
    """
    if sum(x is not None for x in (v, a, tb)) != 1:
        raise ValueError("give exactly one of v, a, tb")
    scalar = np.ndim(q0) == 0 and np.ndim(qf) == 0
    q0, qf = np.broadcast_arrays(np.atleast_1d(np.asarray(q0, float)), np.atleast_1d(np.asarray(qf, float)))
    dq = qf - q0
    D = float(np.max(np.abs(dq)))
    if D == 0.0:
        return PiecewisePoly([0.0, T], np.stack([q0, 0 * q0], axis=-1)[None], scalar,
                             info=dict(tb=0.0, v=0 * q0, a=0 * q0, T=T, kind="static"))
    lim = lspb_limits(D, T)
    tol = 1e-12
    if a is not None:
        a = abs(float(a))
        if a < lim["a_min"] * (1 - tol):
            raise ValueError(f"acceleration {a:g} too small: need a >= 4D/T^2 = {lim['a_min']:g}")
        tb_ = T / 2 - np.sqrt(max(a * a * T * T - 4 * a * D, 0.0)) / (2 * a)
    elif v is not None:
        v = abs(float(v))
        if not (lim["v_min"] * (1 + tol) < v <= lim["v_max"] * (1 + tol)):
            raise ValueError(f"cruise speed {v:g} outside ({lim['v_min']:g}, {lim['v_max']:g}] = (D/T, 2D/T]")
        tb_ = T - D / v
    else:
        tb_ = float(tb)
        if not (0 < tb_ <= T / 2 * (1 + tol)):
            raise ValueError(f"blend time {tb_:g} outside (0, T/2] = (0, {T / 2:g}]")
    tb_ = min(tb_, T / 2)
    vel = dq / (T - tb_)                                      # signed cruise velocity of every joint
    acc = vel / tb_
    info = dict(tb=tb_, v=vel, a=acc, T=T, kind="triangle" if abs(T - 2 * tb_) < 1e-12 else "trapezoid")
    c1 = np.stack([q0, 0 * q0, acc / 2], axis=-1)             # blend in
    qb = q0 + 0.5 * vel * tb_
    c2 = np.stack([qb, vel, 0 * q0], axis=-1)                 # cruise
    # blend-out in local time tau = t - (T - tb):  q = qf - a/2 (T - t)^2 = qf - a/2 (tb - tau)^2
    q_in = qf - 0.5 * acc * tb_ ** 2
    c3 = np.stack([q_in, vel, -acc / 2], axis=-1)
    if info["kind"] == "triangle":
        return PiecewisePoly([0.0, tb_, T], np.stack([c1, c3]), scalar, info)
    return PiecewisePoly([0.0, tb_, T - tb_, T], np.stack([c1, c2, c3]), scalar, info)


def min_time_trapezoid(D, vmax, amax):
    """Shortest rest-to-rest time for distance ``D`` under ``|v| <= vmax`` and ``|a| <= amax``.

    Returns ``dict(T, v, tb, kind)``; ``kind`` is ``"trapezoid"`` when the speed limit is reached and ``"triangle"``
    (bang-bang) when it is not.
    """
    D = abs(float(D))
    if vmax <= 0 or amax <= 0:
        raise ValueError("limits must be positive")
    if D >= vmax ** 2 / amax:
        return dict(T=D / vmax + vmax / amax, v=vmax, tb=vmax / amax, kind="trapezoid")
    vp = np.sqrt(D * amax)
    return dict(T=2 * vp / amax, v=vp, tb=vp / amax, kind="triangle")


def bang_bang(q0, qf, amax):
    """Time-optimal rest-to-rest motion of a double integrator with ``|a| <= amax``: full acceleration, then full braking."""
    D = float(np.max(np.abs(np.asarray(qf, float) - np.asarray(q0, float))))
    T = 2 * np.sqrt(D / amax)
    return lspb(q0, qf, T, tb=T / 2)


def lspb_via(points, durations, accel):
    """Linear segments with parabolic blends through via points (Craig's blends).

    ``points`` are ``q_0 ... q_m``; ``durations[k]`` is the time allotted to the segment ``k -> k+1`` and
    ``accel[j]`` the blend-acceleration magnitude at point ``j`` (``m + 1`` values; only the first and last are used
    as the start/stop accelerations when ``m = 1``).  The trajectory starts and ends at rest, blends are centred in
    time on the interior via points and the straight lines pass *through* them when extended, so the trajectory
    cuts the corner: it does **not** reach the interior via points.  Several joints are treated independently.
    """
    scalar = np.ndim(points) == 1
    q = np.asarray(points, dtype=float)
    q = q[:, None] if scalar else q
    td = np.asarray(durations, dtype=float)
    acc = np.asarray(accel, dtype=float)
    if acc.ndim == 1:
        acc = np.repeat(acc[:, None], q.shape[1], axis=1)
    trajs = [_lspb_via_scalar(q[:, j], td, acc[:, j]) for j in range(q.shape[1])]
    return trajs[0] if scalar else Stack(trajs)


def _lspb_via_scalar(q, td, acc):
    m = len(q) - 1
    if m == 1:
        return lspb(q[0], q[1], td[0], a=acc[0])
    sg = lambda x: 1.0 if x >= 0 else -1.0
    # blend times and linear velocities
    tbl = np.zeros(m + 1)
    vel = np.zeros(m)
    d0 = q[1] - q[0]
    a0 = sg(d0) * abs(acc[0])
    disc = td[0] ** 2 - 2 * d0 / a0
    if disc < 0:
        raise ValueError(f"first blend infeasible: need |a| >= 2|dq|/t^2 = {2 * abs(d0) / td[0] ** 2:g}")
    tbl[0] = td[0] - np.sqrt(disc)
    vel[0] = d0 / (td[0] - 0.5 * tbl[0])
    for k in range(1, m - 1):
        vel[k] = (q[k + 1] - q[k]) / td[k]
    dm = q[m] - q[m - 1]
    am = -sg(dm) * abs(acc[m])
    disc = td[m - 1] ** 2 + 2 * dm / am
    if disc < 0:
        raise ValueError(f"last blend infeasible: need |a| >= 2|dq|/t^2 = {2 * abs(dm) / td[m - 1] ** 2:g}")
    tbl[m] = td[m - 1] - np.sqrt(disc)
    vel[m - 1] = dm / (td[m - 1] - 0.5 * tbl[m])
    blend_acc = np.zeros(m + 1)
    blend_acc[0], blend_acc[m] = a0, am
    for j in range(1, m):
        dv = vel[j] - vel[j - 1]
        blend_acc[j] = sg(dv) * abs(acc[j])
        tbl[j] = abs(dv) / abs(acc[j]) if dv != 0 else 0.0
    # assemble pieces: (duration, acceleration) with forward integration for the start states
    T_via = np.concatenate([[0.0], np.cumsum(td)])
    pieces = []                                               # (t_start, t_end, accel)
    t = 0.0
    for j in range(m + 1):
        if j == 0:
            b0, b1 = 0.0, tbl[0]
        elif j == m:
            b0, b1 = T_via[m] - tbl[m], T_via[m]
        else:
            b0, b1 = T_via[j] - 0.5 * tbl[j], T_via[j] + 0.5 * tbl[j]
        if b0 < t - 1e-12:
            raise ValueError("blends overlap: increase the durations or the accelerations")
        if b0 > t + 1e-12:
            pieces.append((t, b0, 0.0, "line"))
        if b1 > b0 + 1e-15:
            pieces.append((b0, b1, blend_acc[j], "blend"))
        t = max(t, b1)
    times, coeffs = [0.0], []
    qc, vc = q[0], 0.0
    for (ta, tb_, a, kind) in pieces:
        h = tb_ - ta
        coeffs.append([[qc, vc, a / 2]])
        qc, vc = qc + vc * h + 0.5 * a * h * h, vc + a * h
        times.append(tb_)
    coeffs = np.array(coeffs).reshape(-1, 1, 3)
    traj = PiecewisePoly(times, coeffs, True, info=dict(blend_times=tbl, velocities=vel, blend_acc=blend_acc,
                                                       final=qc, via_times=T_via))
    return traj


def scurve(q0, qf, vmax, amax, jmax):
    """Jerk-limited rest-to-rest profile (S-curve) of a scalar move, as a piecewise cubic with up to seven pieces.

    The motion is symmetric: jerk ``+j, 0, -j`` (acceleration), cruise, ``-j, 0, +j`` (deceleration).  The limits
    ``vmax``, ``amax``, ``jmax`` are respected; phases that are not needed (constant acceleration, cruise) get
    zero length.  ``info`` records the phase durations and the peak speed.
    """
    D = float(qf - q0)
    sgn = 1.0 if D >= 0 else -1.0
    D = abs(D)
    if D == 0:
        return PiecewisePoly([0.0, 1.0], np.array([[[q0, 0, 0, 0]]]), True, info=dict(T=0.0))
    # speed reached at the end of the acceleration half
    if vmax >= amax ** 2 / jmax:                              # amax can be reached before vmax
        Tj, Ta_c = amax / jmax, vmax / amax - amax / jmax
    else:
        Tj, Ta_c = np.sqrt(vmax / jmax), 0.0
    Tacc = 2 * Tj + Ta_c
    if D >= vmax * Tacc:                                      # cruise phase exists
        vp, Tv = vmax, (D - vmax * Tacc) / vmax
    else:                                                     # speed limit not reached
        a_reach = amax ** 2 / jmax
        vp_a = 0.5 * amax * (-amax / jmax + np.sqrt((amax / jmax) ** 2 + 4 * D / amax))
        if vp_a >= a_reach:
            vp = vp_a
            Tj, Ta_c = amax / jmax, vp / amax - amax / jmax
        else:
            vp = (D * np.sqrt(jmax) / 2) ** (2 / 3)
            Tj, Ta_c = np.sqrt(vp / jmax), 0.0
        Tv = 0.0
    durs = [Tj, Ta_c, Tj, Tv, Tj, Ta_c, Tj]
    jerks = [jmax, 0, -jmax, 0, -jmax, 0, jmax]
    jerks = [sgn * j for j in jerks]
    times, coeffs = [0.0], []
    q, v, a = float(q0), 0.0, 0.0
    for d, j in zip(durs, jerks):
        if d <= 1e-14:
            continue
        coeffs.append([[q, v, a / 2, j / 6]])
        q, v, a = q + v * d + a * d * d / 2 + j * d ** 3 / 6, v + a * d + j * d * d / 2, a + j * d
        times.append(times[-1] + d)
    info = dict(T=times[-1], vpeak=vp, durations=durs, final=q)
    return PiecewisePoly(times, np.array(coeffs).reshape(-1, 1, 4), True, info)


def coordinated_ptp(q0, qf, vmax, amax, profile="trapezoid", jmax=None):
    """Synchronised point-to-point move: every joint follows the same normalised profile ``s(t)``.

    The limits are converted to limits on ``s``: ``|s'| <= min_i vmax_i / |dq_i|`` and
    ``|s''| <= min_i amax_i / |dq_i|``.  ``profile`` is ``"trapezoid"``, ``"cubic"``, ``"quintic"`` or ``"scurve"``
    (which also needs ``jmax``).  Returns a :class:`JointLine` whose ``s.info`` holds the timing.
    """
    q0, qf = np.atleast_1d(np.asarray(q0, float)), np.atleast_1d(np.asarray(qf, float))
    dq = np.abs(qf - q0)
    vmax = np.broadcast_to(np.asarray(vmax, float), dq.shape)
    amax = np.broadcast_to(np.asarray(amax, float), dq.shape)
    mv = dq > 0
    if not mv.any():
        return JointLine(q0, qf, linear(0.0, 1.0, 1.0))
    sv, sa = np.min(vmax[mv] / dq[mv]), np.min(amax[mv] / dq[mv])
    if profile == "trapezoid":
        tm = min_time_trapezoid(1.0, sv, sa)
        s = lspb(0.0, 1.0, tm["T"], tb=tm["tb"]) if tm["tb"] < tm["T"] / 2 - 1e-12 else bang_bang(0.0, 1.0, sa)
    elif profile in ("cubic", "quintic"):
        cv, ca = (1.5, 6.0) if profile == "cubic" else (15 / 8, 10 / np.sqrt(3))
        T = max(cv / sv, np.sqrt(ca / sa))
        s = cubic(0.0, 1.0, T) if profile == "cubic" else quintic(0.0, 1.0, T)
    elif profile == "scurve":
        if jmax is None:
            raise ValueError("scurve needs jmax (limit on the jerk of s)")
        s = scurve(0.0, 1.0, sv, sa, jmax)
    else:
        raise ValueError("profile must be trapezoid, cubic, quintic or scurve")
    return JointLine(q0, qf, s)


# ---------------------------------------------------------------------------------------------
# Cartesian paths: positions, orientations, IK stepping
# ---------------------------------------------------------------------------------------------
def rotation_geodesic(R0, R1, s):
    """Orientation along the shortest rotation from ``R0`` to ``R1``: ``R(s) = R0 Rot(k, s*theta)``.

    ``(k, theta)`` is the angle-axis form of the relative rotation ``R0^T R1`` (``k`` is expressed in the frame of ``R0``).
    ``s`` may be an array; the result has shape ``s.shape + (3, 3)``.  This is SLERP written with rotation matrices.
    """
    k, th = _tf.axis_angle_from_rot(np.asarray(R0).T @ np.asarray(R1))
    s = np.asarray(s, dtype=float)
    Rs = np.array([_tf.rot_axis_angle(k, si * th) for si in s.ravel()]).reshape(s.shape + (3, 3))
    return np.asarray(R0) @ Rs


def euler_interp(R0, R1, s, convention="rpy"):
    """Orientation obtained by interpolating the three X-Y-Z fixed (roll-pitch-yaw) angles linearly in ``s``.

    The angle differences are wrapped into ``(-pi, pi]``.  Shown for comparison with :func:`rotation_geodesic`.
    """
    if convention != "rpy":
        raise ValueError("only the X-Y-Z fixed-angle (rpy) convention is implemented")
    a0, a1 = np.array(_tf.rot_to_rpy(R0)), np.array(_tf.rot_to_rpy(R1))
    d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    s = np.asarray(s, dtype=float)
    out = [_tf.rpy_to_rot(*(a0 + si * d)) for si in s.ravel()]
    return np.array(out).reshape(s.shape + (3, 3))


def matrix_lerp(R0, R1, s):
    """Element-wise linear interpolation of two rotation matrices (**not** a rotation in between; for demonstration)."""
    s = np.asarray(s, dtype=float)[..., None, None]
    return (1 - s) * np.asarray(R0) + s * np.asarray(R1)


def pose_interp(T0, T1, s):
    """Straight-line position and geodesic orientation between two poses; ``s`` scalar or array in ``[0, 1]``."""
    T0, T1 = np.asarray(T0, float), np.asarray(T1, float)
    s = np.asarray(s, dtype=float)
    out = np.tile(np.eye(4), s.shape + (1, 1))
    out[..., :3, :3] = rotation_geodesic(T0[:3, :3], T1[:3, :3], s)
    out[..., :3, 3] = T0[:3, 3] + s[..., None] * (T1[:3, 3] - T0[:3, 3])
    return out


class CartesianLine:
    """Straight-line tool motion with SLERP orientation and a scalar timing law ``s(t)``, ``s: 0 -> 1``.

    ``pose(t)`` is a 4x4 transform (or an array of them); ``twist(t)`` returns the linear velocity of the tool
    origin and the angular velocity of the tool, both written in the base frame.
    """

    def __init__(self, T0, T1, s):
        self.T0, self.T1, self.s = np.asarray(T0, float), np.asarray(T1, float), s
        self.dp = self.T1[:3, 3] - self.T0[:3, 3]
        k, th = _tf.axis_angle_from_rot(self.T0[:3, :3].T @ self.T1[:3, :3])
        self.theta = th
        self.axis_world = self.T0[:3, :3] @ k                 # constant rotation axis in the base frame
        self.t0, self.t1 = s.t0, s.t1

    @property
    def duration(self):
        return self.t1 - self.t0

    def pose(self, t):
        return pose_interp(self.T0, self.T1, self.s.pos(t))

    def twist(self, t):
        sd = np.asarray(self.s.vel(t))[..., None]
        return sd * self.dp, sd * self.theta * self.axis_world

    def accel(self, t):
        sdd = np.asarray(self.s.acc(t))[..., None]
        return sdd * self.dp, sdd * self.theta * self.axis_world

    def speeds(self, t):
        v, w = self.twist(t)
        return np.linalg.norm(v, axis=-1), np.linalg.norm(w, axis=-1)


def pose_error(T_cur, T_des):
    """Six-vector ``[p_des - p_cur; rotation vector of R_des R_cur^T]`` (task error in the base frame)."""
    T_cur, T_des = np.asarray(T_cur, float), np.asarray(T_des, float)
    k, th = _tf.axis_angle_from_rot(T_des[:3, :3] @ T_cur[:3, :3].T)
    return np.concatenate([T_des[:3, 3] - T_cur[:3, 3], k * th])


def dls_step(J, e, lam=0.05):
    """Damped-least-squares joint step ``dq = J^T (J J^T + lam^2 I)^{-1} e`` (bounded near singularities)."""
    J = np.asarray(J, float)
    return J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(J.shape[0]), e)


def track_poses(q_start, poses, fk, jac, lam=0.02, tol=1e-9, max_iter=60, step_max=0.3):
    """Joint solutions for a list of 4x4 poses by damped-least-squares iteration, each seeded with the previous solution.

    ``fk(q) -> 4x4`` and ``jac(q) -> 6 x n`` (linear rows first, both in the base frame).  Returns ``Q`` ``(m, n)`` and
    the final task-error norms ``(m,)``.
    """
    q = np.array(q_start, dtype=float)
    Q, res = [], []
    for Td in poses:
        for _ in range(max_iter):
            e = pose_error(fk(q), Td)
            if np.linalg.norm(e) < tol:
                break
            dq = dls_step(jac(q), e, lam)
            nrm = np.linalg.norm(dq)
            q = q + (dq if nrm <= step_max else dq * step_max / nrm)
        Q.append(q.copy())
        res.append(np.linalg.norm(pose_error(fk(q), Td)))
    return np.array(Q), np.array(res)


# ---------------------------------------------------------------------------------------------
# Time scaling and execution
# ---------------------------------------------------------------------------------------------
class Path:
    """A geometric path ``q(s)`` with its first and second derivatives with respect to the path parameter ``s``.

    ``q``, ``dq``, ``ddq`` are callables returning arrays of shape ``(n,)``.  If ``dq`` / ``ddq`` are omitted they
    are obtained from central differences with step ``h``.
    """

    def __init__(self, q, dq=None, ddq=None, h=1e-4):
        self.q = q
        self._dq, self._ddq, self.h = dq, ddq, h

    def d1(self, s):
        if self._dq is not None:
            return np.asarray(self._dq(s), float)
        h = self.h
        return (np.asarray(self.q(s + h)) - np.asarray(self.q(s - h))) / (2 * h)

    def d2(self, s):
        if self._ddq is not None:
            return np.asarray(self._ddq(s), float)
        h = self.h
        return (np.asarray(self.q(s + h)) - 2 * np.asarray(self.q(s)) + np.asarray(self.q(s - h))) / h ** 2


class ScaledPath(Trajectory):
    """Trajectory ``q(s(t))`` where ``s(t)`` is piecewise parabolic (constant ``s''`` on each grid interval).

    Built by :func:`time_optimal_scaling`; attributes ``s``, ``sdot``, ``u`` (= ``sdot**2``) and ``tk`` describe
    the solution at the grid points.  Derivatives up to the acceleration are available.
    """

    def __init__(self, path, s_grid, u):
        self.path, self.sg, self.u = path, np.asarray(s_grid, float), np.asarray(u, float)
        self.sdot = np.sqrt(np.maximum(self.u, 0.0))
        ds = np.diff(self.sg)
        self.sdd = np.diff(self.u) / (2 * ds)
        denom = self.sdot[:-1] + self.sdot[1:]
        if np.any(denom <= 0):
            raise ValueError("scaling is not feasible: zero speed along the path")
        self.dt = 2 * ds / denom
        self.tk = np.concatenate([[0.0], np.cumsum(self.dt)])
        self.t0, self.t1 = 0.0, float(self.tk[-1])
        q0 = np.atleast_1d(path.q(self.sg[0]))
        self.n, self.scalar = q0.size, np.ndim(path.q(self.sg[0])) == 0

    def s_of_t(self, t):
        """Path parameter, its rate and its second derivative at times ``t`` (arrays)."""
        k = np.clip(np.searchsorted(self.tk, t, side="right") - 1, 0, len(self.dt) - 1)
        tau = t - self.tk[k]
        s = self.sg[k] + self.sdot[k] * tau + 0.5 * self.sdd[k] * tau ** 2
        return s, self.sdot[k] + self.sdd[k] * tau, self.sdd[k]

    def _eval(self, t, d):
        s, sd, sdd = self.s_of_t(t)
        if d == 0:
            return np.array([np.atleast_1d(self.path.q(si)) for si in s])
        g = np.array([np.atleast_1d(self.path.d1(si)) for si in s])
        if d == 1:
            return g * sd[:, None]
        if d == 2:
            h = np.array([np.atleast_1d(self.path.d2(si)) for si in s])
            return h * (sd ** 2)[:, None] + g * sdd[:, None]
        raise NotImplementedError("jerk of a time-optimal scaling is not defined (acceleration jumps)")


def _sdd_interval(g, h, u, amax):
    """Interval ``[lo, hi]`` of ``s''`` such that ``|g_i s'' + h_i u| <= amax_i`` for all joints (``u = sdot**2``)."""
    lo, hi = -np.inf, np.inf
    flat = np.abs(g) < 1e-12
    if np.any(np.abs(h[flat] * u) > amax[flat] * (1 + 1e-12)):
        return np.inf, -np.inf
    gg, hh, aa = g[~flat], h[~flat], amax[~flat]
    if gg.size:
        b1, b2 = (-aa - hh * u) / gg, (aa - hh * u) / gg
        lo, hi = np.max(np.minimum(b1, b2)), np.min(np.maximum(b1, b2))
    return lo, hi


def time_optimal_scaling(path, vmax, amax, N=400, margin=0.0):
    """Minimum-time traversal of ``path`` (``s`` from 0 to 1, rest at both ends) under joint velocity and acceleration bounds.

    The problem is posed in the phase plane ``(s, u = sdot**2)``, where ``du/ds = 2 s''`` and the bounds are linear
    in ``(u, s'')``.  The *limit curve* ``u_lim(s)`` (largest ``u`` for which some admissible ``s''`` exists) is found
    by bisection; a forward pass then accelerates as hard as allowed, and a backward pass from the end brakes as hard as
    allowed; the pointwise minimum is the solution.  ``margin`` shrinks the limits by that fraction (a safety factor
    against the discretisation error).  Returns a :class:`ScaledPath`.
    """
    vmax = np.asarray(vmax, float) * (1 - margin)
    amax = np.asarray(amax, float) * (1 - margin)
    sg = np.linspace(0.0, 1.0, N + 1)
    ds = 1.0 / N
    G = np.array([np.atleast_1d(path.d1(s)) for s in sg])
    H = np.array([np.atleast_1d(path.d2(s)) for s in sg])
    n = G.shape[1]
    if np.min(np.linalg.norm(G, axis=1)) < 1e-9:
        k = int(np.argmin(np.linalg.norm(G, axis=1)))
        raise ValueError(f"q'(s) vanishes at s = {sg[k]:.3g}: reparametrise the path (for example by arc length)")
    vmax = np.broadcast_to(vmax, (n,))
    amax = np.broadcast_to(amax, (n,))
    ulim = np.empty(N + 1)
    for k in range(N + 1):
        nz = np.abs(G[k]) > 1e-12
        top = np.min((vmax[nz] / np.abs(G[k][nz])) ** 2) if nz.any() else 1e12
        lo_u, hi_u = 0.0, top
        lo, hi = _sdd_interval(G[k], H[k], hi_u, amax)
        if lo <= hi:
            ulim[k] = hi_u
            continue
        for _ in range(60):                                   # bisection on the feasibility boundary
            mid = 0.5 * (lo_u + hi_u)
            lo, hi = _sdd_interval(G[k], H[k], mid, amax)
            if lo <= hi:
                lo_u = mid
            else:
                hi_u = mid
        ulim[k] = lo_u
    u = np.zeros(N + 1)
    for k in range(N):                                        # forward: accelerate as hard as possible
        lo, hi = _sdd_interval(G[k], H[k], u[k], amax)
        # the same constant s'' must also be admissible at the next node, where u = u_k + 2 ds s''
        lo2, hi2 = _sdd_interval(G[k + 1] + 2 * ds * H[k + 1], H[k + 1], u[k], amax)
        if max(lo, lo2) <= min(hi, hi2):
            hi = min(hi, hi2)
        u[k + 1] = min(max(u[k] + 2 * ds * hi, 0.0), ulim[k + 1]) if np.isfinite(hi) else ulim[k + 1]
    u[-1] = 0.0
    for k in range(N, 0, -1):                                 # backward: brake as hard as possible
        lo, hi = _sdd_interval(G[k], H[k], u[k], amax)
        lo2, hi2 = _sdd_interval(G[k - 1] - 2 * ds * H[k - 1], H[k - 1], u[k], amax)   # u_{k-1} = u_k - 2 ds s''
        if max(lo, lo2) <= min(hi, hi2):
            lo = max(lo, lo2)
        up = u[k] - 2 * ds * lo if np.isfinite(lo) else np.inf
        u[k - 1] = min(u[k - 1], max(up, 0.0))
    u = np.minimum(u, ulim)
    sp = ScaledPath(path, sg, u)
    sp.ulim = ulim
    return sp


def peak_values(traj, num=4001):
    """Largest ``|q|``, ``|qd|``, ``|qdd|``, ``|qddd|`` of each coordinate, from a dense evaluation of ``traj``."""
    t = np.linspace(traj.t0, traj.t1, num)
    out = {}
    for name, f in (("pos", traj.pos), ("vel", traj.vel), ("acc", traj.acc)):
        out[name] = np.max(np.abs(np.atleast_2d(f(t).T).T), axis=0)
    try:
        out["jerk"] = np.max(np.abs(np.atleast_2d(traj.jerk(t).T).T), axis=0)
    except NotImplementedError:
        out["jerk"] = np.full(out["pos"].shape, np.nan)
    return out


def required_time_stretch(traj, vmax=None, amax=None):
    """Smallest factor ``lam`` such that ``TimeScaled(traj, lam)`` obeys the velocity and acceleration limits.

    Velocity scales as ``1/lam`` and acceleration as ``1/lam**2``, hence ``lam = max(v_peak/vmax, sqrt(a_peak/amax))``.
    A value below one means the trajectory could be executed faster.
    """
    pk = peak_values(traj)
    lam_v = 0.0 if vmax is None else float(np.max(pk["vel"] / np.asarray(vmax, float)))
    lam_a = 0.0 if amax is None else float(np.sqrt(np.max(pk["acc"] / np.asarray(amax, float))))
    return max(lam_v, lam_a)


def check_limits(traj, q_lim=None, v_lim=None, a_lim=None, num=4001):
    """Check a trajectory against joint limits. Returns a dict with ``ok`` and per-quantity worst-case ratios.

    ``q_lim`` is a pair ``(lower, upper)``; ``v_lim`` and ``a_lim`` are symmetric magnitudes.  A ratio above one means
    the limit is violated.
    """
    t = np.linspace(traj.t0, traj.t1, num)
    q = np.atleast_2d(traj.pos(t).T).T
    rep = {}
    if q_lim is not None:
        lo, hi = np.asarray(q_lim[0], float), np.asarray(q_lim[1], float)
        half, mid = (hi - lo) / 2, (hi + lo) / 2
        rep["pos"] = np.max(np.abs(q - mid) / half, axis=0)
    if v_lim is not None:
        rep["vel"] = np.max(np.abs(np.atleast_2d(traj.vel(t).T).T), axis=0) / np.asarray(v_lim, float)
    if a_lim is not None:
        rep["acc"] = np.max(np.abs(np.atleast_2d(traj.acc(t).T).T), axis=0) / np.asarray(a_lim, float)
    rep["ok"] = all(np.all(v <= 1 + 1e-9) for k, v in rep.items())
    return rep


def servo_times(T, rate=None, dt=None):
    """Sample instants ``0, dt, 2 dt, ..., T`` of a servo loop (the last sample is exactly ``T``)."""
    dt = 1.0 / rate if dt is None else dt
    n = int(np.floor(T / dt + 1e-9))
    t = np.arange(n + 1) * dt
    if T - t[-1] > 1e-9:
        t = np.append(t, T)
    return t


class OnlinePTP:
    """Run-time point-to-point generator: one tick at a time, for a target that may change while moving.

    Each joint is driven by a square-root braking law: the reference speed is
    ``min(vmax, sqrt(2 kappa amax |e|))`` toward the target (``e`` = remaining distance, ``kappa < 1`` leaves a margin
    so that braking is always possible); the speed follows the reference with the acceleration clipped to ``amax``.
    Joints are independent (they do not arrive simultaneously).
    """

    def __init__(self, q0, vmax, amax, dt, kappa=0.8):
        self.q = np.atleast_1d(np.array(q0, dtype=float))
        self.v = np.zeros_like(self.q)
        self.vmax, self.amax = np.broadcast_to(np.asarray(vmax, float), self.q.shape), np.broadcast_to(np.asarray(amax, float), self.q.shape)
        self.dt, self.kappa = float(dt), float(kappa)

    def step(self, target):
        """Advance one servo period toward ``target``; returns the new ``(q, v)``."""
        e = np.atleast_1d(np.asarray(target, float)) - self.q
        v_ref = np.sign(e) * np.minimum(self.vmax, np.sqrt(2 * self.kappa * self.amax * np.abs(e)))
        dv = np.clip(v_ref - self.v, -self.amax * self.dt, self.amax * self.dt)
        v_new = self.v + dv
        q_new = self.q + 0.5 * (self.v + v_new) * self.dt
        done = (np.abs(e) < 0.5 * self.amax * self.dt ** 2) & (np.abs(self.v) < self.amax * self.dt)
        self.q = np.where(done, np.asarray(target, float), q_new)
        self.v = np.where(done, 0.0, v_new)
        return self.q.copy(), self.v.copy()

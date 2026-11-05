"""Classical and state-space linear control, in plain NumPy/SciPy.

This module collects the small, tested helpers used by Chapter 1 of *Robot Motion Control* (control-systems
foundations).  The widely used ``control`` package is deliberately *not* required: everything a first course
needs - transfer-function algebra, second-order step-response metrics, the Routh array, a root-locus sweep,
Bode/Nyquist data and stability margins, Ziegler-Nichols tuning, a discrete PID with anti-windup,
controllability/observability, Ackermann pole placement, observers and LQR - fits in a few hundred readable lines.

Conventions
-----------
* Polynomials are coefficient arrays in **descending** powers of ``s`` (``[1, 3, 2]`` is ``s**2 + 3 s + 2``).
* Frequencies ``w`` are in rad/s; phases are returned in degrees; magnitudes either linear or in dB as stated.
* A transfer function is a :class:`TF`; state-space models are tuples ``(A, B, C, D)`` of 2-D arrays.
* Time is in seconds, angles in radians, torques in N m.

The module is pure Python/NumPy/SciPy (no MuJoCo), so it can be tested and used everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import linalg, optimize, signal

__all__ = [
    # transfer functions
    "TF", "tf_from_zpk", "first_order", "second_order", "integrator", "pade_delay", "series", "parallel", "feedback",
    # time response
    "second_order_metrics", "zeta_from_overshoot", "second_order_from_specs", "step_metrics",
    # stability
    "routh_array", "routh_stability", "is_hurwitz", "steady_state_error_constants",
    # root locus
    "root_locus", "rl_asymptotes", "rl_breakaway_points", "rl_imaginary_crossings", "rl_departure_angle",
    "rl_gain_at", "rl_real_axis_segments",
    # frequency response
    "bode", "margins", "nyquist_curve", "nyquist_encirclements", "bandwidth", "sensitivity", "peak_sensitivity",
    # PID and tuning
    "PID", "zn_step", "zn_ultimate", "fit_foptd", "ultimate_gain_period",
    # discretisation
    "c2d_ss", "c2d_tf", "dlsim_tf",
    # state space
    "ctrb", "obsv", "rank_controllable", "rank_observable", "acker", "place", "observer_gain", "lqr", "lqr_discrete",
    "controllable_canonical", "ss_to_tf", "prefilter_gain", "numerical_jacobian", "linearize",
    "dc_motor_tf", "dc_motor_ss",
]


# ======================================================================================================
# Transfer functions
# ======================================================================================================
def _trim(p, tol: float = 0.0) -> np.ndarray:
    """Remove leading zero coefficients of a polynomial; keep at least one coefficient."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    nz = np.flatnonzero(np.abs(p) > tol)
    return p[nz[0]:] if nz.size else np.array([0.0])


class TF:
    """Rational transfer function ``num(s)/den(s)`` with the usual algebra.

    >>> G = TF([10], [1, 3, 2])
    >>> (G * G).den
    array([ 1.,  6., 13., 12.,  4.])
    >>> G.feedback().den          # closed loop under unity negative feedback
    array([ 1.,  3., 12.])
    """

    def __init__(self, num, den=(1.0,)):
        self.num = _trim(num)
        self.den = _trim(den)
        if self.den.size == 1 and self.den[0] == 0.0:
            raise ZeroDivisionError("denominator is identically zero")

    # -- algebra ----------------------------------------------------------------------------------------
    @staticmethod
    def _coerce(x) -> "TF":
        return x if isinstance(x, TF) else TF([float(x)], [1.0])

    def __mul__(self, other):
        o = self._coerce(other)
        return TF(np.polymul(self.num, o.num), np.polymul(self.den, o.den))

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._coerce(other)
        return TF(np.polymul(self.num, o.den), np.polymul(self.den, o.num))

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __add__(self, other):
        o = self._coerce(other)
        return TF(np.polyadd(np.polymul(self.num, o.den), np.polymul(o.num, self.den)), np.polymul(self.den, o.den))

    __radd__ = __add__

    def __neg__(self):
        return TF(-self.num, self.den)

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __pow__(self, n: int):
        out = TF([1.0])
        for _ in range(int(n)):
            out = out * self
        return out

    def feedback(self, H=None, sign: int = -1) -> "TF":
        """Closed loop ``G/(1 - sign*G*H)``; ``sign=-1`` (default) is negative feedback, ``H`` defaults to 1."""
        H = TF([1.0]) if H is None else self._coerce(H)
        num = np.polymul(self.num, H.den)
        den = np.polyadd(np.polymul(self.den, H.den), -sign * np.polymul(self.num, H.num))
        return TF(num, den)

    # -- analysis ---------------------------------------------------------------------------------------
    @property
    def order(self) -> int:
        return self.den.size - 1

    @property
    def relative_degree(self) -> int:
        return (self.den.size - 1) - (self.num.size - 1)

    def poles(self) -> np.ndarray:
        return np.roots(self.den)

    def zeros(self) -> np.ndarray:
        return np.roots(self.num) if self.num.size > 1 else np.array([], dtype=complex)

    def dcgain(self) -> float:
        """``G(0)``; ``inf`` if there is an integrator."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(self.num[-1] / self.den[-1]) if self.den[-1] != 0 else float("inf")

    def __call__(self, s):
        s = np.asarray(s, dtype=complex)
        return np.polyval(self.num, s) / np.polyval(self.den, s)

    def is_stable(self, tol: float = 0.0) -> bool:
        """True if every pole has a strictly negative real part."""
        p = self.poles()
        return bool(p.size == 0 or np.all(p.real < -tol))

    def minreal(self, tol: float = 1e-8) -> "TF":
        """Cancel pole-zero pairs closer than ``tol`` (a crude minimal realisation)."""
        z, p = list(self.zeros()), list(self.poles())
        k = self.num[0] / self.den[0]
        keep_z = []
        for zi in z:
            hit = [j for j, pj in enumerate(p) if abs(pj - zi) < tol * max(1.0, abs(zi))]
            if hit:
                p.pop(hit[0])
            else:
                keep_z.append(zi)
        num = np.real(k * np.poly(keep_z)) if keep_z else np.array([k])
        den = np.real(np.poly(p)) if p else np.array([1.0])
        return TF(num, den)

    def ss(self):
        """Controllable-canonical state-space realisation ``(A, B, C, D)`` (SciPy's ``tf2ss``)."""
        return signal.tf2ss(self.num, self.den)

    # -- simulation -------------------------------------------------------------------------------------
    def step(self, t) -> np.ndarray:
        """Unit-step response sampled at the times ``t``."""
        return signal.step((self.num, self.den), T=np.asarray(t, float))[1]

    def impulse(self, t) -> np.ndarray:
        return signal.impulse((self.num, self.den), T=np.asarray(t, float))[1]

    def lsim(self, u, t) -> np.ndarray:
        """Response to the input samples ``u`` (linear interpolation between samples)."""
        return signal.lsim((self.num, self.den), U=np.asarray(u, float), T=np.asarray(t, float))[1]

    def __repr__(self) -> str:
        def poly(c):
            terms, n = [], c.size - 1
            for i, a in enumerate(c):
                k = n - i
                if a == 0:
                    continue
                coef = f"{a:.4g}"
                terms.append(coef if k == 0 else (f"{coef}s" if k == 1 else f"{coef}s^{k}"))
            return " + ".join(terms) if terms else "0"
        return f"TF( {poly(self.num)}  /  {poly(self.den)} )"


def tf_from_zpk(zeros: Sequence[complex], poles: Sequence[complex], gain: float = 1.0) -> TF:
    """Transfer function ``gain * prod(s - z) / prod(s - p)`` (complex roots must come in conjugate pairs)."""
    num = np.real(gain * np.poly(zeros)) if len(zeros) else np.array([float(gain)])
    den = np.real(np.poly(poles)) if len(poles) else np.array([1.0])
    return TF(num, den)


def first_order(K: float, tau: float) -> TF:
    """``K/(tau s + 1)``."""
    return TF([K], [tau, 1.0])


def second_order(wn: float, zeta: float, K: float = 1.0) -> TF:
    """Standard second-order system ``K wn^2 / (s^2 + 2 zeta wn s + wn^2)``."""
    return TF([K * wn**2], [1.0, 2 * zeta * wn, wn**2])


def integrator(k: float = 1.0) -> TF:
    return TF([k], [1.0, 0.0])


def pade_delay(T: float, order: int = 2) -> TF:
    """Pade approximation of a pure delay ``exp(-T s)`` (all-pass, with right-half-plane zeros)."""
    num, den = _pade(T, order)
    return TF(num, den)


def _pade(T: float, n: int):
    # coefficients of the (n, n) Pade approximant of exp(-T s)
    from math import factorial
    c = [factorial(2 * n - k) * factorial(n) / (factorial(2 * n) * factorial(k) * factorial(n - k)) for k in range(n + 1)]
    den = np.array([c[k] * T**k for k in range(n, -1, -1)])
    num = np.array([(-1) ** k * c[k] * T**k for k in range(n, -1, -1)])
    return num, den


def series(*Gs: TF) -> TF:
    out = TF([1.0])
    for G in Gs:
        out = out * G
    return out


def parallel(*Gs: TF) -> TF:
    out = TF([0.0])
    for G in Gs:
        out = out + G
    return out


def feedback(G: TF, H: TF | None = None, sign: int = -1) -> TF:
    return G.feedback(H, sign)


# ======================================================================================================
# Time response
# ======================================================================================================
def second_order_metrics(wn: float, zeta: float, band: float = 0.02) -> dict:
    """Closed-form step-response figures of ``wn^2/(s^2 + 2 zeta wn s + wn^2)`` for ``0 < zeta < 1``.

    Returns ``wd`` (damped frequency), ``sigma`` (decay rate zeta*wn), ``Mp`` (fractional overshoot),
    ``tp`` (peak time), ``tr`` (0-to-100 % rise time) and ``ts`` (settling time, the usual
    ``-ln(band * sqrt(1 - zeta^2)) / (zeta wn)`` estimate; ``4/(zeta wn)`` for 2 %).
    """
    if not 0 < zeta < 1:
        raise ValueError("closed-form underdamped metrics need 0 < zeta < 1")
    sigma = zeta * wn
    wd = wn * np.sqrt(1 - zeta**2)
    return dict(
        wd=wd, sigma=sigma,
        Mp=float(np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))),
        tp=float(np.pi / wd),
        tr=float((np.pi - np.arccos(zeta)) / wd),
        ts=float(-np.log(band * np.sqrt(1 - zeta**2)) / sigma),
    )


def zeta_from_overshoot(Mp: float) -> float:
    """Damping ratio that gives the fractional overshoot ``Mp`` (e.g. 0.1 for 10 %)."""
    L = np.log(Mp)
    return float(-L / np.sqrt(np.pi**2 + L**2))


def second_order_from_specs(Mp: float, ts: float, band: float = 0.02) -> tuple[float, float]:
    """``(wn, zeta)`` meeting an overshoot ``Mp`` and a settling time ``ts`` (using the standard estimates)."""
    zeta = zeta_from_overshoot(Mp)
    sigma = -np.log(band * np.sqrt(1 - zeta**2)) / ts
    return float(sigma / zeta), zeta


def step_metrics(t, y, final=None, band: float = 0.02, rise=(0.1, 0.9)) -> dict:
    """Measure a sampled step response.

    ``final`` defaults to the mean of the last 5 % of the record.  Returns a dict with ``final``, ``Mp``
    (fractional overshoot, 0 if none), ``tp`` (time of the peak), ``tr`` (``rise`` fractions of the final value,
    by linear interpolation), ``ts`` (last time the response is outside the +/- ``band`` tube; ``nan`` if it never
    settles within the record) and ``ess`` (``final`` minus the commanded 1, i.e. for a unit step).
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    yf = float(np.mean(y[-max(2, len(y) // 20):])) if final is None else float(final)
    if yf == 0:
        raise ValueError("final value is zero; metrics are defined relative to a non-zero final value")
    sgn = np.sign(yf)
    yn = y / yf
    ip = int(np.argmax(sgn * y))
    Mp = max(0.0, float(yn[ip] - 1.0))
    lo, hi = rise

    def crossing(level):
        idx = np.flatnonzero(yn >= level)
        if idx.size == 0:
            return np.nan
        i = idx[0]
        if i == 0:
            return float(t[0])
        return float(np.interp(level, [yn[i - 1], yn[i]], [t[i - 1], t[i]]))

    outside = np.flatnonzero(np.abs(yn - 1.0) > band)
    if outside.size == 0:
        ts = float(t[0])
    elif outside[-1] >= len(t) - 1:
        ts = float("nan")
    else:
        ts = float(t[outside[-1] + 1])
    return dict(final=yf, Mp=Mp, tp=float(t[ip]), tr=crossing(hi) - crossing(lo), ts=ts)


# ======================================================================================================
# Stability: Routh array
# ======================================================================================================
def routh_array(coeffs, eps: float | None = None):
    """Routh array of the polynomial with coefficients ``coeffs`` (descending powers).

    The elements may be floats or SymPy expressions.  Returns ``(table, info)`` where ``table`` is a list of rows
    (rows padded with zeros to equal length) and ``info`` is a dict with

    * ``first_column`` - the first column,
    * ``sign_changes`` - the number of sign changes in it (number of roots in the open right half-plane, for
      numeric input without special cases),
    * ``special`` - ``None``, ``"zero_pivot"`` (a zero in the first column; replaced by ``eps``) or
      ``"zero_row"`` (a whole row of zeros; replaced by the derivative of the auxiliary polynomial).

    For *symbolic* input no special-case handling is attempted: a symbolic pivot is simply carried along, so the
    first column can be inspected for the conditions that make every entry positive.
    """
    c = list(coeffs)
    symbolic = any(not isinstance(x, (int, float, np.floating, np.integer)) for x in c)
    n = len(c) - 1
    if n < 1:
        raise ValueError("need a polynomial of degree >= 1")
    ncol = n // 2 + 1
    rows = [c[0::2], c[1::2]]
    for r in rows:
        r.extend([0] * (ncol - len(r)))
    if not symbolic:
        scale = max(abs(float(x)) for x in c)
        eps = 1e-9 * scale if eps is None else eps
    info = dict(special=None, aux_poly=None)
    for k in range(2, n + 1):
        prev2, prev1 = rows[k - 2], rows[k - 1]
        if not symbolic and all(abs(float(x)) < 1e-14 * max(1.0, scale) for x in prev1):
            # whole row of zeros: auxiliary polynomial from the row above, differentiated
            order = n - (k - 2)                      # power of s of the row above
            aux = [prev2[j] for j in range(ncol)]
            powers = [order - 2 * j for j in range(ncol)]
            dpoly = [aux[j] * powers[j] if powers[j] > 0 else 0.0 for j in range(ncol)]
            info.update(special="zero_row", aux_poly=list(zip(powers, aux)))
            prev1 = dpoly
            rows[k - 1] = prev1
        if not symbolic and abs(float(prev1[0])) < 1e-14 * max(1.0, scale):
            info["special"] = info["special"] or "zero_pivot"
            prev1 = [eps] + list(prev1[1:])
            rows[k - 1] = prev1
        new = []
        for j in range(ncol - 1):
            a, b = prev2[j + 1], prev1[j + 1]
            val = (prev1[0] * a - prev2[0] * b) / prev1[0]
            if symbolic:
                import sympy as sp
                val = sp.simplify(val)
            new.append(val)
        new.append(0)
        rows.append(new)
    first = [r[0] for r in rows]
    if symbolic:
        info["first_column"], info["sign_changes"] = first, None
    else:
        fc = np.array([float(x) for x in first])
        info["first_column"] = fc
        info["sign_changes"] = int(np.sum(np.sign(fc[:-1]) * np.sign(fc[1:]) < 0))
    return rows, info


def routh_stability(coeffs) -> tuple[bool, int]:
    """``(stable, n_rhp)``: stability verdict and the number of right-half-plane roots from the Routh array."""
    _, info = routh_array(coeffs)
    n = info["sign_changes"]
    return (n == 0 and info["special"] is None), n


def is_hurwitz(coeffs) -> bool:
    """True if every root of the polynomial lies strictly in the left half-plane (checked by the roots)."""
    return bool(np.all(np.roots(_trim(coeffs)).real < 0))


def steady_state_error_constants(L: TF) -> dict:
    """Static error constants ``Kp, Kv, Ka`` and the system type of the open loop ``L`` (unity feedback).

    ``Kp = lim L(s)``, ``Kv = lim s L(s)``, ``Ka = lim s^2 L(s)`` as ``s -> 0``; the type is the number of poles
    at the origin.  Entries that are ``inf`` or ``0`` are returned as such.
    """
    den = L.den.copy()
    typ = 0
    while den.size > 1 and abs(den[-1]) < 1e-14:
        den = den[:-1]
        typ += 1
    g0 = L.num[-1] / den[-1]
    Kp = g0 if typ == 0 else np.inf
    Kv = 0.0 if typ == 0 else (g0 if typ == 1 else np.inf)
    Ka = 0.0 if typ < 2 else (g0 if typ == 2 else np.inf)
    return dict(type=typ, Kp=float(Kp), Kv=float(Kv), Ka=float(Ka))


# ======================================================================================================
# Root locus
# ======================================================================================================
def root_locus(G: TF, gains) -> np.ndarray:
    """Closed-loop poles of ``1 + K G(s) = 0`` for each ``K`` in ``gains``, tracked into continuous branches.

    Returns an array of shape ``(len(gains), n)``; column ``j`` follows branch ``j`` (it starts at open-loop pole
    ``j`` for a small first gain).  Branches are matched between consecutive gains by minimum total distance.
    """
    gains = np.asarray(gains, float)
    n = G.den.size - 1
    out = np.empty((len(gains), n), dtype=complex)
    prev = None
    for i, K in enumerate(gains):
        p = np.roots(np.polyadd(G.den, K * np.concatenate([np.zeros(G.den.size - G.num.size), G.num])))
        if p.size < n:                        # leading coefficient vanished: pad with huge roots
            p = np.concatenate([p, np.full(n - p.size, 1e12)])
        if prev is not None:
            cost = np.abs(prev[:, None] - p[None, :])
            r, c = optimize.linear_sum_assignment(cost)
            p = p[c[np.argsort(r)]]
        else:
            p = p[np.lexsort((p.imag, p.real))]                  # deterministic start order
        out[i] = p
        prev = p
    return out


def rl_asymptotes(G: TF) -> tuple[float, np.ndarray]:
    """``(centroid, angles_deg)`` of the asymptotes of the locus for ``K > 0``."""
    p, z = G.poles(), G.zeros()
    q = len(p) - len(z)
    if q <= 0:
        return float("nan"), np.array([])
    centroid = float((p.sum() - z.sum()).real / q)
    angles = np.array([(2 * k + 1) * 180.0 / q for k in range(q)])
    return centroid, angles


def rl_real_axis_segments(G: TF) -> list[tuple[float, float]]:
    """Intervals of the real axis on the locus for ``K > 0`` (to the left of an odd number of real poles+zeros)."""
    pts = sorted([r.real for r in np.concatenate([G.poles(), G.zeros()]) if abs(r.imag) < 1e-9], reverse=True)
    segs = []
    for i in range(0, len(pts), 2):
        hi = pts[i]
        lo = pts[i + 1] if i + 1 < len(pts) else -np.inf
        segs.append((float(lo), float(hi)))
    return segs


def rl_breakaway_points(G: TF) -> list[tuple[float, float]]:
    """Real breakaway / break-in points ``(sigma, K)`` on the locus for ``K > 0``.

    Solves ``d/ds [den/num] = 0``, i.e. ``den' num - den num' = 0``, and keeps real roots where ``K = -den/num > 0``.
    """
    d = np.polysub(np.polymul(np.polyder(G.den), G.num), np.polymul(G.den, np.polyder(G.num)))
    out = []
    for r in np.roots(d):
        if abs(r.imag) < 1e-7:
            s = r.real
            K = -np.polyval(G.den, s) / np.polyval(G.num, s)
            if K > 1e-12:
                out.append((float(s), float(K)))
    return sorted(out)


def rl_imaginary_crossings(G: TF, kmax: float = 1e6, n: int = 4000) -> list[tuple[float, float]]:
    """Gains at which the closed loop first reaches the imaginary axis, as ``(K, omega)`` pairs.

    Scans ``K`` on a logarithmic grid for sign changes of the largest real part of the closed-loop poles and
    refines each crossing with Brent's method.  A pair at ``omega = 0`` means a real root passes through the origin.
    """
    Ks = np.logspace(-6, np.log10(kmax), n)

    def amax(K):
        p = np.roots(np.polyadd(G.den, K * np.concatenate([np.zeros(G.den.size - G.num.size), G.num])))
        return p[np.argmax(p.real)]

    vals = np.array([amax(K).real for K in Ks])
    out = []
    for i in np.flatnonzero(np.sign(vals[:-1]) != np.sign(vals[1:])):
        K = optimize.brentq(lambda k: amax(k).real, Ks[i], Ks[i + 1], xtol=1e-12 * Ks[i + 1])
        out.append((float(K), float(abs(amax(K).imag))))
    return out


def rl_departure_angle(G: TF, pole: complex) -> float:
    """Departure angle (degrees) of the locus from the complex open-loop pole ``pole``.

    ``phi = 180 + sum(angles from zeros) - sum(angles from the other poles)``.
    """
    z, p = G.zeros(), [x for x in G.poles() if abs(x - pole) > 1e-9]
    ang = 180.0 + sum(np.degrees(np.angle(pole - zi)) for zi in z) - sum(np.degrees(np.angle(pole - pj)) for pj in p)
    return float((ang + 180.0) % 360.0 - 180.0)


def rl_gain_at(G: TF, s: complex) -> float:
    """Magnitude condition: the gain ``K = |den(s)/num(s)|`` for which ``s`` would be a closed-loop pole."""
    return float(abs(np.polyval(G.den, s) / np.polyval(G.num, s)))


# ======================================================================================================
# Frequency response
# ======================================================================================================
def bode(G: TF, w) -> tuple[np.ndarray, np.ndarray]:
    """``(magnitude_dB, phase_deg)`` of ``G(jw)`` with the phase unwrapped (starting from its principal value)."""
    w = np.asarray(w, float)
    H = G(1j * w)
    return 20 * np.log10(np.abs(H)), np.degrees(np.unwrap(np.angle(H)))


def margins(G: TF, wmin: float = 1e-3, wmax: float = 1e5, n: int = 20000) -> dict:
    """Gain and phase margins of the open loop ``G`` (unity negative feedback).

    Returns ``gm`` (linear factor by which the gain can grow, ``inf`` if there is no phase crossover),
    ``gm_db``, ``wpc`` (phase-crossover frequency, where ``G(jw)`` is real and negative), ``pm`` (degrees, ``inf`` if
    the magnitude never reaches 1) and ``wgc`` (gain-crossover frequency).  When there are several crossovers,
    the one giving the smallest margin is reported.
    """
    w = np.logspace(np.log10(wmin), np.log10(wmax), n)
    H = G(1j * w)
    # phase crossover: Im G(jw) changes sign while Re G(jw) < 0
    gm, wpc = np.inf, np.nan
    s = np.sign(H.imag)
    for i in np.flatnonzero(s[:-1] * s[1:] < 0):
        lw = optimize.brentq(lambda x: G(1j * 10 ** x).imag, np.log10(w[i]), np.log10(w[i + 1]))
        g = G(1j * 10 ** lw)
        if g.real < 0:
            gain = 1.0 / abs(g)
            if abs(np.log(gain)) < abs(np.log(gm)) if np.isfinite(gm) else True:
                gm, wpc = float(gain), float(10 ** lw)
    # gain crossover: |G| = 1
    pm, wgc = np.inf, np.nan
    f = np.abs(H) - 1.0
    for i in np.flatnonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0):
        lw = optimize.brentq(lambda x: abs(G(1j * 10 ** x)) - 1.0, np.log10(w[i]), np.log10(w[i + 1]))
        ph = np.degrees(np.angle(G(1j * 10 ** lw)))
        m = (ph + 360.0) % 360.0 - 180.0                 # phase + 180 deg, wrapped to (-180, 180]
        if abs(m) < abs(pm) if np.isfinite(pm) else True:
            pm, wgc = float(m), float(10 ** lw)
    return dict(gm=gm, gm_db=float(20 * np.log10(gm)) if np.isfinite(gm) else np.inf, wpc=wpc, pm=pm, wgc=wgc)


def nyquist_curve(G: TF, wmax: float = 1e4, n: int = 4000, eps: float = 1e-3) -> np.ndarray:
    """Points of ``G`` along the Nyquist contour: ``s`` from ``-j wmax`` to ``+j wmax``.

    Poles at the origin are bypassed by a semicircle of radius ``eps`` into the right half-plane (so they are
    not counted as unstable).  The large arc is omitted: for a strictly proper ``G`` it maps to the origin and
    the curve closes there.  Poles on the imaginary axis away from the origin are not supported.
    """
    p = G.poles()
    if np.any((np.abs(p.real) < 1e-9) & (np.abs(p.imag) > 1e-9)):
        raise NotImplementedError("imaginary-axis poles other than at the origin are not supported")
    at_origin = bool(np.any(np.abs(p) < 1e-9))
    if at_origin:
        wpos = np.logspace(np.log10(eps), np.log10(wmax), n)
        phi = np.linspace(-np.pi / 2, np.pi / 2, 400)
        return np.concatenate([G(-1j * wpos[::-1]), G(eps * np.exp(1j * phi)), G(1j * wpos)])
    wpos = np.concatenate([[0.0], np.logspace(-4, np.log10(wmax), n)])
    return np.concatenate([G(-1j * wpos[:0:-1]), G(1j * wpos)])


def nyquist_encirclements(G: TF, **kw) -> int:
    """Net clockwise encirclements ``N`` of the point ``-1`` by the Nyquist curve of ``G``.

    The Nyquist criterion reads ``Z = N + P``: ``Z`` closed-loop poles in the right half-plane, ``P`` open-loop ones.
    """
    c = nyquist_curve(G, **kw) + 1.0
    ang = np.unwrap(np.angle(c))
    return int(np.round(-(ang[-1] - ang[0]) / (2 * np.pi)))


def bandwidth(T: TF, wmax: float = 1e5, n: int = 20000) -> float:
    """-3 dB bandwidth of a (low-pass) closed-loop transfer function: first ``w`` where ``|T(jw)| = |T(0)|/sqrt(2)``."""
    w = np.logspace(-3, np.log10(wmax), n)
    m = np.abs(T(1j * w))
    ref = abs(T(1e-9j))
    below = np.flatnonzero(m < ref / np.sqrt(2))
    if below.size == 0:
        return float("inf")
    i = below[0]
    if i == 0:
        return float(w[0])
    lw = optimize.brentq(lambda x: abs(T(1j * 10 ** x)) - ref / np.sqrt(2), np.log10(w[i - 1]), np.log10(w[i]))
    return float(10 ** lw)


def sensitivity(L: TF) -> tuple[TF, TF]:
    """``(S, T)``: sensitivity ``1/(1+L)`` and complementary sensitivity ``L/(1+L)``."""
    d = np.polyadd(L.den, L.num)
    return TF(L.den, d), TF(L.num, d)


def peak_sensitivity(L: TF, wmin: float = 1e-3, wmax: float = 1e5, n: int = 20000) -> tuple[float, float]:
    """``(Ms, w)``: the peak of ``|S(jw)|`` (a robustness measure; ``1/Ms`` is the distance of the Nyquist curve to -1)."""
    S, _ = sensitivity(L)
    w = np.logspace(np.log10(wmin), np.log10(wmax), n)
    m = np.abs(S(1j * w))
    i = int(np.argmax(m))
    return float(m[i]), float(w[i])


# ======================================================================================================
# PID control, tuning rules
# ======================================================================================================
class PID:
    """Discrete PID controller in positional form with a filtered derivative, set-point weights and anti-windup.

    ``u = sat( kp (b r - y) + I + D )`` with

    * ``I`` updated each sample from the error ``r - y`` with gain ``ki`` (``ki = kp / Ti``),
    * ``D`` the filtered derivative of ``(c r - y)``, ``Tf d/dt D + D = kd d/dt (c r - y)``, ``Tf`` being the filter
      time constant (default ``kd / (kp N)`` with ``N = 10``),
    * ``antiwindup`` one of ``"none"``, ``"clamp"`` (conditional integration: the integrator is frozen while the
      output is saturated and the error pushes further into saturation) or ``"backcalc"`` (back-calculation with
      tracking time constant ``Tt``, default ``sqrt(Ti Td)`` or ``Ti`` for a PI controller).

    Call the object once per sample: ``u = pid(r, y)``.
    """

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0, Ts: float = 1e-3, N: float = 10.0,
                 Tf: float | None = None, umin: float = -np.inf, umax: float = np.inf, antiwindup: str = "none",
                 Tt: float | None = None, b: float = 1.0, c: float = 0.0, integral: str = "backward"):
        self.kp, self.ki, self.kd, self.Ts = float(kp), float(ki), float(kd), float(Ts)
        self.Tf = Tf if Tf is not None else (self.kd / (self.kp * N) if self.kp > 0 and self.kd > 0 else Ts)
        self.umin, self.umax, self.aw = umin, umax, antiwindup
        if Tt is None:
            Ti = self.kp / self.ki if self.ki > 0 else np.inf
            Td = self.kd / self.kp if self.kp > 0 else 0.0
            Tt = np.sqrt(Ti * Td) if (np.isfinite(Ti) and Td > 0) else (Ti if np.isfinite(Ti) else 1.0)
        self.Tt, self.b, self.c, self.integral = Tt, b, c, integral
        self.reset()

    def reset(self):
        self.I = 0.0
        self.D = 0.0
        self._prev_e = 0.0
        self._prev_dy = None
        self.u_unsat = 0.0
        self.u = 0.0

    def __call__(self, r: float, y: float) -> float:
        e = r - y
        P = self.kp * (self.b * r - y)
        # derivative of the weighted signal, filtered (backward Euler)
        dsig = self.c * r - y
        if self._prev_dy is None:
            self._prev_dy = dsig
        a = self.Tf / (self.Tf + self.Ts)
        self.D = a * self.D + self.kd / (self.Tf + self.Ts) * (dsig - self._prev_dy)
        self._prev_dy = dsig
        # integral candidate
        if self.integral == "tustin":
            dI = self.ki * self.Ts * 0.5 * (e + self._prev_e)
        elif self.integral == "forward":
            dI = self.ki * self.Ts * self._prev_e
        else:                                                    # backward Euler
            dI = self.ki * self.Ts * e
        self._prev_e = e
        I_new = self.I + dI
        u_unsat = P + I_new + self.D
        u = float(np.clip(u_unsat, self.umin, self.umax))
        if self.aw == "clamp":
            if u != u_unsat and np.sign(e) == np.sign(u_unsat - u):
                I_new = self.I                                    # freeze the integrator
        elif self.aw == "backcalc":
            I_new = self.I + dI + self.Ts * (u - u_unsat) / self.Tt
            u_unsat = P + I_new + self.D
            u = float(np.clip(u_unsat, self.umin, self.umax))
        self.I = I_new
        self.u_unsat, self.u = P + self.I + self.D, u
        return u


def zn_step(K: float, L: float, T: float, kind: str = "PID") -> dict:
    """Ziegler-Nichols *reaction-curve* rules for the plant ``K exp(-L s)/(T s + 1)``.

    ``kind`` is ``"P"``, ``"PI"`` or ``"PID"``.  Returns ``kp``, ``Ti``, ``Td``, ``ki = kp/Ti``, ``kd = kp Td``.
    """
    a = K * L / T
    table = {"P": (1 / a, np.inf, 0.0), "PI": (0.9 / a, L / 0.3, 0.0), "PID": (1.2 / a, 2 * L, 0.5 * L)}
    kp, Ti, Td = table[kind]
    return dict(kp=kp, Ti=Ti, Td=Td, ki=kp / Ti if np.isfinite(Ti) else 0.0, kd=kp * Td)


def zn_ultimate(Ku: float, Pu: float, kind: str = "PID") -> dict:
    """Ziegler-Nichols *ultimate-gain* rules from the ultimate gain ``Ku`` and ultimate period ``Pu``."""
    table = {"P": (0.5 * Ku, np.inf, 0.0), "PI": (0.45 * Ku, Pu / 1.2, 0.0), "PID": (0.6 * Ku, Pu / 2, Pu / 8)}
    kp, Ti, Td = table[kind]
    return dict(kp=kp, Ti=Ti, Td=Td, ki=kp / Ti if np.isfinite(Ti) else 0.0, kd=kp * Td)


def fit_foptd(t, y, u_step: float = 1.0) -> dict:
    """Tangent method: fit ``K exp(-L s)/(T s + 1)`` to a measured step response (input step of size ``u_step``).

    ``K`` is the final value over ``u_step``; ``L`` is where the tangent at the point of steepest slope meets the
    initial value and ``L + T`` where it meets the final value.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    y0, yf = y[0], float(np.mean(y[-max(2, len(y) // 20):]))
    dy = np.gradient(y, t)
    i = int(np.argmax(dy * np.sign(yf - y0)))
    slope = dy[i]
    t0 = t[i] - (y[i] - y0) / slope
    t1 = t[i] + (yf - y[i]) / slope
    return dict(K=(yf - y0) / u_step, L=float(max(t0, 0.0)), T=float(t1 - max(t0, 0.0)))


def ultimate_gain_period(G: TF) -> tuple[float, float]:
    """``(Ku, Pu)``: gain and oscillation period at which proportional control of ``G`` reaches the stability limit.

    Equivalent to the phase-crossover gain margin: ``Ku = 1 / |G(j w_pc)|`` and ``Pu = 2 pi / w_pc``.
    """
    m = margins(G)
    if not np.isfinite(m["gm"]):
        raise ValueError("no phase crossover: the plant has no finite ultimate gain")
    return float(m["gm"]), float(2 * np.pi / m["wpc"])


# ======================================================================================================
# Discretisation
# ======================================================================================================
def c2d_ss(A, B, C, D, Ts: float, method: str = "zoh"):
    """Discretise ``(A, B, C, D)``; ``method`` in ``"zoh"``, ``"bilinear"`` (Tustin), ``"euler"``, ``"backward_diff"``."""
    Ad, Bd, Cd, Dd, _ = signal.cont2discrete((np.atleast_2d(A), np.atleast_2d(B), np.atleast_2d(C), np.atleast_2d(D)),
                                             Ts, method=method)
    return Ad, Bd, Cd, Dd


def c2d_tf(G: TF, Ts: float, method: str = "zoh") -> tuple[np.ndarray, np.ndarray]:
    """Discretise a transfer function; returns ``(numz, denz)`` in descending powers of ``z``."""
    numz, denz, _ = signal.cont2discrete((G.num, G.den), Ts, method=method)
    return np.atleast_1d(np.squeeze(numz)), np.atleast_1d(denz)


def dlsim_tf(numz, denz, u) -> np.ndarray:
    """Output of the discrete transfer function ``numz/denz`` for the input samples ``u`` (zero initial state)."""
    return signal.lfilter(np.asarray(numz, float), np.asarray(denz, float), np.asarray(u, float))


# ======================================================================================================
# State space
# ======================================================================================================
def ctrb(A, B) -> np.ndarray:
    """Controllability matrix ``[B, AB, ..., A^(n-1) B]``."""
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    n = A.shape[0]
    cols = [B]
    for _ in range(n - 1):
        cols.append(A @ cols[-1])
    return np.hstack(cols)


def obsv(A, C) -> np.ndarray:
    """Observability matrix ``[C; CA; ...; C A^(n-1)]``."""
    A, C = np.atleast_2d(A), np.atleast_2d(C)
    n = A.shape[0]
    rows = [C]
    for _ in range(n - 1):
        rows.append(rows[-1] @ A)
    return np.vstack(rows)


def rank_controllable(A, B, tol: float | None = None) -> int:
    return int(np.linalg.matrix_rank(ctrb(A, B), tol))


def rank_observable(A, C, tol: float | None = None) -> int:
    return int(np.linalg.matrix_rank(obsv(A, C), tol))


def acker(A, B, poles) -> np.ndarray:
    """Ackermann's formula for a single-input system: ``K`` with ``eig(A - B K) = poles``.

    ``K = [0 ... 0 1] C^{-1} phi(A)`` where ``C`` is the controllability matrix and ``phi`` the desired
    characteristic polynomial.  Numerically fragile for ``n > 5`` or so; prefer :func:`place` then.
    """
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    n = A.shape[0]
    if B.shape[1] != 1:
        raise ValueError("Ackermann's formula needs a single input")
    phi = np.real(np.poly(poles))
    PhiA = np.zeros_like(A, dtype=float)
    for c in phi:
        PhiA = PhiA @ A + c * np.eye(n)
    e = np.zeros((1, n))
    e[0, -1] = 1.0
    return e @ np.linalg.solve(ctrb(A, B), PhiA)


def place(A, B, poles) -> np.ndarray:
    """Pole placement by SciPy's robust algorithm (multi-input capable); returns ``K`` with ``eig(A-BK) = poles``."""
    return signal.place_poles(np.atleast_2d(A), np.atleast_2d(B), poles).gain_matrix


def observer_gain(A, C, poles) -> np.ndarray:
    """Luenberger observer gain ``L`` (``n x p``) with ``eig(A - L C) = poles``, by duality with pole placement."""
    A, C = np.atleast_2d(A), np.atleast_2d(C)
    if C.shape[0] == 1:
        return acker(A.T, C.T, poles).T
    return place(A.T, C.T, poles).T


def lqr(A, B, Q, R):
    """Continuous-time LQR: returns ``(K, P, poles)`` minimising ``integral(x'Qx + u'Ru)``; ``u = -K x``."""
    A, B, Q, R = (np.atleast_2d(np.asarray(m, float)) for m in (A, B, Q, R))
    P = linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P, np.linalg.eigvals(A - B @ K)


def lqr_discrete(Ad, Bd, Q, R):
    """Discrete-time LQR: returns ``(K, P, poles)`` with ``u_k = -K x_k``."""
    Ad, Bd, Q, R = (np.atleast_2d(np.asarray(m, float)) for m in (Ad, Bd, Q, R))
    P = linalg.solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P, np.linalg.eigvals(Ad - Bd @ K)


def controllable_canonical(num, den):
    """Controllable canonical form ``(A, B, C, D)`` of a proper transfer function (last row of ``A`` = ``-a_n ... -a_1``)."""
    num, den = _trim(num), _trim(den)
    num, den = num / den[0], den / den[0]
    if num.size > den.size:
        raise ValueError("improper transfer function")
    n = den.size - 1
    num = np.concatenate([np.zeros(n + 1 - num.size), num])
    D = num[0]
    A = np.zeros((n, n))
    A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -den[:0:-1]
    B = np.zeros((n, 1))
    B[-1, 0] = 1.0
    C = (num[:0:-1] - D * den[:0:-1]).reshape(1, n)
    return A, B, C, np.array([[D]])


def ss_to_tf(A, B, C, D=0.0, out: int = 0, inp: int = 0) -> TF:
    """Transfer function from input ``inp`` to output ``out`` of ``(A, B, C, D)``, with numerically cancelled zeros trimmed."""
    D = np.atleast_2d(D)
    num, den = signal.ss2tf(np.atleast_2d(A), np.atleast_2d(B), np.atleast_2d(C), D, input=inp)
    return TF(_trim(num[out], 1e-10 * max(1.0, np.abs(num[out]).max())), den)


def prefilter_gain(A, B, C, K) -> float:
    """Reference gain ``Nbar`` for ``u = -K x + Nbar r`` so that the DC gain from ``r`` to ``y = C x`` is one."""
    A, B, C, K = (np.atleast_2d(m) for m in (A, B, C, K))
    return float(-1.0 / (C @ np.linalg.solve(A - B @ K, B)).item())


def numerical_jacobian(f: Callable, x, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of the vector function ``f`` at ``x``."""
    x = np.asarray(x, float)
    f0 = np.atleast_1d(f(x))
    J = np.zeros((f0.size, x.size))
    for j in range(x.size):
        h = eps * max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (np.atleast_1d(f(xp)) - np.atleast_1d(f(xm))) / (2 * h)
    return J


def linearize(f: Callable, x0, u0, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian linearisation ``(A, B)`` of ``xdot = f(x, u)`` about ``(x0, u0)``."""
    x0, u0 = np.atleast_1d(np.asarray(x0, float)), np.atleast_1d(np.asarray(u0, float))
    A = numerical_jacobian(lambda x: f(x, u0), x0, eps)
    B = numerical_jacobian(lambda u: f(x0, u), u0, eps)
    return A, B


# ======================================================================================================
# DC-motor joint
# ======================================================================================================
def dc_motor_ss(Kt: float, Ke: float, R: float, L: float, J: float, b: float, N: float = 1.0, kg: float = 0.0):
    """State-space model ``(A, B, C, D)`` of a voltage-driven DC motor with gearbox and a link, state ``[theta, omega, i]``.

    ``theta, omega`` are the **joint** angle and speed, ``i`` the armature current, ``V`` the input voltage and
    ``J`` the total inertia reflected to the joint (rotor ``N^2 J_m`` plus load).  ``kg`` is a restoring stiffness
    (N m/rad) of the linearised gravity torque: ``kg = m g d cos(theta0)`` for a pendulum about the hanging
    equilibrium, negative about the upright one.  The output is ``theta``.

    ``dtheta = omega;  J domega = N Kt i - b omega - kg theta;  L di = V - R i - N Ke omega``.
    """
    A = np.array([[0.0, 1.0, 0.0],
                  [-kg / J, -b / J, N * Kt / J],
                  [0.0, -N * Ke / L, -R / L]])
    B = np.array([[0.0], [0.0], [1.0 / L]])
    C = np.array([[1.0, 0.0, 0.0]])
    return A, B, C, np.zeros((1, 1))


def dc_motor_tf(Kt: float, Ke: float, R: float, L: float, J: float, b: float, N: float = 1.0, output: str = "angle") -> TF:
    """Transfer function from armature voltage to joint angle (``output="angle"``) or joint speed (``"speed"``).

    ``Omega/V = N Kt / [ (L s + R)(J s + b) + N^2 Kt Ke ]`` and ``Theta/V = Omega/(V s)``.
    """
    den = np.polyadd(np.polymul([L, R], [J, b]), [N**2 * Kt * Ke])
    G = TF([N * Kt], den)
    return G * integrator() if output == "angle" else G

"""One-degree-of-freedom models of bilateral teleoperation (Robot Motion Control, Chapter 6).

A *teleoperator* couples a **master** that a human operator moves with a **slave** that acts on a remote
**environment**, through a **communication channel** that may delay the signals.  This module provides

* the five elements of the loop for one translational degree of freedom
  (:class:`Body1D` for master and slave, :class:`Operator`, :class:`Environment`);
* controller architectures (:class:`FourChannel`, which contains unilateral, position-position,
  position-force and the transparent "four-channel" scheme as special cases, and :class:`WaveVariable`);
* a fixed-step time-domain simulator with transmission delay (:func:`simulate`);
* frequency-domain analysis of the *linear* architectures: the hybrid matrix of the two-port, transmitted
  impedance, Llewellyn's absolute-stability test and a passivity margin
  (:func:`hybrid_matrix`, :func:`transmitted_impedance`, :func:`llewellyn`, :func:`passivity_margin`);
* the characteristic equation of the delayed loop, its exact critical delay and its poles
  (:func:`loop_polys`, :func:`critical_delay`, :func:`loop_poles`);
* the scattering (wave-variable) transformation and energy bookkeeping of delayed channels
  (:func:`wave_encode`, :func:`wave_decode`, :func:`simulate_channel`, :func:`port_energy`);
* helpers for haptics (:func:`sampled_wall_radius`) and virtual fixtures (:func:`closest_on_polyline`,
  :func:`guidance_fixture`).

Sign conventions (used consistently in the notebooks)
-----------------------------------------------------
* ``x``, ``v``, ``a`` are position, velocity and acceleration of a body along the one axis; a positive
  force pushes in the positive direction.
* ``f_h`` is the force of the **hand** on the master.  ``f_e`` is the *magnitude* of the **environment's
  reaction** on the slave: the environment pushes the slave with force ``-f_e``.  A hand pushing a stiff wall
  therefore has ``f_h > 0`` and ``f_e > 0``, and in ideal transparency ``f_h = f_e`` and ``v_s = v_m``.
* Impedances are force / velocity: ``Z(s) = m s + b + k / s``.
* The two-port is described in the *power-in* convention of Hannaford: ``[f_h; -v_s] = H [v_m; f_e]``, so that
  the ideal (perfectly transparent) teleoperator has ``H = [[0, 1], [-1, 0]]``.

Everything is SI-like but unit-free: the equations only need consistent units.
"""

from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "Body1D", "Operator", "Environment", "Sig", "Result", "Architecture", "FourChannel", "WaveVariable",
    "simulate", "IDEAL_H", "hybrid_matrix", "transmitted_impedance", "llewellyn", "passivity_margin",
    "loop_polys", "critical_delay", "pade_exp", "loop_poles",
    "wave_encode", "wave_decode", "simulate_channel", "port_energy",
    "sampled_wall_radius", "sampled_wall_critical_K", "closest_on_polyline", "guidance_fixture",
]


# ----------------------------------------------------------------------------------------------------------
# 1.  The elements of the loop
# ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Body1D:
    """Master or slave as a mass with viscous friction and (optionally) a grounding spring:
    ``m x'' + b x' + k x = f``.  ``k`` is zero for a free mechanism and non-zero when, for example, gravity
    has been linearised about a hanging posture."""

    m: float
    b: float = 0.0
    k: float = 0.0

    def Z(self, s):
        """Mechanical impedance (force / velocity) at complex frequency ``s``."""
        return self.m * s + self.b + self.k / s


@dataclass(frozen=True)
class Operator:
    """Human operator: a passive impedance that tracks an intended motion ``ref(t)``.

    ``f_h = kh (ref(t) - x_m) - bh v_m``.  The mass of the hand should be added to the master's mass.
    """

    kh: float = 50.0
    bh: float = 5.0
    ref: Callable[[float], float] = lambda t: 0.0

    def force(self, t, x, v):
        return self.kh * (self.ref(t) - x) - self.bh * v

    def Z(self, s):
        """Impedance of the passive part (force / velocity)."""
        return self.bh + self.kh / s


@dataclass(frozen=True)
class Environment:
    """Spring-damper environment.

    With ``x0`` a number the environment is a one-sided *wall* at ``x = x0`` (no force while ``x <= x0``;
    ``f_e = max(k (x - x0) + b v, 0)`` afterwards).  With ``x0 = None`` it is the two-sided linear impedance
    ``f_e = k x + b v``, which is what the frequency-domain formulas assume.
    """

    k: float = 0.0
    b: float = 0.0
    x0: Optional[float] = 0.0

    def force(self, x, v):
        if self.x0 is None:
            return self.k * x + self.b * v
        pen = x - self.x0
        if pen <= 0.0:
            return 0.0
        return max(self.k * pen + self.b * v, 0.0)

    def Z(self, s):
        return self.b + self.k / s


#: Signals measured at one site: position, velocity, acceleration and the force sensed there
#: (hand force at the master, environment force at the slave).
Sig = namedtuple("Sig", "x v a f")


# ----------------------------------------------------------------------------------------------------------
# 2.  Controller architectures
# ----------------------------------------------------------------------------------------------------------
class Architecture:
    """Interface of a teleoperator controller.

    The controller is split into what the master side and the slave side compute, so that every signal that
    crosses the channel can be delayed by the simulator.  ``fwd`` and ``bwd`` build the vectors that are
    *sent*; ``master`` and ``slave`` return the actuator force from the local signals and the (delayed)
    *received* vector.
    """

    name = "architecture"

    def fwd(self, m: Sig, rx: np.ndarray) -> np.ndarray:  # master -> slave
        raise NotImplementedError

    def bwd(self, s: Sig, rx: np.ndarray) -> np.ndarray:  # slave -> master
        raise NotImplementedError

    def master(self, m: Sig, rx: np.ndarray) -> float:
        raise NotImplementedError

    def slave(self, s: Sig, rx: np.ndarray) -> float:
        raise NotImplementedError

    n_fwd = 4
    n_bwd = 4


def _pda(gain, x, v, a):
    """Force of a velocity-type block ``C(s) = kx/s + kv + ka s`` acting on the signals (x, v, a)."""
    kx, kv, ka = gain
    return kx * x + kv * v + ka * a


@dataclass(frozen=True)
class FourChannel(Architecture):
    """Lawrence's four-channel structure (and everything that is a special case of it).

    Every block ``C`` that acts on a velocity is written by three numbers ``(kx, kv, ka)``: its force is
    ``kx x + kv v + ka a`` (``C(s) = kx/s + kv + ka s``).  The blocks and the signals they carry are

    =====  ============================  =====================================================
    block  signal                        acts on
    =====  ============================  =====================================================
    c1     master (x, v, a), delayed     slave actuator       (forward position/velocity)
    c4     slave (x, v, a), delayed      master actuator      (backward position/velocity)
    cm     master (x, v, a), local       master actuator, with a minus sign
    cs     slave (x, v, a), local        slave actuator, with a minus sign
    c3     hand force f_h, delayed       slave actuator       (forward force)
    c2     environment force f_e, delayed master actuator, with a minus sign (force reflection)
    c6     hand force f_h, local         master actuator, with a minus sign
    c5     environment force f_e, local  slave actuator
    =====  ============================  =====================================================

    Actuator forces::

        f_cm = -cm(master) + c4(slave~) - c6 f_h - c2 f_e~
        f_cs =  c1(master~) - cs(slave) + c3 f_h~ + c5 f_e            (~ = received through the channel)

    Perfect transparency (``H = [[0, n_f], [-n_p, 0]]``) requires
    ``c1 = n_p (Zs + cs)``, ``c4 = (Zm + cm)/n_p``, ``c2 = n_f (1 - c6)``, ``c3 = (1 - c5)/n_f``
    (see :meth:`ideal`).
    """

    c1: tuple = (0.0, 0.0, 0.0)
    c4: tuple = (0.0, 0.0, 0.0)
    cm: tuple = (0.0, 0.0, 0.0)
    cs: tuple = (0.0, 0.0, 0.0)
    c2: float = 0.0
    c3: float = 0.0
    c5: float = 0.0
    c6: float = 0.0
    label: str = "four-channel"

    @property
    def name(self):  # type: ignore[override]
        return self.label

    # -- standard special cases -------------------------------------------------------------------------
    @classmethod
    def unilateral(cls, kp, kd):
        """Slave follows the master with a PD law; nothing is sent back (no force feedback)."""
        return cls(c1=(kp, kd, 0.0), cs=(kp, kd, 0.0), label="unilateral")

    @classmethod
    def position_position(cls, kp, kd, scale=1.0):
        """Symmetric P-P: a PD spring-damper joins master and slave positions, so the master is pulled
        towards the slave exactly as the slave is pulled towards the master."""
        g = (kp, kd, 0.0)
        return cls(c1=tuple(scale * c for c in g), c4=tuple(c / scale for c in g), cm=g, cs=g,
                   label="position-position")

    @classmethod
    def position_force(cls, kp, kd, kf=1.0, n_p=1.0):
        """P-F (force reflection): the slave tracks ``n_p x_m`` with a PD law and the *measured* environment
        force, multiplied by ``kf``, is applied to the master."""
        g = (kp, kd, 0.0)
        return cls(c1=tuple(n_p * c for c in g), cs=g, c2=kf, label="position-force")

    @classmethod
    def ideal(cls, master: Body1D, slave: Body1D, cm=(0.0, 0.0, 0.0), cs=(0.0, 0.0, 0.0), c5=0.0, c6=0.0,
              n_p=1.0, n_f=1.0, model_error=(1.0, 1.0)):
        """Four-channel controller that satisfies the transparency conditions for the *nominal* ``master``
        and ``slave``.  ``cm`` and ``cs`` (local servo gains) are free design parameters.
        ``model_error=(em, es)`` multiplies the nominal masses and frictions that the controller *believes*
        (to study robustness: the true plants stay as given to :func:`simulate`)."""
        em, es = model_error
        zm = (master.k * em, master.b * em, master.m * em)
        zs = (slave.k * es, slave.b * es, slave.m * es)
        c1 = tuple(n_p * (z + c) for z, c in zip(zs, cs))
        c4 = tuple((z + c) / n_p for z, c in zip(zm, cm))
        return cls(c1=c1, c4=c4, cm=tuple(cm), cs=tuple(cs), c2=n_f * (1.0 - c6), c3=(1.0 - c5) / n_f,
                   c5=c5, c6=c6, label="four-channel")

    @classmethod
    def blend(cls, master: Body1D, slave: Body1D, kp, kd, lam, inertia=True):
        """Interpolate between position-position (``lam = 0``) and the ideal four-channel controller
        (``lam = 1``): the force channels carry the fraction ``lam`` of the forces and the dynamics of the
        *other* arm (``Z_m`` and ``Z_s``) are fed forward with the same fraction.  ``inertia=False`` omits the
        acceleration terms (which need differentiated measurements)."""
        g = (kp, kd, 0.0)
        z = lambda b: (lam * b.k, lam * b.b, lam * b.m if inertia else 0.0)
        c1 = tuple(a + c for a, c in zip(g, z(slave)))
        c4 = tuple(a + c for a, c in zip(g, z(master)))
        return cls(c1=c1, c4=c4, cm=g, cs=g, c2=lam, c3=lam, label=f"four-channel blend (lambda={lam:g})")

    # -- time domain --------------------------------------------------------------------------------------
    def fwd(self, m, rx):
        return np.array([m.x, m.v, m.a, m.f])

    def bwd(self, s, rx):
        return np.array([s.x, s.v, s.a, s.f])

    def master(self, m, rx):
        return (-_pda(self.cm, m.x, m.v, m.a) + _pda(self.c4, *rx[:3]) - self.c6 * m.f - self.c2 * rx[3])

    def slave(self, s, rx):
        return (_pda(self.c1, *rx[:3]) - _pda(self.cs, s.x, s.v, s.a) + self.c3 * rx[3] + self.c5 * s.f)

    # -- frequency domain ---------------------------------------------------------------------------------
    def blocks(self, s):
        """The eight blocks as complex numbers at ``s`` (velocity-type blocks in force/velocity units)."""
        def C(g):
            kx, kv, ka = g
            return kx / s + kv + ka * s
        return dict(c1=C(self.c1), c4=C(self.c4), cm=C(self.cm), cs=C(self.cs),
                    c2=self.c2, c3=self.c3, c5=self.c5, c6=self.c6)


@dataclass(frozen=True)
class WaveVariable(Architecture):
    """Scattering (wave-variable) transmission with wave impedance ``b`` (Niemeyer & Slotine).

    Master side: velocity ``v_m`` is measured; the incoming wave ``w`` is received; the outgoing wave is
    ``u = sqrt(2b) v_m - w`` and the master actuator applies ``sqrt(2b) w - b v_m``.
    Slave side: the incoming wave ``w`` gives the commanded velocity ``v_cmd = sqrt(2/b) w - f_e / b``,
    followed by a velocity servo ``kv (v_cmd - v_s)``; the reflected wave is ``u = w - sqrt(2/b) f_e``.
    ``kc`` adds a weak position-error correction ``kc (x_m~ - x_s)`` to ``v_cmd`` (a heuristic against
    drift; it is *not* passive).
    """

    b: float = 10.0
    kv: float = 200.0
    kc: float = 0.0
    label: str = "wave variables"
    n_fwd = 2
    n_bwd = 1

    @property
    def name(self):  # type: ignore[override]
        return self.label

    def fwd(self, m, rx):
        return np.array([math.sqrt(2 * self.b) * m.v - rx[0], m.x])

    def bwd(self, s, rx):
        return np.array([rx[0] - math.sqrt(2.0 / self.b) * s.f])

    def master(self, m, rx):
        return math.sqrt(2 * self.b) * rx[0] - self.b * m.v

    def slave(self, s, rx):
        v_cmd = math.sqrt(2.0 / self.b) * rx[0] - s.f / self.b + self.kc * (rx[1] - s.x)
        return self.kv * (v_cmd - s.v)


# ----------------------------------------------------------------------------------------------------------
# 3.  Time-domain simulation
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Result:
    """Time histories of a simulation (all 1-D arrays of equal length)."""

    t: np.ndarray
    xm: np.ndarray
    vm: np.ndarray
    xs: np.ndarray
    vs: np.ndarray
    fh: np.ndarray       # hand force on the master
    fe: np.ndarray       # magnitude of the environment's reaction on the slave
    taum: np.ndarray     # master actuator force
    taus: np.ndarray     # slave actuator force
    dt: float = 1e-3

    def hand_energy(self):
        """Cumulative mechanical energy delivered by the hand, ``int f_h v_m dt``."""
        return np.cumsum(self.fh * self.vm) * self.dt

    def window(self, t0, t1):
        """Boolean mask of the samples with ``t0 <= t < t1``."""
        return (self.t >= t0) & (self.t < t1)


def simulate(master: Body1D, slave: Body1D, operator: Operator, env: Environment, arch: Architecture,
             t_end: float = 5.0, dt: float = 1e-3, delay=(0.0, 0.0), x0=(0.0, 0.0), blow_up: float = 1e6) -> Result:
    """Simulate the teleoperator loop with fixed step ``dt`` and channel delays ``delay = (T_fwd, T_bwd)``.

    A semi-implicit Euler step with implicit viscous friction is used.  The controller runs at every step
    (zero-order hold) and every transmitted signal is delayed by ``max(1, round(T/dt))`` samples: even an
    "instantaneous" channel costs one sample, as in any digital implementation.  The run stops early (and the
    arrays are truncated) when a position exceeds ``blow_up``.
    """
    n = int(round(t_end / dt))
    Nf = max(1, int(round(delay[0] / dt)))
    Nb = max(1, int(round(delay[1] / dt)))
    xm, vm = float(x0[0]), 0.0
    xs, vs = float(x0[1]), 0.0
    am = as_ = 0.0
    out = {k: np.zeros(n) for k in "t xm vm xs vs fh fe taum taus".split()}
    fwd_hist = np.zeros((n, _nsig(arch, "fwd")))
    bwd_hist = np.zeros((n, _nsig(arch, "bwd")))
    last = n
    for k in range(n):
        t = k * dt
        f_h = operator.force(t, xm, vm)
        f_e = env.force(xs, vs)
        m_sig, s_sig = Sig(xm, vm, am, f_h), Sig(xs, vs, as_, f_e)
        rx_f = fwd_hist[k - Nf] if k - Nf >= 0 else fwd_hist[0]
        rx_b = bwd_hist[k - Nb] if k - Nb >= 0 else bwd_hist[0]
        if k == 0:                                   # nothing has crossed the channel yet: send, then read
            fwd_hist[0] = arch.fwd(m_sig, np.zeros(_nsig(arch, "bwd")))
            bwd_hist[0] = arch.bwd(s_sig, np.zeros(_nsig(arch, "fwd")))
            rx_f, rx_b = fwd_hist[0], bwd_hist[0]
        else:
            fwd_hist[k] = arch.fwd(m_sig, rx_b)
            bwd_hist[k] = arch.bwd(s_sig, rx_f)
        tau_m = arch.master(m_sig, rx_b)
        tau_s = arch.slave(s_sig, rx_f)
        out["t"][k], out["xm"][k], out["vm"][k], out["xs"][k], out["vs"][k] = t, xm, vm, xs, vs
        out["fh"][k], out["fe"][k], out["taum"][k], out["taus"][k] = f_h, f_e, tau_m, tau_s
        vm_new = (master.m * vm + dt * (f_h + tau_m - master.k * xm)) / (master.m + dt * master.b)
        vs_new = (slave.m * vs + dt * (-f_e + tau_s - slave.k * xs)) / (slave.m + dt * slave.b)
        am, as_ = (vm_new - vm) / dt, (vs_new - vs) / dt
        vm, vs = vm_new, vs_new
        xm += dt * vm
        xs += dt * vs
        if abs(xm) > blow_up or abs(xs) > blow_up or not (np.isfinite(xm) and np.isfinite(xs)):
            last = k + 1
            break
    return Result(dt=dt, **{k: v[:last] for k, v in out.items()})


def _nsig(arch: Architecture, which: str) -> int:
    """Length of the vector sent by ``fwd`` or ``bwd`` (internal helper)."""
    return arch.n_fwd if which == "fwd" else arch.n_bwd


# ----------------------------------------------------------------------------------------------------------
# 4.  Frequency domain: hybrid matrix, transmitted impedance, Llewellyn, passivity
# ----------------------------------------------------------------------------------------------------------
IDEAL_H = np.array([[0.0, 1.0], [-1.0, 0.0]])


def hybrid_matrix(master: Body1D, slave: Body1D, arch: FourChannel, w, delay=(0.0, 0.0)) -> np.ndarray:
    """Hybrid matrix ``H(jw)`` of the linear teleoperator, ``[f_h; -v_s] = H [v_m; f_e]`` (array ``(len(w), 2, 2)``).

    ``delay = (T_f, T_b)`` delays blocks c1, c3 (forward) and c2, c4 (backward).
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    H = np.zeros((len(w), 2, 2), dtype=complex)
    for i, wi in enumerate(w):
        s = 1j * wi
        C = arch.blocks(s)
        ef, eb = np.exp(-s * delay[0]), np.exp(-s * delay[1])
        Zm, Zs = master.Z(s), slave.Z(s)
        A = np.array([[1 - C["c6"], C["c4"] * eb], [C["c3"] * ef, -(Zs + C["cs"])]])
        B = np.array([[Zm + C["cm"], C["c2"] * eb], [-C["c1"] * ef, 1 - C["c5"]]])
        G = np.linalg.solve(A, B)                    # [f_h; v_s] = G [v_m; f_e]
        H[i] = np.diag([1.0, -1.0]) @ G
    return H


def transmitted_impedance(H: np.ndarray, Ze) -> np.ndarray:
    """Impedance ``Z_to = f_h / v_m`` felt by the hand when the slave terminates in ``Z_e = f_e / v_s``.

    ``Z_to = h11 - h12 h21 Ze / (1 + h22 Ze)``; for ideal ``H`` this is ``Ze``.
    """
    h11, h12, h21, h22 = H[..., 0, 0], H[..., 0, 1], H[..., 1, 0], H[..., 1, 1]
    return h11 - h12 * h21 * Ze / (1 + h22 * Ze)


def llewellyn(H: np.ndarray) -> dict:
    """The three frequency-wise quantities of Llewellyn's absolute-stability criterion for a two-port with
    hybrid matrix ``H``: ``Re h11 >= 0``, ``Re h22 >= 0`` and
    ``2 Re h11 Re h22 - Re(h12 h21) - |h12 h21| >= 0``.  (The criterion also requires that ``h11`` and ``h22``
    have no poles in the open right half plane, which holds for all architectures here when the loop is
    stable.)  Returns the three arrays under the keys ``"r11"``, ``"r22"``, ``"c"``."""
    h11, h12, h21, h22 = H[..., 0, 0], H[..., 0, 1], H[..., 1, 0], H[..., 1, 1]
    c = 2 * h11.real * h22.real - (h12 * h21).real - np.abs(h12 * h21)
    return dict(r11=h11.real, r22=h22.real, c=c)


def passivity_margin(H: np.ndarray) -> np.ndarray:
    """Smallest eigenvalue of the Hermitian part ``(H + H^H)/2`` at each frequency.  A stable two-port is
    passive iff this is non-negative at all frequencies."""
    Hh = 0.5 * (H + np.conj(np.swapaxes(H, -1, -2)))
    return np.linalg.eigvalsh(Hh)[..., 0]


# ----------------------------------------------------------------------------------------------------------
# 5.  Closed loop with delay: characteristic equation, critical delay, poles
# ----------------------------------------------------------------------------------------------------------
def loop_polys(master: Body1D, slave: Body1D, arch: FourChannel, hand=(0.0, 0.0), env=(0.0, 0.0)):
    """Polynomial matrix of the closed loop (hand and environment modelled as spring-dampers).

    With position ``x`` as the variable and ``hand = (k_h, b_h)``, ``env = (k_e, b_e)`` the loop equations are::

        P11(s) x_m = P12(s) exp(-s T_b) x_s            P22(s) x_s = P21(s) exp(-s T_f) x_m

    so the characteristic equation is ``P11 P22 - P12 P21 exp(-s (T_f + T_b)) = 0``: **only the round-trip
    delay matters**.  Returns the four :class:`numpy.poly1d` objects ``(P11, P22, P12, P21)``.
    """
    def pol(g):                                  # kx + kv s + ka s^2
        return np.poly1d([g[2], g[1], g[0]])

    Zh, Ze = np.poly1d([hand[1], hand[0]]), np.poly1d([env[1], env[0]])
    Zm, Zs = np.poly1d([master.m, master.b, master.k]), np.poly1d([slave.m, slave.b, slave.k])
    P11 = Zm + (1 - arch.c6) * Zh + pol(arch.cm)
    P22 = Zs + (1 - arch.c5) * Ze + pol(arch.cs)
    P12 = pol(arch.c4) - arch.c2 * Ze
    P21 = pol(arch.c1) - arch.c3 * Zh
    return P11, P22, P12, P21


def critical_delay(master, slave, arch, hand=(0.0, 0.0), env=(0.0, 0.0), w_max=3000.0, n=300000):
    """Smallest **round-trip** delay ``T_f + T_b`` at which the loop first loses stability, and the angular
    frequency of the oscillation: ``(T_round, omega)``; ``(inf, nan)`` if no delay destabilises it.

    On the stability boundary ``s = j w`` and ``P11 P22 / (P12 P21) = exp(-j w T)``: the modulus fixes ``w``
    and the phase fixes ``T``.  The delay-free loop is assumed stable.
    """
    P11, P22, P12, P21 = loop_polys(master, slave, arch, hand, env)
    w = np.linspace(1e-3, w_max, n)
    jw = 1j * w
    D = P11(jw) * P22(jw) / (P12(jw) * P21(jw))
    g = np.abs(D) - 1.0
    idx = np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]
    best = (math.inf, math.nan)
    for i in idx:
        wc = w[i] - g[i] * (w[i + 1] - w[i]) / (g[i + 1] - g[i])
        Dc = P11(1j * wc) * P22(1j * wc) / (P12(1j * wc) * P21(1j * wc))
        T = (-np.angle(Dc)) % (2 * np.pi) / wc          # e^{-j w T} = D  ->  w T = -arg D (mod 2 pi)
        if T < best[0]:
            best = (T, wc)
    return best


def pade_exp(T: float, order: int = 6):
    """Padé approximant of ``exp(-s T)`` as ``(numerator, denominator)`` :class:`numpy.poly1d` objects."""
    n = order
    c = [math.factorial(2 * n - k) * math.factorial(n) / (math.factorial(2 * n) * math.factorial(k) * math.factorial(n - k))
         for k in range(n + 1)]
    num = np.poly1d([c[k] * (-T) ** k for k in range(n, -1, -1)])
    den = np.poly1d([c[k] * T ** k for k in range(n, -1, -1)])
    return num, den


def loop_poles(master, slave, arch, T_round, hand=(0.0, 0.0), env=(0.0, 0.0), order: int = 6) -> np.ndarray:
    """Closed-loop poles for a round-trip delay ``T_round`` (Padé approximation of the delay; the extra poles it
    introduces lie far in the left half plane)."""
    P11, P22, P12, P21 = loop_polys(master, slave, arch, hand, env)
    if T_round <= 0:
        return (P11 * P22 - P12 * P21).roots
    num, den = pade_exp(T_round, order)
    return (P11 * P22 * den - P12 * P21 * num).roots


# ----------------------------------------------------------------------------------------------------------
# 6.  Scattering (wave) variables and delayed channels
# ----------------------------------------------------------------------------------------------------------
def wave_encode(v, F, b):
    """Scattering transformation: ``u = (b v + F)/sqrt(2b)`` (incoming), ``w = (b v - F)/sqrt(2b)`` (outgoing).
    Power into the port is ``F v = (u^2 - w^2)/2``."""
    r = math.sqrt(2.0 * b)
    return (b * np.asarray(v) + np.asarray(F)) / r, (b * np.asarray(v) - np.asarray(F)) / r


def wave_decode(u, w, b):
    """Inverse of :func:`wave_encode`: returns ``(v, F)``."""
    u, w = np.asarray(u), np.asarray(w)
    return (u + w) / math.sqrt(2.0 * b), (u - w) * math.sqrt(b / 2.0)


def simulate_channel(kind: str, N, v_L, F_R, dt: float = 1.0, b: float = 1.0):
    """Drive a delayed two-port with arbitrary signals and return what it does to them.

    Left port: the velocity ``v_L`` is imposed; right port: the force ``F_R`` is imposed.  All forces are
    forces *exerted by the ports on the channel* and both velocities point the same way, so a rigid rod
    has ``F_L = -F_R`` and ``v_R = v_L`` and absorbs no power.  ``N`` is the one-way delay in samples (an
    integer, or an array giving the delay at each sample for time-varying delay).

    ``kind = "direct"``: velocity travels left to right and force right to left unchanged.
    ``kind = "wave"``: the channel is a matched transmission line of impedance ``b`` carrying the waves
    ``u_L`` and ``u_R`` (see :func:`wave_encode`).

    Returns ``(F_L, v_R)``.  Use :func:`port_energy` to test passivity.
    """
    v_L, F_R = np.asarray(v_L, float), np.asarray(F_R, float)
    n = len(v_L)
    Nn = np.broadcast_to(np.asarray(N), (n,)).astype(int)
    F_L, v_R = np.zeros(n), np.zeros(n)
    if kind == "direct":
        for k in range(n):
            j = k - Nn[k]
            v_R[k] = v_L[j] if j >= 0 else 0.0
            F_L[k] = -F_R[j] if j >= 0 else 0.0
        return F_L, v_R
    r = math.sqrt(2.0 * b)
    u_L, u_R = np.zeros(n), np.zeros(n)
    for k in range(n):
        j = k - Nn[k]
        w_R = u_L[j] if j >= 0 else 0.0                    # wave arriving at the right port
        w_L = u_R[j] if j >= 0 else 0.0                    # wave arriving at the left port
        v_R[k] = (r * w_R + F_R[k]) / b
        u_R[k] = r * v_R[k] - w_R
        u_L[k] = r * v_L[k] - w_L
        F_L[k] = b * v_L[k] - r * w_L
    return F_L, v_R


def port_energy(F_L, v_L, F_R, v_R, dt: float = 1.0):
    """Cumulative energy absorbed by a two-port: ``int (F_L v_L + F_R v_R) dt`` (never negative for a passive
    two-port that starts empty)."""
    return np.cumsum(np.asarray(F_L) * np.asarray(v_L) + np.asarray(F_R) * np.asarray(v_R)) * dt


# ----------------------------------------------------------------------------------------------------------
# 7.  Haptics: the sampled virtual wall
# ----------------------------------------------------------------------------------------------------------
def sampled_wall_radius(m: float, b: float, K: float, B: float, Ts: float) -> float:
    """Spectral radius of the sampled-data loop of a device (mass ``m``, viscous friction ``b``) that renders a
    wall ``f = -K x - B v`` computed once per sample ``Ts`` and held constant (zero-order hold), while the
    hand does not interfere.  The loop is stable iff the value is below one."""
    from scipy.linalg import expm

    A = np.array([[0.0, 1.0], [0.0, -b / m]])
    Bv = np.array([[0.0], [1.0 / m]])
    M = np.zeros((3, 3))
    M[:2, :2], M[:2, 2:] = A * Ts, Bv * Ts
    E = expm(M)
    Phi, Gam = E[:2, :2], E[:2, 2:]
    cl = Phi - Gam @ np.array([[K, B]])
    return float(np.max(np.abs(np.linalg.eigvals(cl))))


def sampled_wall_critical_K(m: float, b: float, B: float, Ts: float, K_max: float = 1e6) -> float:
    """Largest wall stiffness (found by bisection) for which :func:`sampled_wall_radius` is below one."""
    lo, hi = 0.0, K_max
    if sampled_wall_radius(m, b, hi, B, Ts) < 1.0:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if sampled_wall_radius(m, b, mid, B, Ts) < 1.0:
            lo = mid
        else:
            hi = mid
    return lo


# ----------------------------------------------------------------------------------------------------------
# 8.  Virtual fixtures (planar)
# ----------------------------------------------------------------------------------------------------------
def closest_on_polyline(P, p):
    """Closest point of the polyline ``P`` (array ``(n, 2)``) to ``p``.

    Returns ``(q, t_hat, dist, s)``: the point, the unit tangent of its segment, the signed distance (positive
    to the left of the direction of travel) and the arc length from the start of the polyline.
    """
    P, p = np.asarray(P, float), np.asarray(p, float)
    best = None
    s0 = 0.0
    for a, b in zip(P[:-1], P[1:]):
        d = b - a
        L = np.linalg.norm(d)
        u = np.clip(np.dot(p - a, d) / L ** 2, 0.0, 1.0)
        q = a + u * d
        dist = np.linalg.norm(p - q)
        if best is None or dist < best[2]:
            t_hat = d / L
            sign = np.sign(t_hat[0] * (p - q)[1] - t_hat[1] * (p - q)[0]) or 1.0
            best = (q, t_hat, sign * dist, s0 + u * L)
        s0 += L
    return best


def guidance_fixture(v_cmd, t_hat, lam: float = 0.0):
    """Guidance virtual fixture on a commanded velocity: keep the component along the path direction
    ``t_hat`` and scale the perpendicular component by the *compliance* ``lam`` in [0, 1]
    (``lam = 0`` is a hard fixture, ``lam = 1`` removes the assistance)."""
    v_cmd, t_hat = np.asarray(v_cmd, float), np.asarray(t_hat, float)
    t_hat = t_hat / np.linalg.norm(t_hat)
    v_par = np.dot(v_cmd, t_hat) * t_hat
    return v_par + lam * (v_cmd - v_par)

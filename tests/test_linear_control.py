"""Tests for rgkit.linear_control: every helper is checked against an independent calculation."""

import numpy as np
import pytest
from scipy import linalg, signal

from rgkit import linear_control as lc
from rgkit.linear_control import TF

rng = np.random.default_rng(1)


# ---------------------------------------------------------------- transfer-function algebra
def test_tf_algebra_and_feedback():
    G = TF([10], [1, 3, 2])
    assert np.allclose((G * G).den, [1, 6, 13, 12, 4])
    assert np.allclose(G.feedback().den, [1, 3, 12])
    assert np.allclose(G.feedback().num, [10])
    # (G + G) = 2G
    assert np.allclose((G + G)(2j), 2 * G(2j))
    # series/parallel helpers
    H = TF([1], [1, 1])
    assert np.allclose(lc.series(G, H)(1j), G(1j) * H(1j))
    assert np.allclose(lc.parallel(G, H)(1j), G(1j) + H(1j))
    # positive feedback
    assert np.allclose(G.feedback(sign=+1).den, [1, 3, -8])
    # feedback with a sensor H: G/(1+GH)
    assert np.allclose(G.feedback(H)(0.7j), G(0.7j) / (1 + G(0.7j) * H(0.7j)))


def test_tf_step_dcgain_poles_minreal():
    G = TF([2], [1, 3, 2])                       # poles -1, -2 ; dc gain 1
    assert np.isclose(G.dcgain(), 1.0)
    assert np.allclose(sorted(G.poles().real), [-2, -1])
    t = np.linspace(0, 10, 500)
    y = G.step(t)
    assert np.isclose(y[-1], 1.0, atol=1e-3)
    exact = 1 - 2 * np.exp(-t) + np.exp(-2 * t)
    assert np.allclose(y, exact, atol=1e-6)
    cancelled = TF([1, 1], [1, 3, 2]).minreal()
    assert cancelled.order == 1 and np.allclose(cancelled.poles(), [-2])
    assert lc.integrator().dcgain() == np.inf


def test_pade_delay_phase():
    T = 0.1
    P = lc.pade_delay(T, 3)
    w = 5.0
    assert np.isclose(np.angle(P(1j * w)), -w * T, atol=1e-4)
    assert np.isclose(abs(P(1j * w)), 1.0)


# ---------------------------------------------------------------- second-order metrics
@pytest.mark.parametrize("zeta", [0.2, 0.5, 0.7])
def test_second_order_metrics_match_simulation(zeta):
    wn = 8.0
    G = lc.second_order(wn, zeta)
    m = lc.second_order_metrics(wn, zeta)
    t = np.linspace(0, 6, 60001)
    sm = lc.step_metrics(t, G.step(t), final=1.0)
    assert np.isclose(sm["Mp"], m["Mp"], atol=2e-4)
    i = np.argmax(G.step(t))
    assert np.isclose(t[i], m["tp"], atol=2e-3)
    # the closed-form settling-time estimate is a bound on the true 2 % settling time
    assert sm["ts"] <= m["ts"] + 1e-3


def test_overshoot_inverse_and_specs():
    for Mp in (0.05, 0.1, 0.25):
        z = lc.zeta_from_overshoot(Mp)
        assert np.isclose(lc.second_order_metrics(1.0, z)["Mp"], Mp)
    wn, z = lc.second_order_from_specs(0.1, 0.5)
    m = lc.second_order_metrics(wn, z)
    assert np.isclose(m["Mp"], 0.1) and np.isclose(m["ts"], 0.5)


def test_step_metrics_first_order():
    tau = 0.2
    t = np.linspace(0, 2, 20001)
    y = 1 - np.exp(-t / tau)
    m = lc.step_metrics(t, y, final=1.0)
    assert np.isclose(m["tr"], tau * np.log(9), atol=1e-3)
    assert np.isclose(m["ts"], -tau * np.log(0.02), atol=1e-3)
    assert m["Mp"] == 0.0


# ---------------------------------------------------------------- Routh
def test_routh_counts_rhp_roots_random():
    for _ in range(200):
        n = rng.integers(2, 8)
        roots = rng.normal(size=n) + 1j * rng.normal(size=n)
        roots = np.concatenate([roots, roots.conj()])           # real polynomial of degree 2n
        coeffs = np.real(np.poly(roots)) * rng.uniform(0.5, 2)
        _, info = lc.routh_array(coeffs)
        if info["special"] is None:
            assert info["sign_changes"] == int(np.sum(np.roots(coeffs).real > 0))


def test_routh_textbook_cases():
    # s^3 + 2 s^2 + 3 s + 10 : two RHP roots (roots at -2 and 0 +- j sqrt 5 ... check against numpy)
    for c in ([1, 2, 3, 10], [1, 6, 11, 6], [1, 1, 2, 8, 12, 0.5]):
        _, info = lc.routh_array(c)
        assert info["sign_changes"] == int(np.sum(np.roots(c).real > 0))
    stable, n = lc.routh_stability([1, 6, 11, 6])
    assert stable and n == 0


def test_routh_zero_pivot_and_zero_row():
    # s^4 + s^3 + 2 s^2 + 2 s + 3 -> zero in the first column handled via epsilon
    c = [1, 1, 2, 2, 3]
    _, info = lc.routh_array(c)
    assert info["special"] == "zero_pivot"
    assert info["sign_changes"] == int(np.sum(np.roots(c).real > 0)) == 2
    # s^5 + 2 s^4 + 24 s^3 + 48 s^2 - 25 s - 50 : zero row; auxiliary polynomial 2 s^4 + 48 s^2 - 50
    c = [1, 2, 24, 48, -25, -50]
    _, info = lc.routh_array(c)
    assert info["special"] == "zero_row"
    assert info["sign_changes"] == int(np.sum(np.roots(c).real > 0)) == 1


def test_routh_symbolic_gain_range():
    import sympy as sp
    K = sp.symbols("K", positive=True)
    rows, info = lc.routh_array([1, 6, 8, K])        # s^3 + 6 s^2 + 8 s + K : stable for 0 < K < 48
    first = info["first_column"]
    assert sp.simplify(first[2] - (48 - K) / 6) == 0
    assert sp.solve(first[2] > 0, K) == (K < 48)


# ---------------------------------------------------------------- steady-state error constants
def test_error_constants():
    L1 = TF([5], [1, 2, 0])                     # type 1, Kv = 2.5
    c = lc.steady_state_error_constants(L1)
    assert c["type"] == 1 and np.isclose(c["Kv"], 2.5) and c["Kp"] == np.inf and c["Ka"] == 0
    L0 = TF([4], [1, 3])
    c = lc.steady_state_error_constants(L0)
    assert c["type"] == 0 and np.isclose(c["Kp"], 4 / 3) and c["Kv"] == 0
    L2 = TF([3, 3], [1, 0, 0, 5])
    assert lc.steady_state_error_constants(TF([2], [1, 0, 0]))["Ka"] == 2


# ---------------------------------------------------------------- root locus
def test_root_locus_properties():
    G = TF([1], [1, 3, 2, 0])                   # K/(s(s+1)(s+2))
    c, ang = lc.rl_asymptotes(G)
    assert np.isclose(c, -1.0) and np.allclose(ang, [60, 180, 300])
    br = lc.rl_breakaway_points(G)
    assert len(br) == 1 and np.isclose(br[0][0], -0.4226, atol=1e-3) and np.isclose(br[0][1], 0.3849, atol=1e-3)
    (K, w), = lc.rl_imaginary_crossings(G)[:1]
    assert np.isclose(K, 6.0, rtol=1e-6) and np.isclose(w, np.sqrt(2), rtol=1e-6)
    segs = lc.rl_real_axis_segments(G)
    assert (-1.0, 0.0) in segs and (-np.inf, -2.0) in segs
    gains = np.concatenate([[0], np.logspace(-3, 2, 800)])
    P = lc.root_locus(G, gains)
    assert P.shape == (801, 3)
    # branches are continuous (no jumps beyond a modest step) and every point satisfies the characteristic equation
    jumps = np.abs(np.diff(P, axis=0)).max()
    assert jumps < 0.6
    for K, row in zip(gains[::97], P[::97]):
        assert np.allclose(np.polyval(np.polyadd(G.den, K * G.num), row), 0, atol=1e-7)
    assert np.isclose(lc.rl_gain_at(G, P[400, 0]), gains[400], rtol=1e-6)


def test_departure_angle():
    G = TF([1], np.polymul([1, 2, 2], [1, 0]))          # poles 0, -1+-j
    # textbook value: departure from -1+j is -45 deg for this configuration ... verify by tracking the locus
    ang = lc.rl_departure_angle(G, -1 + 1j)
    gains = np.array([0, 1e-4])
    P = lc.root_locus(G, gains)
    p0 = [p for p in P[0] if abs(p - (-1 + 1j)) < 1e-6][0]
    j = int(np.argmin(np.abs(P[0] - p0)))
    d = P[1, j] - P[0, j]
    assert np.isclose(np.degrees(np.angle(d)), ang, atol=0.5)


# ---------------------------------------------------------------- frequency response
def test_bode_margins_known_case():
    # L = 10 / (s (s+1) (s+5)): compare to brute-force evaluation
    L = TF([10], np.polymul([1, 0], np.polymul([1, 1], [1, 5])))
    m = lc.margins(L)
    w = np.logspace(-2, 2, 400001)
    H = L(1j * w)
    i = np.argmin(np.abs(np.abs(H) - 1))
    assert np.isclose(m["wgc"], w[i], rtol=1e-3)
    assert np.isclose(m["pm"], np.degrees(np.angle(H[i])) + 180, atol=0.05)
    # phase crossover of this loop: s^3 + 6 s^2 + 5 s -> w = sqrt 5, gm = 30/10 = 3
    assert np.isclose(m["wpc"], np.sqrt(5), rtol=1e-5)
    assert np.isclose(m["gm"], 3.0, rtol=1e-5)
    mag, ph = lc.bode(L, np.array([1.0]))
    assert np.isclose(mag[0], 20 * np.log10(abs(L(1j))))


def test_margins_predict_closed_loop_stability():
    for K in (2.0, 20.0, 29.0, 31.0, 40.0):
        L = TF([K], np.polymul([1, 0], np.polymul([1, 1], [1, 5])))
        m = lc.margins(L)
        stable = L.feedback().is_stable()
        assert stable == (m["gm"] > 1 and m["pm"] > 0)


def test_nyquist_criterion():
    den = np.polymul([1, 0], np.polymul([1, 1], [1, 2]))
    for K, expect_Z in ((2.0, 0), (5.0, 0), (7.0, 2), (20.0, 2)):
        L = TF([K], den)
        N = lc.nyquist_encirclements(L)
        P = int(np.sum(L.poles().real > 1e-9))
        assert N + P == expect_Z == int(np.sum(L.feedback().poles().real > 0))
    # open-loop unstable: L = K/(s-1): closed loop stable for K > 1
    for K in (0.5, 2.0):
        L = TF([K], [1, -1])
        assert lc.nyquist_encirclements(L) + 1 == int(np.sum(L.feedback().poles().real > 0))


def test_bandwidth_and_sensitivity():
    wn = 10.0
    T = lc.second_order(wn, 0.7071067811865476)
    assert np.isclose(lc.bandwidth(T), wn, rtol=1e-3)           # Butterworth: -3 dB exactly at wn
    L = TF([10], [1, 1, 0])
    S, Tc = lc.sensitivity(L)
    s = 0.3 + 2j
    assert np.isclose(S(s) + Tc(s), 1.0)
    Ms, w = lc.peak_sensitivity(L)
    assert Ms > 1 and w > 0


# ---------------------------------------------------------------- tuning rules and PID
def test_zn_rules():
    r = lc.zn_step(K=2.0, L=0.1, T=1.0, kind="PID")
    assert np.isclose(r["kp"], 6.0) and np.isclose(r["Ti"], 0.2) and np.isclose(r["Td"], 0.05)
    assert np.isclose(r["ki"], 30.0) and np.isclose(r["kd"], 0.3)
    u = lc.zn_ultimate(Ku=10, Pu=2.0)
    assert np.isclose(u["kp"], 6.0) and np.isclose(u["Ti"], 1.0) and np.isclose(u["Td"], 0.25)
    assert lc.zn_ultimate(10, 2, "P")["ki"] == 0


def test_ultimate_gain_matches_routh():
    G = TF([1], np.polymul([1, 0], np.polymul([1, 1], [1, 2])))
    Ku, Pu = lc.ultimate_gain_period(G)
    assert np.isclose(Ku, 6.0, rtol=1e-6) and np.isclose(Pu, 2 * np.pi / np.sqrt(2), rtol=1e-6)


def test_fit_foptd_recovers_parameters():
    K, L, T = 2.0, 0.3, 1.5
    t = np.linspace(0, 12, 12001)
    y = K * (1 - np.exp(-(t - L) / T)) * (t > L)
    f = lc.fit_foptd(t, y)
    assert np.isclose(f["K"], K, rtol=1e-3) and np.isclose(f["L"], L, atol=0.01) and np.isclose(f["T"], T, atol=0.02)


def _sim(plant_num, plant_den, pid, r, n, Ts, dist=None):
    """Simulate a continuous plant (ZOH-discretised) under the discrete controller ``pid``."""
    A, B, C, D = signal.tf2ss(plant_num, plant_den)
    Ad, Bd, Cd, Dd = lc.c2d_ss(A, B, C, D, Ts)
    x = np.zeros((A.shape[0], 1))
    y, u = np.zeros(n), np.zeros(n)
    for k in range(n):
        y[k] = (Cd @ x).item()
        u[k] = pid(r, y[k])
        d = 0.0 if dist is None else dist(k * Ts)
        x = Ad @ x + Bd * (u[k] + d)
    return y, u


def test_pid_removes_steady_state_error_and_respects_limits():
    Ts = 0.002
    pid = lc.PID(kp=5.0, ki=5.0, kd=0.2, Ts=Ts)
    y, u = _sim([1.0], [1, 2, 1], pid, 1.0, 15000, Ts)
    assert abs(y[-1] - 1.0) < 1e-3
    # constant input disturbance rejected by the integral action
    pid = lc.PID(kp=5.0, ki=5.0, kd=0.2, Ts=Ts)
    y, _ = _sim([1.0], [1, 2, 1], pid, 1.0, 15000, Ts, dist=lambda t: -0.5)
    assert abs(y[-1] - 1.0) < 1e-3
    # P-only leaves an offset under the same disturbance
    pid = lc.PID(kp=5.0, Ts=Ts)
    y, _ = _sim([1.0], [1, 2, 1], pid, 1.0, 15000, Ts, dist=lambda t: -0.5)
    assert abs(y[-1] - (5 * 1.0 - 0.5) / 6) < 1e-3
    pid = lc.PID(kp=5.0, ki=5.0, Ts=Ts, umin=-2, umax=2)
    _, u = _sim([1.0], [1, 2, 1], pid, 1.0, 3000, Ts)
    assert u.max() <= 2 + 1e-12


def test_antiwindup_reduces_overshoot():
    Ts = 0.005
    plant = ([1.0], [1, 1, 0])                         # integrating plant, easy to wind up
    res = {}
    for aw in ("none", "clamp", "backcalc"):
        pid = lc.PID(kp=4.0, ki=8.0, Ts=Ts, umin=-1.0, umax=1.0, antiwindup=aw, Tt=0.3)
        y, u = _sim(*plant, pid, 3.0, 3000, Ts)
        res[aw] = y.max() - 3.0
    assert res["clamp"] < res["none"] and res["backcalc"] < res["none"]


def test_pid_derivative_filter_and_weights():
    pid = lc.PID(kp=1.0, kd=0.1, Ts=0.001, N=10.0)
    # step in measurement: derivative kick is limited to roughly kp*N = 10 * step by the filter
    pid(0.0, 0.0)
    u = pid(0.0, 1.0)
    assert abs(u - (-1.0 + pid.D)) < 1e-12 and abs(pid.D) <= 10.5
    # set-point weight b = 0: no proportional kick on a reference step
    pid = lc.PID(kp=2.0, ki=0.0, Ts=0.01, b=0.0)
    assert pid(1.0, 0.0) == 0.0


# ---------------------------------------------------------------- discretisation
def test_c2d_zoh_matches_exact_sampling():
    G = TF([2], [1, 3, 2])
    Ts = 0.05
    numz, denz = lc.c2d_tf(G, Ts)
    n = 200
    y = lc.dlsim_tf(numz, denz, np.ones(n))
    t = np.arange(n) * Ts
    assert np.allclose(y, G.step(t), atol=1e-6)
    # Tustin preserves the DC gain
    nz, dz = lc.c2d_tf(G, Ts, "bilinear")
    assert np.isclose(np.polyval(nz, 1) / np.polyval(dz, 1), G.dcgain())


# ---------------------------------------------------------------- state space
def test_controllability_observability():
    A = np.array([[0, 1.0], [-2, -3]])
    B = np.array([[0], [1.0]])
    assert lc.rank_controllable(A, B) == 2
    assert lc.rank_controllable(A, np.array([[1.0], [-1.0]])) == 1          # eigenvector direction -> uncontrollable
    C = np.array([[1.0, 0]])
    assert lc.rank_observable(A, C) == 2
    assert lc.rank_observable(A, np.array([[1.0, 1.0]])) == 1


def test_acker_matches_place_and_poles():
    A = np.array([[0, 1, 0], [0, -2, 5.0], [0, -3, -10.0]])
    B = np.array([[0], [0], [100.0]])
    want = [-4, -5 + 3j, -5 - 3j]
    K1, K2 = lc.acker(A, B, want), lc.place(A, B, want)
    assert np.allclose(K1, K2, rtol=1e-6, atol=1e-6)
    ev = np.sort_complex(np.linalg.eigvals(A - B @ K1))
    assert np.allclose(ev, np.sort_complex(np.array(want)))


def test_observer_gain_and_separation():
    A = np.array([[0, 1.0], [-2, -3]])
    B = np.array([[0], [1.0]])
    C = np.array([[1.0, 0]])
    L = lc.observer_gain(A, C, [-10, -12])
    assert np.allclose(np.sort(np.linalg.eigvals(A - L @ C).real), [-12, -10])
    K = lc.acker(A, B, [-3, -4])
    Acl = np.block([[A, -B @ K], [L @ C, A - B @ K - L @ C]])               # plant + observer-based controller
    ev = np.sort(np.linalg.eigvals(Acl).real)
    assert np.allclose(ev, np.sort([-3, -4, -10, -12]), atol=1e-6)         # separation principle


def test_lqr_satisfies_riccati_and_beats_perturbed_gain():
    A = np.array([[0, 1.0], [0, -1.0]])
    B = np.array([[0], [1.0]])
    Q, R = np.diag([10.0, 1.0]), np.array([[0.5]])
    K, P, poles = lc.lqr(A, B, Q, R)
    res = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
    assert np.allclose(res, 0, atol=1e-9)
    assert np.all(poles.real < 0)
    # cost of u = -Kx from x0 equals x0' P x0; a perturbed gain costs more
    x0 = np.array([1.0, 0.0])

    def cost(Kg):
        Acl = A - B @ Kg
        Qc = Q + Kg.T @ R @ Kg
        X = linalg.solve_continuous_lyapunov(Acl.T, -Qc)                  # Acl' X + X Acl = -Qc
        return x0 @ X @ x0

    assert np.isclose(cost(K), x0 @ P @ x0, rtol=1e-7)
    assert cost(K * 1.3) > cost(K) and cost(K * 0.7) > cost(K)


def test_discrete_lqr_matches_continuous_for_small_ts():
    A = np.array([[0, 1.0], [-1, -0.5]])
    B = np.array([[0], [1.0]])
    Q, R = np.eye(2), np.array([[1.0]])
    K, *_ = lc.lqr(A, B, Q, R)
    Ts = 1e-3
    Ad, Bd, _, _ = lc.c2d_ss(A, B, np.eye(2), np.zeros((2, 1)), Ts)
    Kd, *_ = lc.lqr_discrete(Ad, Bd, Q * Ts, R * Ts)
    assert np.allclose(K, Kd, atol=2e-2)


def test_canonical_form_and_transfer_function_roundtrip():
    num, den = [3, 5], [1, 4, 6, 1]
    A, B, C, D = lc.controllable_canonical(num, den)
    G = lc.ss_to_tf(A, B, C, D)
    w = 1.3j
    assert np.isclose(G(w), np.polyval(num, w) / np.polyval(den, w))
    assert np.allclose(np.sort(np.linalg.eigvals(A)), np.sort(np.roots(den)))
    # feedthrough
    A, B, C, D = lc.controllable_canonical([2, 3, 1], [1, 3, 2])
    assert np.isclose(D.item(), 2.0)
    assert np.isclose((C @ np.linalg.solve(1j * np.eye(2) - A, B) + D).item(),
                      np.polyval([2, 3, 1], 1j) / np.polyval([1, 3, 2], 1j))


def test_prefilter_gain_gives_unit_dc_gain():
    A = np.array([[0, 1.0], [-2, -3]])
    B = np.array([[0], [1.0]])
    C = np.array([[1.0, 0]])
    K = lc.acker(A, B, [-5, -6])
    Nbar = lc.prefilter_gain(A, B, C, K)
    Acl, Bcl = A - B @ K, B * Nbar
    assert np.isclose((C @ np.linalg.solve(-Acl, Bcl)).item(), 1.0)


def test_linearize_pendulum():
    g, l = 9.81, 0.5
    f = lambda x, u: np.array([x[1], -g / l * np.sin(x[0]) + u[0]])
    A, B = lc.linearize(f, [0.0, 0.0], [0.0])
    assert np.allclose(A, [[0, 1], [-g / l, 0]], atol=1e-6) and np.allclose(B, [[0], [1]], atol=1e-6)
    A, _ = lc.linearize(f, [np.pi, 0.0], [0.0])
    assert np.allclose(A, [[0, 1], [g / l, 0]], atol=1e-6)


# ---------------------------------------------------------------- DC motor
def test_dc_motor_tf_matches_state_space():
    p = dict(Kt=0.03, Ke=0.03, R=1.2, L=6e-4, J=0.08, b=0.01, N=50.0)
    A, B, C, D = lc.dc_motor_ss(**p)
    G_ss = lc.ss_to_tf(A, B, C, D)
    G = lc.dc_motor_tf(**p)
    for w in (0.1, 3.0, 40.0, 900.0):
        assert np.isclose(G(1j * w), G_ss(1j * w), rtol=1e-6)
    # speed transfer function has dc gain N Kt / (R b + N^2 Kt Ke)
    Gw = lc.dc_motor_tf(**p, output="speed")
    assert np.isclose(Gw.dcgain(), p["N"] * p["Kt"] / (p["R"] * p["b"] + p["N"] ** 2 * p["Kt"] * p["Ke"]))
    # gravity stiffness moves the integrator pole off the origin
    A, *_ = lc.dc_motor_ss(**p, kg=2.0)
    assert np.all(np.linalg.eigvals(A).real < 0)
    A, *_ = lc.dc_motor_ss(**p, kg=-2.0)
    assert np.any(np.linalg.eigvals(A).real > 0)

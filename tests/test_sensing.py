import numpy as np
import pytest

from rgkit import sensing as sn


# ---------------------------------------------------------------------------
# Quantisation and encoders
# ---------------------------------------------------------------------------
def test_quantize_error_bounds_and_variance():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 200_000)
    q = 0.05
    e_floor = sn.quantize(x, q, "floor") - x
    e_round = sn.quantize(x, q, "round") - x
    assert e_floor.max() <= 1e-12 and e_floor.min() > -q - 1e-12
    assert abs(e_floor.mean() + q / 2) < 2e-4                       # floor quantiser is biased by -q/2
    assert abs(e_round.mean()) < 2e-4
    assert abs(e_round.std() - q / np.sqrt(12)) < 2e-4              # the classical q/sqrt(12)
    with pytest.raises(ValueError):
        sn.quantize(x, q, "ceil")


def test_encoder_resolution_and_counts():
    enc = sn.IncrementalEncoder(lines=1024, decoding=4)
    assert enc.counts_per_rev == 4096
    assert np.isclose(enc.resolution, 2 * np.pi / 4096)
    assert enc.counts(np.pi / 2) == 1024
    assert enc.counts(-1e-9) == -1                                   # floor, not truncation
    assert np.isclose(enc.angle(enc.counts(1.0)), 1.0 - (1.0 % enc.resolution))


def test_quadrature_decode_matches_floor_forward_and_back():
    lines = 40
    theta = np.concatenate([np.linspace(0, 3.0, 4000), np.linspace(3.0, 0.2, 4000)])
    A, B = sn.quadrature_signals(theta, lines)
    counts = sn.quadrature_decode(A, B)
    res = 2 * np.pi / (4 * lines)
    assert np.array_equal(counts, np.floor(theta / res).astype(int))
    assert counts[4000 - 1] == int(np.floor(3.0 / res))


def test_quadrature_rejects_chatter_but_single_channel_counting_drifts():
    lines = 100
    res = 2 * np.pi / (4 * lines)
    edge = 37 * res                                                   # sit exactly on a count boundary
    theta = edge + 0.3 * res * np.sin(np.linspace(0, 40 * np.pi, 2000))
    A, B = sn.quadrature_signals(theta, lines)
    counts = sn.quadrature_decode(A, B)
    assert counts.max() - counts.min() <= 1                           # the count flickers by one, never drifts
    naive = np.cumsum(np.diff(A, prepend=A[0]) == 1)                  # count rising edges of A only
    assert naive.max() > 5                                            # grows with every vibration cycle


def test_quadrature_sequence_forward_is_gray_cycle():
    A, B = sn.quadrature_signals(np.array([0.1, 0.35, 0.6, 0.85]) * 2 * np.pi / 10, 10)
    assert list(zip(A, B)) == [(0, 0), (1, 0), (1, 1), (0, 1)]


def test_gray_code_properties():
    n = np.arange(256)
    g = sn.gray_encode(n)
    assert np.array_equal(sn.gray_decode(g), n)
    flips = np.array([bin(int(a) ^ int(b)).count("1") for a, b in zip(g[:-1], g[1:])])
    assert np.all(flips == 1)
    assert sn.gray_encode(7) == 4 and sn.gray_encode(8) == 12


def test_absolute_encoder_gray_is_safe_binary_is_not():
    bits = 6
    q = 2 * np.pi / 2**bits
    theta = np.linspace(0, 2 * np.pi, 20_000, endpoint=False)
    truth = np.floor(theta / q).astype(int)
    for code in ("binary", "gray"):
        assert np.array_equal(sn.read_absolute(theta, bits, code), truth)
    off = np.random.default_rng(3).uniform(-0.45, 0.45, bits)
    def circ(e):
        e = np.abs(e) % 2**bits
        return np.minimum(e, 2**bits - e)
    e_gray = circ(sn.read_absolute(theta, bits, "gray", off) - truth)
    e_bin = circ(sn.read_absolute(theta, bits, "binary", off) - truth)
    assert e_gray.max() <= 1
    assert e_bin.max() >= 2 ** (bits - 2)


# ---------------------------------------------------------------------------
# Velocity estimation
# ---------------------------------------------------------------------------
def _constant_speed_setup(omega, res=2 * np.pi / 4096, Ts=1e-3, T=0.5, dt=1e-6):
    t = np.arange(0, T, dt)
    theta = omega * t + 0.3 * res
    ts = np.arange(0, T, Ts)
    return t, theta, ts, res


def test_velocity_M_quantum():
    omega = 3.3
    t, theta, ts, res = _constant_speed_setup(omega)
    counts = np.floor((omega * ts + 0.3 * res) / res)
    v = sn.velocity_M(counts, res, 1e-3)
    assert set(np.round(np.unique(v[1:]) / (res / 1e-3))) <= {3.0, 2.0, 4.0}
    assert np.abs(v[1:] - omega).max() <= res / 1e-3 + 1e-9


def test_velocity_T_and_MT_exact_at_constant_speed_with_ideal_clock():
    omega = 4.0
    t, theta, ts, res = _constant_speed_setup(omega)
    edges = sn.edge_times(t, theta, res)
    assert np.allclose(np.diff(edges), res / omega, rtol=1e-4)
    vT = sn.velocity_T(edges, ts, res)
    vMT = sn.velocity_MT(edges, ts, res)
    assert np.allclose(vT[5:], omega, rtol=1e-3)
    assert np.allclose(vMT[5:], omega, rtol=1e-3)


def test_velocity_T_degrades_with_a_slow_clock_but_MT_is_better_than_M_at_low_speed():
    omega = 0.5                                                     # rad/s: a few pulses per sample
    t, theta, ts, res = _constant_speed_setup(omega, T=1.0)
    edges = sn.edge_times(t, theta, res)
    counts = np.floor((omega * ts + 0.3 * res) / res)
    vM = sn.velocity_M(counts, res, 1e-3)
    vMT = sn.velocity_MT(edges, ts, res, f_clk=1e6)
    err_M = np.abs(vM[100:] - omega).mean()
    err_MT = np.abs(vMT[100:] - omega).mean()
    assert err_MT < 0.05 * err_M


def test_velocity_MT_stopped_shaft_decays_to_zero_estimate():
    t = np.arange(0, 0.2, 1e-5)
    theta = np.where(t < 0.05, 5.0 * t, 0.25)
    res = 2 * np.pi / 4096
    edges = sn.edge_times(t, theta, res)
    ts = np.arange(0, 0.2, 1e-3)
    v = sn.velocity_MT(edges, ts, res)
    assert v[40] > 0.5 * 5.0 and v[-1] < 0.02


def test_lowpass_derivative_of_ramp_settles_to_slope():
    Ts, tau = 1e-3, 0.02
    t = np.arange(0, 0.3, Ts)
    y = sn.lowpass_derivative(3.0 * t, Ts, tau)
    assert abs(y[-1] - 3.0) < 1e-4
    assert abs(y[int(tau / Ts)] - 3.0 * (1 - np.exp(-1))) < 0.05


def test_alpha_beta_poles_and_ramp_tracking():
    Ts, om = 1e-3, 100.0
    a, b = sn.alpha_beta_gains(om, Ts)
    roots = np.roots([1.0, a + b - 2.0, 1.0 - a])
    assert np.allclose(roots, np.exp(-om * Ts), atol=1e-6)
    t = np.arange(0, 0.5, Ts)
    x, v = sn.alpha_beta_filter(2.0 + 5.0 * t, Ts, a, b)
    assert abs(v[-1] - 5.0) < 1e-6 and abs(x[-1] - (2.0 + 5.0 * t[-1])) < 1e-6   # no steady-state error on a ramp


def test_kalman_cv_gains_match_alpha_beta_filter_in_steady_state():
    Ts, q, r = 1e-3, 50.0, 1e-6
    F, H, Q, R = sn.cv_kalman_model(Ts, q, r)
    K, P = sn.cv_kalman_gains(Ts, q, r)
    a, b = K[0], K[1] * Ts
    rng = np.random.default_rng(1)
    z = 0.4 * np.arange(3000) * Ts + 1e-3 * rng.standard_normal(3000)
    xs, _ = sn.kalman_filter(F, H, Q, R, z, x0=[z[0], 0.0], P0=np.diag([1e-3, 1.0]))
    xa, va = sn.alpha_beta_filter(z, Ts, a, b, x0=xs[0, 0], v0=xs[0, 1])
    assert np.allclose(xs[-500:, 0], xa[-500:], atol=1e-7)
    assert np.allclose(xs[-500:, 1], va[-500:], atol=1e-4)
    assert 0 < a < 1 and 0 < b < 1


# ---------------------------------------------------------------------------
# Strain gauges and wrench sensors
# ---------------------------------------------------------------------------
def test_bridge_balanced_and_quarter_bridge():
    assert sn.bridge_output(5.0, 350, 350, 350, 350) == 0.0
    eps = 1e-3
    exact = sn.quarter_bridge(5.0, eps, gauge_factor=2.0)
    assert np.isclose(exact, 5.0 * (350 / (350 + 350 * 1.002) - 0.5))
    assert abs(exact - (-2.0 * eps * 5.0 / 4)) < 5e-6                 # first-order formula (nonlinearity ~ GF*eps/2)
    big = sn.quarter_bridge(5.0, 0.01, 2.0)
    assert abs(big / (-2.0 * 0.01 * 5.0 / 4) - 1) > 0.005             # visibly non-linear at 1 % strain


def test_calibration_recovers_matrix_and_noise_averages_down():
    rng = np.random.default_rng(2)
    C_true = np.diag([1.0, 1.0, 2.0, 0.05, 0.05, 0.04]) + 0.02 * rng.standard_normal((6, 6))
    w0 = rng.standard_normal(6)
    def experiment(n, sigma):
        W = rng.uniform(-1, 1, (n, 6)) * np.array([50, 50, 100, 5, 5, 5])
        V = (W - w0) @ np.linalg.inv(C_true).T + sigma * rng.standard_normal((n, 6))
        return sn.calibrate_wrench(V, W)
    C, off = experiment(60, 0.0)
    assert np.allclose(C, C_true, atol=1e-9) and np.allclose(off, w0, atol=1e-9)
    err = lambda n: np.linalg.norm(experiment(n, 1e-3)[0] - C_true)
    errs_small = np.mean([err(10) for _ in range(30)])
    errs_big = np.mean([err(200) for _ in range(30)])
    assert errs_big < 0.5 * errs_small
    W_hat = sn.apply_calibration(*experiment(60, 0.0), np.zeros((1, 6)))
    assert W_hat.shape == (1, 6)


def test_estimate_payload_recovers_mass_com_and_offsets():
    rng = np.random.default_rng(5)
    m, c = 0.987, np.array([0.01, -0.02, 0.07])
    f0, t0 = np.array([0.3, -0.2, 0.5]), np.array([0.01, 0.02, -0.01])
    g = np.array([0.0, 0.0, -9.81])
    from rgkit.transforms import rot_axis_angle
    Rs = [rot_axis_angle(rng.standard_normal(3), rng.uniform(0, np.pi)) for _ in range(6)]
    skew = lambda w: np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    F = [-m * R.T @ g + f0 for R in Rs]
    T = [np.cross(c, -m * R.T @ g) + t0 for R in Rs]
    m_h, c_h, f_h, t_h = sn.estimate_payload(Rs, F, T, g)
    assert np.isclose(m_h, m) and np.allclose(c_h, c) and np.allclose(f_h, f0) and np.allclose(t_h, t0)


# ---------------------------------------------------------------------------
# Inertial sensing
# ---------------------------------------------------------------------------
def test_gyro_noise_and_bias_statistics():
    rng = np.random.default_rng(7)
    Ts, arw = 1e-2, 2e-3
    y, b = sn.simulate_gyro(np.zeros(400_000), Ts, rng, arw=arw, bias0=0.01)
    assert abs(y.std() - arw / np.sqrt(Ts)) < 2e-4
    assert np.allclose(b, 0.01)
    y2, b2 = sn.simulate_gyro(np.ones(1000), Ts, rng, bias_walk=1e-3, scale_error=0.02)
    assert abs(y2[0] - 1.02) < 1e-12 and b2[-1] != b2[0]


def test_tilt_from_accel():
    th = np.deg2rad(25)
    assert np.isclose(sn.tilt_from_accel(9.81 * np.sin(th), 9.81 * np.cos(th)), th)


def test_complementary_filter_bias_error_is_bias_times_tau():
    Ts, tau, b = 0.005, 0.8, 0.02
    n = 6000
    theta_true = np.full(n, 0.3)
    gyro = np.full(n, b)                                            # stationary body, gyro reads only its bias
    est = sn.complementary_filter(gyro, theta_true, Ts, tau)
    assert abs(est[-1] - 0.3 - b * tau) < 1e-6


def test_kalman_angle_bias_estimates_bias_and_beats_complementary_rmse():
    rng = np.random.default_rng(11)
    Ts, n = 0.01, 6000
    t = np.arange(n) * Ts
    theta = 0.5 * np.sin(2 * np.pi * 0.2 * t)
    omega = np.gradient(theta, Ts)
    arw, bw, bias0, sig_acc = 2e-3, 1e-4, 0.03, 0.03
    gyro, bias = sn.simulate_gyro(omega, Ts, rng, arw=arw, bias0=bias0, bias_walk=bw)
    acc = theta + sig_acc * rng.standard_normal(n)
    th_k, b_k = sn.kalman_angle_bias(gyro, acc, Ts, arw, bw, sig_acc**2)
    th_c = sn.complementary_filter(gyro, acc, Ts, tau=1.0)
    assert abs(b_k[-1] - bias[-1]) < 0.01
    rmse = lambda e: np.sqrt(np.mean(e[1000:] ** 2))
    assert rmse(th_k - theta) < rmse(th_c - theta)
    assert rmse(th_k - theta) < sig_acc / 3


def test_allan_deviation_white_noise_slope_and_level():
    rng = np.random.default_rng(13)
    Ts, N = 0.01, 2e-3                                              # N in rad/s/sqrt(Hz)
    x = N / np.sqrt(Ts) * rng.standard_normal(200_000)
    taus, adev = sn.allan_deviation(x, Ts)
    sel = (taus > 0.05) & (taus < 5)
    assert np.allclose(adev[sel], N / np.sqrt(taus[sel]), rtol=0.15)
    slope = np.polyfit(np.log(taus[sel]), np.log(adev[sel]), 1)[0]
    assert abs(slope + 0.5) < 0.05

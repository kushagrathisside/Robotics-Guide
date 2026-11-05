import numpy as np
import pytest

from rgkit import teleop as tp

M = tp.Body1D(1.0, 2.0)
S = tp.Body1D(1.5, 1.0)


def ramp(t, t0=0.5, t1=2.0, X=0.4):
    u = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return X * (0.5 - 0.5 * np.cos(np.pi * u))


# ---------------------------------------------------------------------------------------------------------
# frequency domain
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_p,n_f", [(1.0, 1.0), (0.2, 5.0), (0.1, 3.0)])
def test_ideal_four_channel_gives_ideal_hybrid_matrix(n_p, n_f):
    arch = tp.FourChannel.ideal(M, S, cm=(100.0, 20.0, 0.0), cs=(50.0, 10.0, 0.0), n_p=n_p, n_f=n_f)
    w = np.logspace(-1, 3, 25)
    H = tp.hybrid_matrix(M, S, arch, w)
    target = np.array([[0.0, n_f], [-n_p, 0.0]])
    assert np.allclose(H, target, atol=1e-9)
    Ze = 300.0 / (1j * w) + 4.0
    assert np.allclose(tp.transmitted_impedance(H, Ze), n_p * n_f * Ze)      # impedance scaling n_p n_f


def test_ideal_conditions_with_local_force_feedback():
    arch = tp.FourChannel.ideal(M, S, cm=(100.0, 20.0, 0.0), cs=(50.0, 10.0, 0.0), c5=0.3, c6=-0.4)
    H = tp.hybrid_matrix(M, S, arch, np.logspace(-1, 3, 9))
    assert np.allclose(H, tp.IDEAL_H, atol=1e-9)


def test_position_position_static_stiffness_is_series_spring():
    kp, ke = 200.0, 2000.0
    arch = tp.FourChannel.position_position(kp, 20.0)
    w = np.array([1e-3])
    H = tp.hybrid_matrix(M, S, arch, w)
    Zt = tp.transmitted_impedance(H, ke / (1j * w))
    assert np.isclose((Zt * 1j * w).real[0], kp * ke / (kp + ke), rtol=1e-4)


def test_llewellyn_and_passivity_of_standard_architectures():
    w = np.logspace(-1, 3, 400)
    pp = tp.hybrid_matrix(M, S, tp.FourChannel.position_position(200.0, 20.0), w)
    L = tp.llewellyn(pp)
    assert L["r11"].min() >= -1e-9 and L["r22"].min() >= -1e-9 and L["c"].min() >= -1e-9
    assert tp.passivity_margin(pp).min() >= -1e-9                          # P-P without delay is passive
    ideal = tp.hybrid_matrix(M, S, tp.FourChannel.ideal(M, S, cm=(100, 20, 0), cs=(100, 20, 0)), w)
    assert np.allclose(tp.llewellyn(ideal)["c"], 0.0, atol=1e-9)            # lossless gyrator: marginal
    assert np.allclose(tp.passivity_margin(ideal), 0.0, atol=1e-9)
    uni = tp.hybrid_matrix(M, S, tp.FourChannel.unilateral(200.0, 20.0), w)
    assert tp.llewellyn(uni)["c"].min() >= -1e-9                           # absolutely stable ...
    assert tp.passivity_margin(uni).min() < -1e-3                          # ... but not passive


def test_delay_breaks_passivity_of_position_position():
    w = np.logspace(-1, 3, 400)
    arch = tp.FourChannel.position_position(200.0, 20.0)
    assert tp.passivity_margin(tp.hybrid_matrix(M, S, arch, w, delay=(0.03, 0.03))).min() > -1e-9
    H = tp.hybrid_matrix(M, S, arch, w, delay=(0.3, 0.3))
    assert tp.passivity_margin(H).min() < -1.0


# ---------------------------------------------------------------------------------------------------------
# characteristic equation and time domain
# ---------------------------------------------------------------------------------------------------------
def test_critical_delay_matches_poles_and_simulation():
    arch = tp.FourChannel.position_position(400.0, 10.0)
    hand = (50.0, 5.0)
    Tc, wc = tp.critical_delay(M, S, arch, hand=hand)
    assert np.isfinite(Tc) and Tc > 0
    assert tp.loop_poles(M, S, arch, 0.9 * Tc, hand).real.max() < 0
    assert tp.loop_poles(M, S, arch, 1.1 * Tc, hand).real.max() > 0
    assert abs(tp.loop_poles(M, S, arch, Tc, hand, order=8).real.max()) < 1e-3
    op = tp.Operator(50.0, 5.0, lambda t: 0.1 * (t < 0.5))
    env = tp.Environment(0.0, 0.0, None)
    ok = tp.simulate(M, S, op, env, arch, t_end=12.0, delay=(0.45 * Tc, 0.45 * Tc))
    bad = tp.simulate(M, S, op, env, arch, t_end=12.0, delay=(0.75 * Tc, 0.75 * Tc))
    assert np.abs(ok.xs[-1000:]).max() < 1e-3
    assert np.abs(bad.xs[-1000:]).max() > 1.0


def test_position_force_critical_delay_agrees_with_simulation():
    S = tp.Body1D(1.0, 2.0)
    M = S
    arch = tp.FourChannel.position_force(400.0, 20.0, kf=1.0)
    hand, env_k = (50.0, 10.0), (1000.0, 5.0)
    Tc, _ = tp.critical_delay(M, S, arch, hand=hand, env=env_k)
    assert 0.03 < Tc < 0.04
    op = tp.Operator(50.0, 10.0, lambda t: 0.01 * (t < 0.5))
    env = tp.Environment(1000.0, 5.0, None)
    stable = tp.simulate(M, S, op, env, arch, t_end=10.0, delay=(0.4 * Tc, 0.4 * Tc))
    unstable = tp.simulate(M, S, op, env, arch, t_end=10.0, delay=(0.7 * Tc, 0.7 * Tc))
    assert np.abs(stable.xm[-1000:]).max() < 1e-3
    assert np.abs(unstable.xm[-1000:]).max() > 0.1


def test_static_contact_forces_and_felt_stiffness():
    M = S = tp.Body1D(1.0, 2.0)
    kp, ke, x0 = 400.0, 1000.0, 0.05
    op = tp.Operator(50.0, 10.0, lambda t: ramp(t, X=0.4))
    env = tp.Environment(ke, 5.0, x0)
    for arch, kt in [(tp.FourChannel.position_position(kp, 20.0), kp * ke / (kp + ke)),
                     (tp.FourChannel.position_force(kp, 20.0), kp * ke / (kp + ke)),
                     (tp.FourChannel.blend(M, S, kp, 20.0, 1.0, inertia=False), ke)]:
        r = tp.simulate(M, S, op, env, arch, t_end=5.0)
        k = r.t > 4.5
        assert np.isclose(r.fh[k].mean(), r.fe[k].mean(), rtol=2e-3)                      # force balance
        felt = r.fh[k].mean() / (r.xm[k].mean() - x0)
        assert np.isclose(felt, kt, rtol=2e-2), (arch.name, felt, kt)


def test_unilateral_gives_no_force_feedback():
    op = tp.Operator(50.0, 10.0, lambda t: ramp(t, X=0.4))
    r = tp.simulate(M, S, op, tp.Environment(1000.0, 5.0, 0.05), tp.FourChannel.unilateral(400.0, 20.0), t_end=4.0)
    assert np.abs(r.taum).max() == 0.0
    assert r.fe[-1] > 5.0


def test_blend_interpolates_and_ideal_with_inertia_is_fragile():
    lam0 = tp.FourChannel.blend(M, S, 400.0, 20.0, 0.0)
    pp = tp.FourChannel.position_position(400.0, 20.0)
    assert lam0.c1 == pp.c1 and lam0.c4 == pp.c4 and lam0.c2 == 0.0
    w = np.logspace(-1, 3, 10)
    H1 = tp.hybrid_matrix(M, S, tp.FourChannel.blend(M, S, 400.0, 20.0, 1.0), w)
    assert np.allclose(H1, tp.IDEAL_H, atol=1e-9)
    # with a one-sample digital delay the perfectly transparent design is unstable on a stiff wall
    op = tp.Operator(50.0, 10.0, lambda t: ramp(t, X=0.4))
    env = tp.Environment(1000.0, 5.0, 0.05)
    r = tp.simulate(M, S, op, env, tp.FourChannel.blend(M, S, 400.0, 20.0, 1.0), t_end=4.0, dt=5e-4)
    assert not np.isfinite(r.xm[-1]) or np.abs(r.xm).max() > 10.0
    r = tp.simulate(M, S, op, env, tp.FourChannel.blend(M, S, 400.0, 20.0, 0.5), t_end=4.0, dt=5e-4)
    assert np.abs(r.xm).max() < 1.0


# ---------------------------------------------------------------------------------------------------------
# wave variables and passivity of delayed channels
# ---------------------------------------------------------------------------------------------------------
def test_wave_transform_inverse_and_power_identity():
    rng = np.random.default_rng(1)
    v, F, b = rng.normal(size=50), rng.normal(size=50), 7.3
    u, w = tp.wave_encode(v, F, b)
    v2, F2 = tp.wave_decode(u, w, b)
    assert np.allclose(v, v2) and np.allclose(F, F2)
    assert np.allclose(F * v, 0.5 * (u ** 2 - w ** 2))


def test_direct_delay_is_not_passive_but_wave_channel_is():
    n, N, dt = 4000, 25, 1e-3
    t = np.arange(n) * dt
    omega = np.pi / 2 / (N * dt)                      # omega T = pi/2
    v, F = np.sin(omega * t), np.cos(omega * t)
    FL, vR = tp.simulate_channel("direct", N, v, F)
    E = tp.port_energy(FL, v, F, vR, dt)
    assert E[-1] < -1.0                                # energy is generated at a constant rate
    FL, vR = tp.simulate_channel("wave", N, v, F, b=5.0)
    E = tp.port_energy(FL, v, F, vR, dt)
    assert E.min() >= -1e-12


def test_wave_channel_energy_equals_stored_waves_for_random_inputs():
    rng = np.random.default_rng(3)
    n, N, b = 600, 17, 3.0
    v, F = rng.normal(size=n), rng.normal(size=n)
    FL, vR = tp.simulate_channel("wave", N, v, F, b=b)
    E = tp.port_energy(FL, v, F, vR)
    # stored energy = 1/2 sum of squared incoming waves over the last N samples
    uL, _ = tp.wave_encode(v, FL, b)
    uR, _ = tp.wave_encode(vR, F, b)
    stored = 0.5 * np.array([(uL[max(0, k - N + 1):k + 1] ** 2).sum() + (uR[max(0, k - N + 1):k + 1] ** 2).sum()
                             for k in range(n)])
    assert np.allclose(E, stored)
    assert E.min() >= 0.0


def test_time_varying_delay_scales_the_energy_of_a_burst():
    n = 6000
    k = np.arange(n)
    burst = np.where(k < 800, np.sin(0.2 * k), 0.0)
    E_in = (burst ** 2).sum()
    for rate, ratio in [(0.0, 1.0), (0.25, 1 / 0.75), (-0.25, 1 / 1.25)]:
        N = (40 + np.floor(rate * np.minimum(k, 1500))).astype(int) if rate >= 0 else (400 + np.floor(rate * np.minimum(k, 1500))).astype(int)
        _, v_out = tp.simulate_channel("direct", N, burst, np.zeros(n))
        assert np.isclose((v_out ** 2).sum() / E_in, ratio, rtol=0.05), (rate, (v_out ** 2).sum() / E_in, ratio)


def test_wave_teleoperator_is_stable_under_large_delay_and_drifts():
    op = tp.Operator(50.0, 10.0, lambda t: ramp(t) if t < 5 else ramp(t) - 0.4 * np.clip((t - 5) / 1.5, 0, 1))
    env = tp.Environment(1000.0, 5.0, 0.05)
    M = S = tp.Body1D(1.0, 2.0)
    arch = tp.WaveVariable(b=20.0, kv=300.0)
    r = tp.simulate(M, S, op, env, arch, t_end=9.0, delay=(0.15, 0.15), dt=5e-4)
    assert np.abs(r.xm).max() < 1.0 and np.abs(r.xs).max() < 1.0
    assert abs(r.xm[-1] - r.xs[-1]) > 0.05                   # position drift after contact
    # free motion: the slave follows the master (with the delay) closely
    r0 = tp.simulate(M, S, tp.Operator(50.0, 10.0, lambda t: 0.1 * np.sin(2 * t)), tp.Environment(0, 0, None), arch,
                     t_end=6.0, dt=5e-4, delay=(0.05, 0.05))
    assert np.abs(r0.xs - r0.xm)[3000:].max() < 0.1 * np.abs(r0.xm).max()


# ---------------------------------------------------------------------------------------------------------
# haptics and fixtures
# ---------------------------------------------------------------------------------------------------------
def test_sampled_wall_stability_threshold():
    m, b, Ts = 0.5, 2.0, 0.005
    Kc = tp.sampled_wall_critical_K(m, b, 0.0, Ts)
    assert tp.sampled_wall_radius(m, b, 0.9 * Kc, 0.0, Ts) < 1.0 < tp.sampled_wall_radius(m, b, 1.1 * Kc, 0.0, Ts)
    assert Kc > 2 * b / Ts * 0.8                       # at least comparable with the passivity bound 2 b / T
    assert tp.sampled_wall_critical_K(m, b, 0.0, Ts / 2) > Kc    # faster sampling allows a stiffer wall


def test_closest_on_polyline_and_guidance():
    P = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    q, t_hat, d, s = tp.closest_on_polyline(P, [0.5, 0.2])
    assert np.allclose(q, [0.5, 0.0]) and np.allclose(t_hat, [1, 0]) and np.isclose(d, 0.2) and np.isclose(s, 0.5)
    q, t_hat, d, s = tp.closest_on_polyline(P, [1.3, 0.5])
    assert np.allclose(q, [1.0, 0.5]) and np.allclose(t_hat, [0, 1]) and np.isclose(d, -0.3) and np.isclose(s, 1.5)
    v = np.array([1.0, 2.0])
    assert np.allclose(tp.guidance_fixture(v, [1, 0], 0.0), [1.0, 0.0])
    assert np.allclose(tp.guidance_fixture(v, [1, 0], 1.0), v)
    assert np.allclose(tp.guidance_fixture(v, [2, 0], 0.25), [1.0, 0.5])

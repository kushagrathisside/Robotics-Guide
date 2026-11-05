import numpy as np

from rgkit import realtime as rt


def test_alias_frequency():
    assert np.isclose(rt.alias_frequency(30.0, 100.0), 30.0)
    assert np.isclose(rt.alias_frequency(70.0, 100.0), 30.0)
    assert np.isclose(rt.alias_frequency(130.0, 100.0), 30.0)
    assert np.isclose(rt.alias_frequency(100.0, 100.0), 0.0)
    # a 950 Hz vibration sampled at 1 kHz looks like 50 Hz
    assert np.isclose(rt.alias_frequency(950.0, 1000.0), 50.0)


def test_pd_double_integrator_margins_match_closed_form():
    J, Kp, Kd = 0.1, 360.0, 12.0                    # wn = 60 rad/s, zeta = 1
    L = rt.pd_joint_L(J, 0.0, Kp, Kd)
    wc, pm = rt.loop_margins(L)
    # |L(jwc)| = 1:  Kd^2 wc^2 + Kp^2 = J^2 wc^4  ->  wc^2 = (Kd^2 + sqrt(Kd^4 + 4 J^2 Kp^2)) / (2 J^2)
    wc_exact = np.sqrt((Kd**2 + np.sqrt(Kd**4 + 4 * J**2 * Kp**2)) / (2 * J**2))
    assert np.isclose(wc, wc_exact, rtol=1e-3)
    pm_exact = np.rad2deg(np.arctan2(Kd * wc_exact, Kp))          # phase margin = atan(Kd wc / Kp) for 1/s^2 plant
    assert np.isclose(pm, pm_exact, atol=0.05)
    Td = rt.delay_margin(L)
    assert np.isclose(Td, np.deg2rad(pm_exact) / wc_exact, rtol=1e-3)
    wc2, pm2 = rt.loop_margins(L, delay=0.5 * Td)
    assert pm2 < pm and np.isclose(np.deg2rad(pm - pm2) / wc, 0.5 * Td, rtol=0.05)


def test_delay_just_below_and_above_the_margin_decides_stability_in_simulation():
    J, Kp, Kd, Ts = 0.1, 360.0, 12.0, 1e-4
    Td = rt.delay_margin(rt.pd_joint_L(J, 0.0, Kp, Kd))
    Ac, Bc, K = [[0, 1], [0, 0]], [[0], [1 / J]], [[Kp, Kd]]
    stable = rt.simulate_delayed_loop(Ac, Bc, K, Ts, 3.0, [0.1, 0.0], delay=0.7 * Td)
    unstable = rt.simulate_delayed_loop(Ac, Bc, K, Ts, 3.0, [0.1, 0.0], delay=1.5 * Td)
    assert abs(stable[1][-1, 0]) < 1e-3
    assert abs(unstable[1][-200:, 0]).max() > 1.0                                    # growing oscillation


def test_delayed_loop_without_delay_matches_continuous_pd_for_small_ts():
    J, Kp, Kd = 0.1, 360.0, 12.0
    t, X, tk, U = rt.simulate_delayed_loop([[0, 1], [0, 0]], [[0], [1 / J]], [[Kp, Kd]], 1e-4, 0.2, [0.1, 0.0])
    # critically damped: q(t) = q0 (1 + wn t) exp(-wn t)
    wn = np.sqrt(Kp / J)
    exact = 0.1 * (1 + wn * t) * np.exp(-wn * t)
    assert np.abs(X[:, 0] - exact).max() < 2e-3
    assert len(tk) == 2000


def test_jitter_and_reference_and_disturbance_options_run():
    out = rt.simulate_delayed_loop([[0, 1], [0, 0]], [[0], [10.0]], [[3600.0, 120.0]], 1e-3, 0.5, [0, 0],
                                   delay=2e-4, jitter=5e-4, x_ref=[0.2, 0.0], dist=lambda t: 0.1, u_max=500.0)
    assert out[1].shape[1] == 2 and abs(out[1][-1, 0] - 0.2) < 0.02


def test_cascade_gains_and_step_responses():
    motor = dict(R=1.0, L=2e-3, Kt=0.05, Ke=0.05, J=2e-4, b=1e-5)
    wi, wv, wp = 2000.0, 400.0, 80.0
    g = rt.motor_cascade_gains(motor["R"], motor["L"], motor["Kt"], motor["J"], motor["b"], wi, wv, wp)
    assert np.isclose(g["Kp_i"], 4.0) and np.isclose(g["Ki_i"], 2000.0)
    assert np.isclose(g["Kp_v"], 400 * 2e-4 / 0.05) and np.isclose(g["Kp_p"], 80.0)
    out = rt.simulate_cascade(motor, g, lambda t: 1.0, 0.25, dt=1e-5, rates=(1, 4, 20))
    assert abs(out["theta"][-1] - 1.0) < 1e-3                                       # integrators give zero steady-state error
    # inner loop alone (lock the rotor to isolate the current loop): 63 % of the step after 1/wi
    locked = dict(motor, J=1e6, b=0.0)
    g2 = dict(g, Kp_v=1.0, Ki_v=0.0)
    cur = rt.simulate_cascade(locked, dict(g2, Kp_p=1e6), lambda t: 1e-6, 5e-3, dt=1e-6, I_max=1.0)
    # the set-point is saturated at I_max = 1 A, so i settles to 1 A with a first-order lag of 1/wi
    k63 = np.argmax(cur["i"] > 0.632)
    assert abs(cur["t"][k63] - 1 / wi) < 0.2 / wi


def test_cascade_respects_current_limit():
    motor = dict(R=1.0, L=2e-3, Kt=0.05, Ke=0.05, J=2e-4, b=1e-5)
    g = rt.motor_cascade_gains(motor["R"], motor["L"], motor["Kt"], motor["J"], motor["b"], 2000.0, 400.0, 80.0)
    out = rt.simulate_cascade(motor, g, lambda t: 5.0, 0.3, dt=1e-5, I_max=5.0, V_max=24.0, rates=(1, 4, 20))
    assert out["i"].max() < 5.3 and out["V"].max() <= 24.0 + 1e-12


def test_scheduling_analysis():
    assert np.isclose(rt.liu_layland_bound(1), 1.0) and np.isclose(rt.liu_layland_bound(3), 0.7798, atol=1e-4)
    assert np.isclose(rt.liu_layland_bound(1000), np.log(2), atol=1e-3)
    C, T = [1, 2, 3], [4, 6, 13]
    assert np.isclose(rt.utilisation(C, T), 1 / 4 + 2 / 6 + 3 / 13)
    assert np.allclose(rt.response_times(C, T), [1, 3, 10])
    R = rt.response_times([1, 2, 3], [4, 6, 8])
    assert np.isinf(R[2])                                                          # utilisation 0.958: misses its deadline
    assert np.allclose(rt.response_times([1, 2], [4, 8], blocking=[0.5, 0.0]), [1.5, 3.0])

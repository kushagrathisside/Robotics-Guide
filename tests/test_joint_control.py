"""Tests for rgkit.joint_control (RMC Chapter 2)."""
import numpy as np
import pytest

from rgkit import joint_control as jc


# ── gearing and second-order formulas ─────────────────────────────────────────────────────────────
def test_effective_inertia_and_damping():
    assert jc.effective_inertia(1e-4, 0.5, 50) == pytest.approx(1e-4 + 0.5 / 2500)
    assert jc.reflected_to_load(1e-4, 0.5, 50) == pytest.approx(0.5 + 2500 * 1e-4)
    # the two descriptions of the same joint differ by exactly r^2
    assert jc.reflected_to_load(1e-4, 0.5, 50) == pytest.approx(50**2 * jc.effective_inertia(1e-4, 0.5, 50))
    assert jc.effective_damping(1e-5, 0.2, 10, Kt=0.05, Kb=0.05, R=1.0) == pytest.approx(1e-5 + 0.002 + 0.0025)
    assert jc.effective_damping(1e-5, 0.2, 10) == pytest.approx(1e-5 + 0.002)


def test_pd_gain_roundtrip():
    J, B = 0.05, 0.4
    kp, kd = jc.pd_gains(J, B, wn=20.0, zeta=0.8)
    wn, z = jc.second_order_params(J, B, kp, kd)
    assert wn == pytest.approx(20.0) and z == pytest.approx(0.8)
    p = np.roots([J, B + kd, kp])
    assert np.allclose(sorted(p.real), sorted([-0.8 * 20] * 2))


def test_overshoot_and_times():
    assert jc.overshoot(0.0) == pytest.approx(100.0)
    assert jc.overshoot(0.5) == pytest.approx(16.303, abs=1e-3)
    assert jc.overshoot(1.0) == 0.0
    assert jc.peak_time(10.0, 0.5) == pytest.approx(np.pi / (10 * np.sqrt(0.75)))
    assert jc.settling_time(10.0, 0.5, 0.02) == pytest.approx(0.7824, abs=1e-3)


def test_pid_pole_placement_and_routh():
    J, B = 0.1, 0.3
    poles = [-10, -15 + 5j, -15 - 5j]
    kp, ki, kd = jc.pid_gains_from_poles(J, B, poles)
    assert np.allclose(sorted(jc.pid_poles(J, B, kp, ki, kd), key=lambda z: (z.real, z.imag)),
                       sorted(poles, key=lambda z: (z.real, z.imag)))
    # Routh: stable just below the limit, unstable just above
    kp, kd = 50.0, 5.0
    lim = jc.ki_limit(J, B, kp, kd)
    assert np.all(jc.pid_poles(J, B, kp, 0.99 * lim, kd).real < 0)
    assert np.any(jc.pid_poles(J, B, kp, 1.01 * lim, kd).real > 0)


def test_step_metrics_on_known_response():
    wn, z = 10.0, 0.4
    t = np.linspace(0, 3, 6001)
    wd = wn * np.sqrt(1 - z**2)
    y = 1 - np.exp(-z * wn * t) * (np.cos(wd * t) + z * wn / wd * np.sin(wd * t))
    m = jc.step_metrics(t, y, y_final=1.0)
    assert m["overshoot"] == pytest.approx(jc.overshoot(z), abs=0.05)
    assert abs(m["final_error"]) < 1e-3
    assert 0.13 < m["rise"] < 0.17


# ── trajectories ─────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("fn", [jc.cubic, jc.quintic])
def test_rest_to_rest_boundary_conditions(fn):
    q, qd, qdd = fn(np.array([0.0, 2.0]), 0.3, 1.3, 2.0)
    assert np.allclose(q, [0.3, 1.3]) and np.allclose(qd, 0.0)
    if fn is jc.quintic:
        assert np.allclose(qdd, 0.0)
    # derivatives are consistent with finite differences
    t = np.linspace(0.1, 1.9, 7)
    h = 1e-6
    q1, v1, _ = fn(t + h, 0.3, 1.3, 2.0)
    q0, v0, _ = fn(t - h, 0.3, 1.3, 2.0)
    _, v, a = fn(t, 0.3, 1.3, 2.0)
    assert np.allclose((q1 - q0) / (2 * h), v, atol=1e-6) and np.allclose((v1 - v0) / (2 * h), a, atol=1e-5)


def test_quintic_peak_velocity():
    t = np.linspace(0, 1, 2001)
    _, qd, _ = jc.quintic(t, 0.0, 1.0, 1.0)
    assert qd.max() == pytest.approx(1.875, abs=1e-4)


# ── single joint, PID, saturation, windup ─────────────────────────────────────────────────────────
def _step_pd(J=0.05, B=0.4, wn=20.0, zeta=1.0, dt=1e-3, T=1.5, **kw):
    kp, kd = jc.pd_gains(J, B, wn, zeta)
    return jc.simulate(jc.Joint(J, B, **kw), jc.PID(kp, 0, kd, dt=dt), 1.0, T, dt)


def test_pd_step_matches_second_order_theory():
    res = _step_pd(zeta=0.5)
    m = jc.step_metrics(res.t, res.q, y_final=1.0, y0=0.0)
    assert m["overshoot"] == pytest.approx(jc.overshoot(0.5), abs=0.5)
    assert abs(res.q[-1] - 1.0) < 1e-3


def test_gravity_gives_steady_state_error_kp_cancels_with_integral():
    J, B, mgl = 0.05, 0.4, 2.0
    kp, kd = jc.pd_gains(J, B, 20.0, 1.0)
    q_d = 0.5
    res = jc.simulate(jc.Joint(J, B, mgl=mgl), jc.PID(kp, 0, kd, dt=1e-3), q_d, 3.0, 1e-3)
    # exact equilibrium of  kp (q_d - q) = mgl cos q
    from scipy.optimize import brentq
    q_eq = brentq(lambda q: kp * (q_d - q) - mgl * np.cos(q), -1, 1)
    assert res.q[-1] == pytest.approx(q_eq, abs=1e-4)
    assert (q_d - res.q[-1]) == pytest.approx(mgl * np.cos(q_eq) / kp, rel=1e-3)
    ki = 0.25 * jc.ki_limit(J, B, kp, kd)
    res_i = jc.simulate(jc.Joint(J, B, mgl=mgl), jc.PID(kp, ki, kd, dt=1e-3), q_d, 12.0, 1e-3)
    assert abs(res_i.q[-1] - q_d) < 1e-4
    res_g = jc.simulate(jc.Joint(J, B, mgl=mgl), jc.PID(kp, 0, kd, dt=1e-3, gravity=lambda q: mgl * np.cos(q)),
                        q_d, 3.0, 1e-3)
    assert abs(res_g.q[-1] - q_d) < 1e-5


def test_saturation_and_antiwindup():
    J, B = 0.05, 0.1
    kp, kd = jc.pd_gains(J, B, 20.0, 1.0)
    ki = 0.3 * jc.ki_limit(J, B, kp, kd)
    out = {}
    for mode in ("none", "clamp", "backcalc"):
        pid = jc.PID(kp, ki, kd, dt=1e-3, u_min=-1.0, u_max=1.0, antiwindup=mode, kaw=kp / 5)
        res = jc.simulate(jc.Joint(J, B, u_max=1.0), pid, 5.0, 6.0, 1e-3)
        out[mode] = jc.step_metrics(res.t, res.q, y_final=5.0, y0=0.0)["overshoot"]
        assert np.max(np.abs(res.u)) <= 1.0 + 1e-12
    assert out["none"] > out["clamp"] + 5 and out["none"] > out["backcalc"] + 5


def test_delay_and_quantisation_run():
    J, B = 0.05, 0.4
    kp, kd = jc.pd_gains(J, B, 20.0, 1.0)
    base = jc.simulate(jc.Joint(J, B), jc.PID(kp, 0, kd, dt=1e-3), 1.0, 1.0, 1e-3)
    late = jc.simulate(jc.Joint(J, B), jc.PID(kp, 0, kd, dt=1e-3), 1.0, 1.0, 1e-3, delay=5)
    assert late.t.shape == base.t.shape and not np.allclose(late.q, base.q)
    q = jc.simulate(jc.Joint(J, B), jc.PID(kp, 0, kd, dt=1e-3), 1.0, 1.0, 1e-3, enc_res=1e-3, vel_from_diff=True)
    assert abs(q.q[-1] - 1.0) < 5e-3


# ── planar 2R ──────────────────────────────────────────────────────────────────────────────────────
def test_2r_matches_sympy_lagrangian():
    sp = pytest.importorskip("sympy")
    t = sp.symbols("t")
    m1, m2, l1, l2, lc1, lc2, I1, I2, g = sp.symbols("m1 m2 l1 l2 lc1 lc2 I1 I2 g", positive=True)
    q1, q2 = sp.Function("q1")(t), sp.Function("q2")(t)
    x1, z1 = lc1 * sp.cos(q1), lc1 * sp.sin(q1)
    x2, z2 = l1 * sp.cos(q1) + lc2 * sp.cos(q1 + q2), l1 * sp.sin(q1) + lc2 * sp.sin(q1 + q2)
    T = sp.Rational(1, 2) * (m1 * (x1.diff(t) ** 2 + z1.diff(t) ** 2) + m2 * (x2.diff(t) ** 2 + z2.diff(t) ** 2)) \
        + sp.Rational(1, 2) * (I1 * q1.diff(t) ** 2 + I2 * (q1 + q2).diff(t) ** 2)
    V = g * (m1 * z1 + m2 * z2)
    L = T - V
    tau = [sp.diff(sp.diff(L, qi.diff(t)), t) - sp.diff(L, qi) for qi in (q1, q2)]
    arm = jc.Planar2R(m1=1.3, m2=0.7, l1=0.9, l2=0.6, lc1=0.4, lc2=0.25, I1=0.03, I2=0.01, g=9.81)
    vals = {m1: 1.3, m2: 0.7, l1: 0.9, l2: 0.6, lc1: 0.4, lc2: 0.25, I1: 0.03, I2: 0.01, g: 9.81}
    rng = np.random.default_rng(1)
    q, qd, qdd = rng.uniform(-2, 2, 2), rng.uniform(-3, 3, 2), rng.uniform(-5, 5, 2)
    sub = {q1: q[0], q2: q[1]}
    for fn, vec in ((q1.diff(t), qd[0]), (q2.diff(t), qd[1]), (q1.diff(t, 2), qdd[0]), (q2.diff(t, 2), qdd[1])):
        sub[fn] = vec
    sub_order = [(q1.diff(t, 2), qdd[0]), (q2.diff(t, 2), qdd[1]), (q1.diff(t), qd[0]), (q2.diff(t), qd[1]),
                 (q1, q[0]), (q2, q[1])]
    ref = np.array([float(e.subs(vals).subs(sub_order)) for e in tau])
    assert np.allclose(arm.tau_inverse(q, qd, qdd), ref, atol=1e-9)


def test_2r_structure_properties():
    arm = jc.Planar2R(Jm=(1e-3, 1e-3), r=(50.0, 50.0))
    rng = np.random.default_rng(0)
    for _ in range(20):
        q, qd = rng.uniform(-3, 3, 2), rng.uniform(-2, 2, 2)
        M = arm.M(q)
        assert np.allclose(M, M.T) and np.all(np.linalg.eigvalsh(M) > 0)
        # skew-symmetry of Mdot - 2C with Mdot by central differences along the motion
        h = 1e-6
        Mdot = (arm.M(q + h * qd) - arm.M(q - h * qd)) / (2 * h)
        N = Mdot - 2 * arm.C(q, qd)
        assert np.allclose(N, -N.T, atol=1e-6)
    # energy is conserved without friction or torque
    arm = jc.Planar2R(g=9.81)
    x = np.array([0.4, 0.9, 0.0, 0.0])
    def energy(x):
        q, qd = x[:2], x[2:]
        e1, tip = arm.fk(q)
        zc1 = arm.lc1 * np.sin(q[0]); zc2 = arm.l1 * np.sin(q[0]) + arm.lc2 * np.sin(q[0] + q[1])
        return 0.5 * qd @ arm.M(q) @ qd + arm.g * (arm.m1 * zc1 + arm.m2 * zc2)
    E0 = energy(x)
    res = jc.simulate(arm, lambda q, qd, a, b, c: np.zeros(2), np.zeros(2), 2.0, 1e-3, x0=x, substeps=5)
    q, qd = res.q, res.qd
    E = np.array([energy(np.concatenate([q[i], qd[i]])) for i in range(0, len(q), 100)])
    assert np.max(np.abs(E - E0)) < 1e-5


def test_2r_matches_mujoco():
    mujoco = pytest.importorskip("mujoco")
    arm = jc.Planar2R(m1=1.3, m2=0.7, l1=0.9, l2=0.6, lc1=0.4, lc2=0.25, I1=0.03, I2=0.01, g=9.81,
                      Jm=(2e-3, 1e-3), r=(30.0, 20.0), Bm=(1e-3, 2e-3))
    xml = f"""<mujoco><option gravity="0 0 -9.81" integrator="RK4" timestep="0.0005"/>
    <worldbody><body name="b1"><joint name="q1" axis="0 -1 0" armature="{30**2 * 2e-3}" damping="{30**2 * 1e-3}"/>
      <inertial pos="{arm.lc1} 0 0" mass="{arm.m1}" diaginertia="{arm.I1} {arm.I1} {arm.I1}"/>
      <body name="b2" pos="{arm.l1} 0 0"><joint name="q2" axis="0 -1 0" armature="{20**2 * 1e-3}" damping="{20**2 * 2e-3}"/>
        <inertial pos="{arm.lc2} 0 0" mass="{arm.m2}" diaginertia="{arm.I2} {arm.I2} {arm.I2}"/></body></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(3)
    for _ in range(5):
        q, qd = rng.uniform(-2.5, 2.5, 2), rng.uniform(-2, 2, 2)
        data.qpos[:], data.qvel[:] = q, qd
        mujoco.mj_forward(model, data)
        assert np.allclose(jc.mj_mass_matrix(model, data), arm.M(q), atol=1e-9)
        assert np.allclose(jc.mj_gravity(model, data, q), arm.gvec(q), atol=1e-9)
        bias = data.qfrc_bias.copy()           # Coriolis + gravity
        assert np.allclose(bias, arm.C(q, qd) @ qd + arm.gvec(q), atol=1e-9)
        # passive (damping) force
        assert np.allclose(-data.qfrc_passive, arm.friction(qd), atol=1e-9)


def test_mj_run_pd_matches_python_simulation():
    mujoco = pytest.importorskip("mujoco")
    xml = """<mujoco><option gravity="0 0 -9.81" integrator="RK4" timestep="0.001"/>
    <worldbody><body><joint name="q" axis="0 -1 0" armature="0.02" damping="0.1"/>
      <inertial pos="0.3 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    J = 0.02 + 0.01 + 1.0 * 0.3**2            # armature + I + m lc^2
    mgl = 1.0 * 9.81 * 0.3
    kp, kd = jc.pd_gains(J, 0.1, 15.0, 0.9)
    pid = jc.PID(kp, 0, kd, dt=1e-3)
    rec = jc.mj_run(model, data, pid, np.array([0.7]), 2.0, q0=np.array([0.0]))
    res = jc.simulate(jc.Joint(J, 0.1, mgl=mgl), jc.PID(kp, 0, kd, dt=1e-3), 0.7, 2.0, 1e-3, substeps=10)
    assert np.max(np.abs(rec["q"][:, 0] - res.q)) < 2e-3


# ── flexible joint and sampling ─────────────────────────────────────────────────────────────────────
def test_two_mass_resonance_matches_eigenvalues_and_frf():
    tm = jc.TwoMass(Jm=0.01, Jl=0.04, k=200.0, c=0.0)
    ev = np.linalg.eigvals(tm.A())
    w = np.abs(ev.imag).max()
    assert w == pytest.approx(tm.resonance(), rel=1e-9)
    assert tm.resonance() == pytest.approx(np.sqrt(200 * 0.05 / (0.01 * 0.04)))
    assert tm.antiresonance() == pytest.approx(np.sqrt(200 / 0.04))
    # low-frequency behaviour: rigid body of total inertia Jm + Jl
    s = 1e-2
    assert abs(tm.frf(s)[()] * (1j * s) ** 2 * (tm.Jm + tm.Jl)) == pytest.approx(1.0, abs=1e-3)
    # the motor-angle response has a zero at the antiresonance (damping 0)
    assert abs(tm.frf(tm.antiresonance() * 0.9999)) < 1e-2 * abs(tm.frf(tm.antiresonance() * 0.5))


def test_sampled_pd_poles():
    J, kp = 0.05, 0.05 * 20.0**2
    kd = 2 * 2.0 * 20.0 * J                      # zeta = 2: two distinct real poles
    Ts = 1e-5
    z = jc.sampled_pd_poles(J, kp, kd, Ts)
    s = np.log(z.astype(complex)) / Ts
    exact = np.roots([J, kd, kp])
    assert np.allclose(sorted(s.real), sorted(exact.real), rtol=1e-3)
    # a long sample time destabilises the same gains; delay lowers the margin further
    assert np.max(np.abs(jc.sampled_pd_poles(J, kp, kd, 0.2))) > 1.0
    m0 = np.max(np.abs(jc.sampled_pd_poles(J, kp, kd, 0.02)))
    m2 = np.max(np.abs(jc.sampled_pd_poles(J, kp, kd, 0.02, delay=2)))
    assert m2 > m0
    # the pole computation agrees with the time-domain simulation (stable vs unstable)
    for Ts_, expect_stable in ((0.01, True), (0.2, False)):
        res = jc.simulate(jc.Joint(J, 0.0), jc.PID(kp, 0, kd, dt=Ts_), 1.0, 20 * Ts_ + 5.0, Ts_, substeps=20)
        stable = np.max(np.abs(jc.sampled_pd_poles(J, kp, kd, Ts_))) < 1
        assert stable == expect_stable
        assert (abs(res.q[-1] - 1.0) < 1e-2) == expect_stable


def test_2r_fast_deriv_equals_matrix_form():
    arm = jc.Planar2R(m1=1.3, m2=0.7, l1=0.9, l2=0.6, lc1=0.4, lc2=0.25, I1=0.03, I2=0.01,
                      Jm=(2e-3, 1e-3), r=(30.0, 20.0), Bm=(1e-3, 2e-3))
    rng = np.random.default_rng(5)
    for _ in range(10):
        q, qd, tau = rng.uniform(-3, 3, 2), rng.uniform(-3, 3, 2), rng.uniform(-5, 5, 2)
        d = arm.deriv(0.0, np.concatenate([q, qd]), tau)
        assert np.allclose(d[:2], qd) and np.allclose(d[2:], arm.qdd(q, qd, tau))


def test_block_diagram_helpers_draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    s = jc.bd_sum(ax, 0, 0)
    b = jc.bd_box(ax, 2, 0, "PD")
    jc.bd_arrow(ax, [s["r"], b["l"]], r"$e$")
    jc.bd_arrow(ax, [b["b"], (2, -1), (0, -1), s["b"]])
    jc.bd_text(ax, 1, 1, "x")
    assert len(ax.patches) >= 2
    plt.close(fig)


def test_save_gif(tmp_path, monkeypatch):
    frames = [np.full((20, 30, 3), 40 * i, np.uint8) for i in range(4)]
    monkeypatch.setenv("RG_SAVE_ASSETS", "0")
    assert jc.save_gif(frames, tmp_path / "a.gif") is None
    monkeypatch.setenv("RG_SAVE_ASSETS", "1")
    out = jc.save_gif(frames, tmp_path / "a.gif", fps=4)
    from PIL import Image
    im = Image.open(out)
    assert im.n_frames == 4 and im.size == (30, 20)

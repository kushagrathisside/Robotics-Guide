"""Tests for rgkit.model_based_control: dynamics identities, simulators, controllers and the MuJoCo bridge."""
import numpy as np
import pytest

from rgkit import model_based_control as mbc

RNG = np.random.default_rng(7)


def arm():
    return mbc.Planar2R.from_links(m1=3.0, m2=2.0, l1=0.5, l2=0.5)


# ------------------------------------------------------------------ parameters and rigid-body identities
def test_parameter_mapping_of_uniform_rods():
    a = arm()
    assert np.allclose(a.pi, [0.916667, 0.25, 0.166667, 1.75, 0.5], atol=1e-6)


def test_inertia_matrix_symmetric_positive_definite():
    a = arm()
    for _ in range(50):
        q = RNG.uniform(-np.pi, np.pi, 2)
        M = a.mass(q)
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0)


def test_mass_dot_matches_finite_difference():
    a = arm()
    for _ in range(20):
        q, qd = RNG.uniform(-2, 2, 2), RNG.uniform(-3, 3, 2)
        h = 1e-6
        fd = (a.mass(q + h * qd) - a.mass(q - h * qd)) / (2 * h)
        assert np.allclose(a.mass_dot(q, qd), fd, atol=1e-8)


def test_mdot_minus_2c_is_skew_symmetric():
    a = arm()
    for _ in range(50):
        q, qd = RNG.uniform(-3, 3, 2), RNG.uniform(-5, 5, 2)
        S = a.mass_dot(q, qd) - 2 * a.coriolis_matrix(q, qd)
        assert np.allclose(S, -S.T, atol=1e-12)
        x = RNG.standard_normal(2)
        assert abs(x @ S @ x) < 1e-12


def test_coriolis_vector_equals_matrix_times_velocity():
    a = arm()
    q, qd = RNG.uniform(-2, 2, 2), RNG.uniform(-3, 3, 2)
    assert np.allclose(a.coriolis(q, qd), a.coriolis_matrix(q, qd) @ qd)


def test_gravity_is_gradient_of_potential_energy():
    a = arm()
    for _ in range(20):
        q = RNG.uniform(-3, 3, 2)
        h = 1e-6
        grad = np.array([(a.potential_energy(q + h * e) - a.potential_energy(q - h * e)) / (2 * h) for e in np.eye(2)])
        assert np.allclose(a.gravity(q), grad, atol=1e-6)
        Jg = np.array([(a.gravity(q + h * e) - a.gravity(q - h * e)) / (2 * h) for e in np.eye(2)]).T
        assert np.allclose(a.gravity_jacobian(q), Jg, atol=1e-5)


def test_matches_symbolic_lagrangian():
    sp = pytest.importorskip("sympy")
    a = arm()
    q1, q2, v1, v2 = sp.symbols("q1 q2 v1 v2")
    m1, m2, l1, l2, g = 3.0, 2.0, 0.5, 0.5, 9.81
    lc1, lc2, I1, I2 = l1 / 2, l2 / 2, m1 * l1**2 / 12, m2 * l2**2 / 12
    # centres of mass
    x1, y1 = lc1 * sp.cos(q1), lc1 * sp.sin(q1)
    x2, y2 = l1 * sp.cos(q1) + lc2 * sp.cos(q1 + q2), l1 * sp.sin(q1) + lc2 * sp.sin(q1 + q2)
    q, v = sp.Matrix([q1, q2]), sp.Matrix([v1, v2])
    J = lambda x, y: sp.Matrix([x, y]).jacobian(q)
    T = sp.Rational(1, 2) * (m1 * (J(x1, y1) * v).dot(J(x1, y1) * v) + m2 * (J(x2, y2) * v).dot(J(x2, y2) * v)
                             + I1 * v1**2 + I2 * (v1 + v2) ** 2)
    U = g * (m1 * y1 + m2 * y2)
    Msym = sp.hessian(T, v)
    gsym = sp.Matrix([sp.diff(U, q1), sp.diff(U, q2)])
    Csym = sp.zeros(2, 2)
    for i in range(2):
        for j in range(2):
            Csym[i, j] = sum(sp.Rational(1, 2) * (sp.diff(Msym[i, j], q[k]) + sp.diff(Msym[i, k], q[j])
                                                  - sp.diff(Msym[j, k], q[i])) * v[k] for k in range(2))
    f = sp.lambdify((q1, q2, v1, v2), (Msym, Csym, gsym), "numpy")
    for _ in range(10):
        qn, vn = RNG.uniform(-3, 3, 2), RNG.uniform(-3, 3, 2)
        Mn, Cn, gn = f(*qn, *vn)
        assert np.allclose(a.mass(qn), Mn)
        assert np.allclose(a.coriolis_matrix(qn, vn), Cn)
        assert np.allclose(a.gravity(qn), np.asarray(gn, float).ravel())


def test_regressor_reproduces_inverse_dynamics():
    a = arm()
    for _ in range(20):
        q, qd, qdd = RNG.uniform(-3, 3, 2), RNG.uniform(-3, 3, 2), RNG.uniform(-5, 5, 2)
        assert np.allclose(a.regressor(q, qd, qdd) @ a.pi, a.inverse_dynamics(q, qd, qdd))


def test_forward_inverse_dynamics_round_trip():
    a = arm()
    q, qd, qdd = RNG.uniform(-2, 2, 2), RNG.uniform(-2, 2, 2), RNG.uniform(-4, 4, 2)
    tau = a.inverse_dynamics(q, qd, qdd)
    assert np.allclose(a.acceleration(q, qd, tau, friction=False), qdd)


def test_payload_adds_point_mass_exactly():
    a = arm()
    p = a.with_payload(1.5)
    q = np.array([0.4, -0.9])
    # kinetic and potential energy of a point mass at the tip
    qd = np.array([0.7, -1.1])
    _, tip = a.forward_kinematics(q)
    v_tip = a.jacobian(q) @ qd
    dT = p.kinetic_energy(q, qd) - a.kinetic_energy(q, qd)
    assert dT == pytest.approx(0.5 * 1.5 * v_tip @ v_tip)
    assert p.potential_energy(q) - a.potential_energy(q) == pytest.approx(1.5 * a.g0 * tip[1])


# ------------------------------------------------------------------ simulation
def test_unforced_frictionless_motion_conserves_energy():
    a = arm()
    sim = mbc.simulate(a, lambda t, q, qd: np.zeros(2), q0=[0.3, 1.0], qd0=[1.0, -2.0], T=3.0, dt=1e-3)
    E = np.array([a.energy(q, qd) for q, qd in zip(sim.q, sim.qd)])
    assert np.ptp(E) < 1e-6 * max(1.0, abs(E[0]))   # RK4 drift at dt = 1 ms


def test_power_balance_for_applied_torque():
    """dE/dt = tau . qd (passivity of the manipulator)."""
    a = arm()
    ctrl = lambda t, q, qd: np.array([2.0 * np.sin(3 * t), -1.0 * np.cos(2 * t)])
    sim = mbc.simulate(a, ctrl, q0=[0.0, 0.5], T=2.0, dt=5e-4)
    E = np.array([a.energy(q, qd) for q, qd in zip(sim.q, sim.qd)])
    work = np.sum(sim.tau[:-1] * np.diff(sim.q, axis=0))     # tau is constant over each step: work = tau . dq
    assert E[-1] - E[0] == pytest.approx(work, abs=1e-6)


def test_pd_alone_leaves_gravity_error_equal_to_fixed_point():
    from scipy.optimize import fsolve
    a = arm()
    Kp, Kd = np.diag([200.0, 150.0]), np.diag([30.0, 20.0])
    qd_ = np.array([0.8, 0.6])
    sim = mbc.simulate(a, mbc.pd_control(Kp, Kd, qd_), q0=[0.0, 0.0], T=8.0, ref=qd_)
    qs = fsolve(lambda q: Kp @ (qd_ - q) - a.gravity(q), qd_)
    assert np.allclose(sim.q[-1], qs, atol=1e-6)
    assert np.linalg.norm(qd_ - qs) > 0.05


@pytest.mark.parametrize("mode", ["current", "desired"])
def test_pd_with_gravity_compensation_converges(mode):
    a = arm()
    Kp, Kd = np.diag([200.0, 150.0]), np.diag([30.0, 20.0])
    qd_ = np.array([0.8, 0.6])
    sim = mbc.simulate(a, mbc.pd_gravity_control(a, Kp, Kd, qd_, mode), q0=[-1.0, 1.0], T=6.0)
    assert np.allclose(sim.q[-1], qd_, atol=1e-6)


def test_lyapunov_derivative_equals_minus_qd_Kd_qd():
    a = arm()
    Kp, Kd = np.diag([200.0, 150.0]), np.diag([30.0, 20.0])
    qd_ = np.array([0.8, 0.6])
    sim = mbc.simulate(a, mbc.pd_gravity_control(a, Kp, Kd, qd_), q0=[-1.0, 1.0], T=2.0, dt=2e-4)
    V = np.array([mbc.pd_lyapunov(a, q, v, qd_, Kp) for q, v in zip(sim.q, sim.qd)])
    Vdot = np.gradient(V, sim.t)
    pred = -np.einsum("ni,ij,nj->n", sim.qd, Kd, sim.qd)
    assert np.max(np.abs(Vdot[5:-5] - pred[5:-5])) < 1e-2 * np.max(np.abs(pred))
    assert np.all(np.diff(V) <= 1e-6 * V[0])      # V decreases (up to the zero-order-hold error)


def test_computed_torque_error_obeys_linear_ode():
    a = arm()
    wn, zeta = 12.0, 1.0
    Kp, Kv = gains = mbc.gains_from_wn_zeta(wn, zeta)
    ref = mbc.sine_reference([0.3, 0.8], [0.5, 0.7], [3.0, 2.0])
    ctrl = mbc.ComputedTorque(a, Kp, Kv, ref)
    q0r, qd0r, _ = ref(0.0)
    e0 = np.array([0.2, -0.3])
    ed0 = np.array([0.0, 0.5])
    sim = mbc.simulate(a, ctrl, q0=q0r - e0, qd0=qd0r - ed0, T=2.0, dt=5e-4, ref=ref)
    # analytic solution of e'' + 2 zeta wn e' + wn^2 e = 0 (critical damping)
    t = sim.t[:, None]
    e_exact = (e0 + (ed0 + wn * e0) * t) * np.exp(-wn * t)
    assert np.max(np.abs(sim.e - e_exact)) < 1e-3   # discretisation (zero-order hold, 0.5 ms) only


def test_computed_torque_with_wrong_model_degrades_and_is_better_than_pd():
    a = arm()
    ref = mbc.sine_reference([0.3, 0.8], [0.6, 0.8], [4.0, 3.0])
    Kp, Kv = mbc.gains_from_wn_zeta(15.0, 1.0)
    run = lambda ctrl: mbc.rms(mbc.simulate(a, ctrl, q0=ref(0)[0], qd0=ref(0)[1], T=3.0, ref=ref).e)
    e_exact = run(mbc.ComputedTorque(a, Kp, Kv, ref))
    e_wrong = run(mbc.ComputedTorque(a.scaled(0.8), Kp, Kv, ref))
    e_pdg = run(mbc.pd_gravity_control(a, np.diag(Kp * 0.5 * np.ones(2)), np.diag(Kv * 0.5 * np.ones(2)), ref))
    assert e_exact < 5e-4 and 10 * e_exact < e_wrong < e_pdg


def test_delay_and_sampling_are_applied():
    a = arm()
    log = []
    ctrl = lambda t, q, qd: (log.append(t), np.array([1.0, 0.0]) * t)[1]
    sim = mbc.simulate(a, ctrl, q0=[0, 0], T=0.1, dt=1e-3, ctrl_dt=1e-2, delay=3e-3)
    assert len(log) == 11                       # 0, 10 ms, ..., 100 ms
    assert np.allclose(sim.tau[:4, 0], 0.0)     # delayed by 3 ms: initial command repeated (t = 0 gives 0)
    assert sim.tau[13, 0] == pytest.approx(0.01)   # still holding the command computed at t = 10 ms


def test_torque_saturation():
    a = arm()
    sim = mbc.simulate(a, lambda t, q, qd: np.array([100.0, -100.0]), q0=[0, 0], T=0.05, tau_max=5.0)
    assert np.allclose(np.abs(sim.tau), 5.0)


def test_quintic_reference_endpoints_and_derivatives():
    ref = mbc.quintic_reference([0, 1], [1, -1], 2.0)
    q0, v0, a0 = ref(0.0)
    q1, v1, a1 = ref(2.0)
    assert np.allclose(q0, [0, 1]) and np.allclose(q1, [1, -1]) and np.allclose(v0, 0) and np.allclose(v1, 0)
    assert np.allclose(a0, 0) and np.allclose(a1, 0)
    h = 1e-6
    assert np.allclose((ref(0.7 + h)[0] - ref(0.7 - h)[0]) / (2 * h), ref(0.7)[1], atol=1e-6)
    assert np.allclose((ref(0.7 + h)[1] - ref(0.7 - h)[1]) / (2 * h), ref(0.7)[2], atol=1e-5)


def test_sine_reference_derivatives():
    ref = mbc.sine_reference([0.1, 0.2], [0.5, 0.4], [2.0, 3.0], [0.3, -0.2])
    h = 1e-6
    assert np.allclose((ref(1.3 + h)[0] - ref(1.3 - h)[0]) / (2 * h), ref(1.3)[1], atol=1e-6)
    assert np.allclose((ref(1.3 + h)[1] - ref(1.3 - h)[1]) / (2 * h), ref(1.3)[2], atol=1e-5)


def test_helpers():
    Kp, Kv = mbc.gains_from_wn_zeta(10.0, 0.7)
    assert Kp == 100.0 and Kv == pytest.approx(14.0)
    t = np.linspace(0, 1, 101)
    y = 1 - np.exp(-10 * t)
    assert mbc.settling_time(t, y, 1.0, tol=0.02) == pytest.approx(np.log(50) / 10, abs=0.011)
    lo, hi = mbc.workspace_inertia_range(arm())
    assert 0 < lo < hi


# ------------------------------------------------------------------ MuJoCo bridge
mujoco = pytest.importorskip("mujoco")

TOY_XML = """
<mujoco><option gravity="0 0 -9.81" timestep="0.001"/>
 <worldbody>
  <body name="l1">
   <joint name="q1" type="hinge" axis="0 -1 0"/>
   <inertial pos="0.25 0 0" mass="3.0" diaginertia="1e-4 {I1} {I1}"/>
   <body name="l2" pos="0.5 0 0">
    <joint name="q2" type="hinge" axis="0 -1 0"/>
    <inertial pos="0.25 0 0" mass="2.0" diaginertia="1e-4 {I2} {I2}"/>
    <site name="tip" pos="0.5 0 0"/>
   </body>
  </body>
 </worldbody></mujoco>""".format(I1=3.0 * 0.5**2 / 12, I2=2.0 * 0.5**2 / 12)


def test_planar_model_matches_mujoco_mass_bias_and_tip():
    model = mujoco.MjModel.from_xml_string(TOY_XML)
    a = arm()
    dyn = mbc.MuJoCoDynamics(model)
    data = mujoco.MjData(model)
    for _ in range(10):
        q, qd = RNG.uniform(-3, 3, 2), RNG.uniform(-3, 3, 2)
        assert np.allclose(dyn.mass(q), a.mass(q), atol=1e-9)
        assert np.allclose(dyn.bias(q, qd), a.bias(q, qd), atol=1e-9)
        assert np.allclose(dyn.gravity(q), a.gravity(q), atol=1e-9)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        assert np.allclose(data.site("tip").xpos[[0, 2]], a.forward_kinematics(q)[1])


def test_mujoco_inverse_dynamics_equals_m_qdd_plus_bias():
    model = mujoco.MjModel.from_xml_string(TOY_XML)
    dyn = mbc.MuJoCoDynamics(model)
    q, qd, qdd = RNG.uniform(-2, 2, 2), RNG.uniform(-2, 2, 2), RNG.uniform(-4, 4, 2)
    M, b = dyn.mass_and_bias(q, qd)
    assert np.allclose(dyn.inverse_dynamics(q, qd, qdd), M @ qdd + b)


def test_same_controller_runs_on_mujoco_and_on_python_model():
    """Computed torque with an exact model gives the same trajectory in MuJoCo and in the RK4 simulator."""
    model = mujoco.MjModel.from_xml_string(TOY_XML)
    data = mujoco.MjData(model)
    mbc.configure_torque_control(model)
    a = arm()
    ref = mbc.sine_reference([0.3, 0.8], [0.5, 0.6], [3.0, 2.0])
    Kp, Kv = mbc.gains_from_wn_zeta(15.0, 1.0)
    q0, v0, _ = ref(0.0)
    s_mj = mbc.simulate_mujoco(model, data, mbc.ComputedTorque(mbc.MuJoCoDynamics(model), Kp, Kv, ref), 1.5, q0, v0, ref=ref)
    s_py = mbc.simulate(a, mbc.ComputedTorque(a, Kp, Kv, ref), q0, 1.5, dt=1e-3, qd0=v0, ref=ref)
    assert mbc.rms(s_mj.e) < 5e-4 and mbc.rms(s_py.e) < 5e-4
    assert np.max(np.abs(s_mj.q - s_py.q)) < 2e-3


def test_flag_helpers():
    model = mujoco.MjModel.from_xml_string(TOY_XML)
    old = mbc.configure_torque_control(model, contacts=False, energy=True)
    d = mbc.describe_flags(model)
    assert "ACTUATION" in d["disabled"] and "CONTACT" in d["disabled"] and "ENERGY" in d["enabled"]
    assert old["disable"] == 0
    assert d["gravity"] == (0.0, 0.0, -9.81) and d["integrator"] == "Euler"

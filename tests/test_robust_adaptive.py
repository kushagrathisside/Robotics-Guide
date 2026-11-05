"""Tests for rgkit.robust_adaptive (RMC Chapter 4)."""
import numpy as np
import pytest

from rgkit import robust_adaptive as ra

RNG = np.random.default_rng(7)

# physical parameters of the textbook arm used throughout the chapter
PHYS = dict(m1=4.0, m2=2.0, l1=1.0, l2=0.8, lc1=0.5, lc2=0.4, I1=4.0 / 12, I2=2.0 * 0.8 ** 2 / 12)


def arm():
    return ra.Arm2R.from_physical(**PHYS)


def generic_arm():
    return ra.PlanarArm([PHYS["l1"], PHYS["l2"]], [PHYS["m1"], PHYS["m2"]], [PHYS["lc1"], PHYS["lc2"]],
                        [PHYS["I1"], PHYS["I2"]])


def random_state():
    return RNG.uniform(-2.5, 2.5, 2), RNG.uniform(-2, 2, 2), RNG.uniform(-3, 3, 2)


def test_lumped_parameters():
    th = arm().theta
    assert np.allclose(th, [3.76, 0.426666666667, 0.8, 4.0, 0.8])


@pytest.mark.parametrize("trial", range(5))
def test_closed_form_matches_generic_planar_arm(trial):
    a, g = arm(), generic_arm()
    q, qd, qdd = random_state()
    assert np.allclose(a.M(q), g.M(q), atol=1e-12)
    assert np.allclose(a.coriolis(q, qd), g.coriolis(q, qd), atol=1e-12)
    assert np.allclose(a.gravity(q), g.gravity(q), atol=1e-12)
    assert np.allclose(a.tau(q, qd, qdd), g.tau(q, qd, qdd), atol=1e-12)
    tau = RNG.uniform(-20, 20, 2)
    assert np.allclose(a.accel(q, qd, tau), g.accel(q, qd, tau), atol=1e-10)
    assert np.allclose(a.tau(q, qd, a.accel(q, qd, tau)), tau, atol=1e-10)
    assert np.allclose(a.C(q, qd), g.C(q, qd), atol=1e-7)


@pytest.mark.parametrize("trial", range(5))
def test_regressor_reproduces_dynamics(trial):
    a = arm()
    q, qd, qdd = random_state()
    qdr, qddr = RNG.uniform(-2, 2, 2), RNG.uniform(-3, 3, 2)
    Y = a.regressor(q, qd, qd, qdd)
    assert np.allclose(Y @ a.theta, a.tau(q, qd, qdd), atol=1e-12)                 # Y theta = tau
    Yr = a.regressor(q, qd, qdr, qddr)
    assert np.allclose(Yr @ a.theta, a.M(q) @ qddr + a.C(q, qd) @ qdr + a.gravity(q), atol=1e-12)


@pytest.mark.parametrize("trial", range(5))
def test_skew_symmetry(trial):
    a = arm()
    q, qd, _ = random_state()
    h = 1e-6
    Mdot = (a.M(q + h * qd) - a.M(q - h * qd)) / (2 * h)
    N = Mdot - 2 * a.C(q, qd)
    assert np.allclose(N, -N.T, atol=1e-7)
    assert np.allclose(Mdot, a.C(q, qd) + a.C(q, qd).T, atol=1e-7)                  # M' = C + C^T
    assert np.allclose(a.C(q, qd) @ qd, a.coriolis(q, qd))


def test_payload_theta_matches_physical_tip_mass():
    mp = 1.3
    a = arm().with_payload(mp)
    # equivalent link 2: merge a point mass at the tip into link 2
    m2n = PHYS["m2"] + mp
    lc2n = (PHYS["m2"] * PHYS["lc2"] + mp * PHYS["l2"]) / m2n
    I2n = PHYS["I2"] + PHYS["m2"] * (lc2n - PHYS["lc2"]) ** 2 + mp * (PHYS["l2"] - lc2n) ** 2
    th = ra.Arm2R.theta_from_physical(PHYS["m1"], m2n, PHYS["l1"], PHYS["lc1"], lc2n, PHYS["I1"], I2n)
    assert np.allclose(a.theta, th)


def test_jacobians_by_finite_differences():
    for model in (arm(), generic_arm()):
        q, qd, _ = random_state()
        h = 1e-6
        J = model.jac(q)
        for k in range(2):
            e = np.zeros(2); e[k] = h
            assert np.allclose(J[:, k], (model.fk(q + e) - model.fk(q - e)) / (2 * h), atol=1e-8)
        Jd = (model.jac(q + h * qd) - model.jac(q - h * qd)) / (2 * h)
        assert np.allclose(model.jac_dot(q, qd), Jd, atol=1e-7)


def test_ik_roundtrip():
    a = arm()
    for _ in range(5):
        q = np.array([RNG.uniform(-3, 3), RNG.uniform(0.2, 2.5)])
        dq = a.ik(a.fk(q), elbow_up=True) - q
        assert np.allclose(np.arctan2(np.sin(dq), np.cos(dq)), 0, atol=1e-9)         # equal modulo 2 pi


def test_three_link_jacobian_and_dynamics_consistency():
    g = ra.PlanarArm([0.5, 0.4, 0.3], [3, 2, 1], [0.25, 0.2, 0.15], [0.06, 0.03, 0.01])
    q, qd = RNG.uniform(-1.5, 1.5, 3), RNG.uniform(-1, 1, 3)
    M = g.M(q)
    assert np.allclose(M, M.T) and np.all(np.linalg.eigvalsh(M) > 0)
    assert np.allclose(g.C(q, qd) @ qd, g.coriolis(q, qd), atol=1e-6)
    h = 1e-6
    Mdot = (g.M(q + h * qd) - g.M(q - h * qd)) / (2 * h)
    N = Mdot - 2 * g.C(q, qd)
    assert np.allclose(N, -N.T, atol=1e-6)


# ─────────────────────────────────── simulation / controllers ───────────────────────────────────

def _ref():
    return ra.sinusoids([0.4, 1.0], [0.5, 0.4], [1.2, 1.6])


def _x0():
    q_d, dq_d, _ = _ref()(0.0)
    return q_d + np.array([0.15, -0.1]), dq_d


def test_simulation_matches_energy_conservation_without_gravity():
    a = arm()
    a.g = 0.0
    sim = ra.simulate(a.accel, lambda t, q, qd, z: (np.zeros(2), None), [0.3, 0.8], [1.0, -0.5], 2.0, dt=1e-3)
    E = [0.5 * qd @ a.M(q) @ qd for q, qd in zip(sim.q, sim.qd)]
    assert max(E) - min(E) < 1e-8


def test_computed_torque_exact_model():
    a = arm()
    q0, qd0 = _x0()
    sim = ra.simulate(a.accel, ra.ct_controller(a, _ref(), 100 * np.eye(2), 20 * np.eye(2)), q0, qd0, 4.0)
    e = sim.q - np.array([_ref()(t)[0] for t in sim.t])
    assert np.abs(e[-1]).max() < 2e-4                      # residual: zero-order-hold sampling at 1 ms


def test_computed_torque_with_wrong_model_has_error_but_sliding_mode_is_bounded():
    a_true = arm().with_payload(2.0)
    a_hat = arm()
    q0, qd0 = _x0()
    ref = _ref()
    ct = ra.simulate(a_true.accel, ra.ct_controller(a_hat, ref, 100 * np.eye(2), 20 * np.eye(2)), q0, qd0, 6.0)
    lam, phi = np.array([10.0, 10.0]), np.array([0.1, 0.1])
    dtheta = np.abs(a_true.theta - a_hat.theta) * 1.2
    sl = ra.simulate(a_true.accel, ra.sliding_controller(a_hat, ref, lam, 5 * np.eye(2), dtheta, 1.0, phi), q0, qd0, 6.0)
    qd_ref = np.array([ref(t)[0] for t in ct.t])
    e_ct = np.abs(ct.q - qd_ref)[3000:].max()
    e_sl = np.abs(sl.q - qd_ref)[3000:].max()
    assert e_sl < e_ct                                   # robustness pays off
    assert e_sl < phi[0] / lam[0] + 1e-9                 # |e| <= Phi / lambda inside the boundary layer


def test_sliding_mode_reaches_surface_in_finite_time():
    a_true = arm().with_payload(1.5)
    a_hat = arm()
    ref = _ref()
    q0, qd0 = _x0()
    lam = np.array([8.0, 8.0])
    dtheta = np.abs(a_true.theta - a_hat.theta) * 1.5
    sim = ra.simulate(a_true.accel, ra.sliding_controller(a_hat, ref, lam, 2 * np.eye(2), dtheta, 2.0, 0.0), q0, qd0, 3.0,
                      dt=1e-4, hold=1)
    s = np.array([ra.sliding_variable(q, qd, ref(t), lam)[0] for t, q, qd in zip(sim.t, sim.q, sim.qd)])
    assert np.abs(s[-2000:]).max() < 2e-2                # chatters in a thin band around s = 0


def test_robust_lyapunov_redesign_is_ultimately_bounded():
    a_true, a_hat = arm().with_payload(1.0), arm()
    ref = _ref()
    q0, qd0 = _x0()
    Kp, Kd = 25 * np.eye(2), 10 * np.eye(2)
    ct = ra.simulate(a_true.accel, ra.ct_controller(a_hat, ref, Kp, Kd), q0, qd0, 6.0)
    rb = ra.simulate(a_true.accel, ra.robust_ct_controller(a_hat, ref, Kp, Kd, rho=20.0, eps=0.05), q0, qd0, 6.0)
    qdv = np.array([ref(t)[0] for t in ct.t])
    assert np.abs(rb.q - qdv)[3000:].max() < np.abs(ct.q - qdv)[3000:].max()


RICH = lambda t: (np.array([0.4 + 0.6 * np.sin(1.1 * t) + 0.2 * np.sin(2.7 * t), 1.0 + 0.7 * np.sin(1.7 * t + 1) + 0.2 * np.sin(3.1 * t)]),
                  np.array([0.6 * 1.1 * np.cos(1.1 * t) + 0.2 * 2.7 * np.cos(2.7 * t),
                            0.7 * 1.7 * np.cos(1.7 * t + 1) + 0.2 * 3.1 * np.cos(3.1 * t)]),
                  np.array([-0.6 * 1.1 ** 2 * np.sin(1.1 * t) - 0.2 * 2.7 ** 2 * np.sin(2.7 * t),
                            -0.7 * 1.7 ** 2 * np.sin(1.7 * t + 1) - 0.2 * 3.1 ** 2 * np.sin(3.1 * t)]))


def test_slotine_li_adaptive_tracking_and_boundedness():
    a_true = arm().with_payload(1.5)
    a_hat = arm()                                        # starts without the payload
    ctrl = ra.adaptive_controller(a_hat, _ref(), 8 * np.ones(2), 20 * np.eye(2), 5.0 * np.eye(5))
    q0, qd0 = _x0()
    sim = ra.simulate(a_true.accel, ctrl, q0, qd0, 30.0, dt=2e-3, z0=a_hat.theta)
    e = sim.q - np.array([_ref()(t)[0] for t in sim.t])
    assert np.abs(e[-500:]).max() < 5e-3                 # tracking error tends to zero
    assert np.abs(e[:500]).max() > 5 * np.abs(e[-500:]).max()
    assert np.abs(sim.z).max() < 50 and sim.z.shape == (len(sim.t), 5)


def test_parameter_convergence_needs_persistent_excitation():
    a_true, a_hat = arm().with_payload(1.5), arm()
    q0 = np.array([0.4, 1.0])
    hold = ra.min_jerk(q0, q0, 1.0)                      # constant set point: poor excitation
    c1 = ra.adaptive_controller(a_hat, hold, 5 * np.ones(2), 20 * np.eye(2), 5.0 * np.eye(5))
    s1 = ra.simulate(a_true.accel, c1, q0 + 0.2, np.zeros(2), 30.0, dt=2e-3, z0=a_hat.theta)
    c2 = ra.adaptive_controller(a_hat, RICH, 8 * np.ones(2), 20 * np.eye(2), 5.0 * np.eye(5))
    s2 = ra.simulate(a_true.accel, c2, RICH(0.0)[0] + 0.1, RICH(0.0)[1], 40.0, dt=2e-3, z0=a_hat.theta)
    err0 = np.linalg.norm(a_hat.theta - a_true.theta)
    err1 = np.linalg.norm(s1.z[-1] - a_true.theta)
    err2 = np.linalg.norm(s2.z[-1] - a_true.theta)
    assert err2 < 0.05 * err0                            # rich trajectory: parameters converge
    assert err1 > 0.3 * err0                             # set point: they do not (only gravity terms are identified)
    assert np.allclose(s1.z[-1][3:], a_true.theta[3:], atol=1e-2)


# ─────────────────────────────────── operational space ───────────────────────────────────

def test_task_space_dynamics_identity():
    a = arm()
    q, qd, _ = random_state()
    q[1] = abs(q[1]) + 0.5
    F = RNG.uniform(-5, 5, 2)
    J = a.jac(q)
    qdd = a.accel(q, qd, J.T @ F)
    xdd = J @ qdd + a.jac_dot(q, qd) @ qd
    Lam, mu, p = ra.task_dynamics(a.M(q), a.coriolis(q, qd), a.gravity(q), J, a.jac_dot(q, qd) @ qd)
    assert np.allclose(Lam @ xdd + mu + p, F, atol=1e-9)
    assert np.allclose(Lam, Lam.T)


def test_payload_adds_its_mass_to_the_task_inertia():
    a = arm()
    q = np.array([0.4, 1.1])
    J = a.jac(q)
    mp = 2.5
    L0 = ra.task_inertia(a.M(q), J)
    L1 = ra.task_inertia(a.with_payload(mp).M(q), J)
    assert np.allclose(L1, L0 + mp * np.eye(2), atol=1e-10)


def test_operational_space_control_decouples_and_tracks_a_line():
    a = arm()
    Kp, Kd = 100.0 * np.eye(2), 20.0 * np.eye(2)
    xa, xb = np.array([1.2, 0.5]), np.array([0.5, 1.1])
    ref = ra.task_line(xa, xb, 1.5)
    q0 = a.ik(xa)

    def ctrl(t, q, qd, z):
        x_d, xd_d, xdd_d = ref(t)
        J, Jd = a.jac(q), a.jac_dot(q, qd)
        x, xd = a.fk(q), J @ qd
        Lam, mu, p = ra.task_dynamics(a.M(q), a.coriolis(q, qd), a.gravity(q), J, Jd @ qd)
        F = ra.osc_force(Lam, mu, p, xdd_d - Kd @ (xd - xd_d) - Kp @ (x - x_d))
        return J.T @ F, None

    sim = ra.simulate(a.accel, ctrl, q0, np.zeros(2), 2.0)
    pts = np.array([a.fk(q) for q in sim.q])
    d = xb - xa
    u = d / np.linalg.norm(d)
    dist = np.abs((pts - xa) @ np.array([-u[1], u[0]]))                # distance from the line
    assert dist.max() < 5e-4                                           # the end point stays on the line (ZOH error)
    assert np.linalg.norm(pts[-1] - xb) < 1e-4


def test_jt_pd_regulation_and_singular_trap():
    a = arm()
    Kp, Kd = 80.0 * np.eye(2), 25.0 * np.eye(2)
    xd = np.array([0.9, 0.9])

    def make(xd):
        def ctrl(t, q, qd, z):
            J = a.jac(q)
            return J.T @ ra.jt_pd_force(a.fk(q) - xd, J @ qd, Kp, Kd) + a.gravity(q), None
        return ctrl

    sim = ra.simulate(a.accel, make(xd), np.array([0.2, 1.5]), np.zeros(2), 6.0)
    assert np.linalg.norm(a.fk(sim.q[-1]) - xd) < 1e-3
    # arm stretched along x, target on the same ray but closer: J^T e = 0, the arm does not move
    sim2 = ra.simulate(a.accel, make(np.array([1.0, 0.0])), np.array([0.0, 0.0]), np.zeros(2), 2.0)
    assert np.abs(sim2.q[-1]).max() < 1e-9


def test_null_space_projectors():
    g = ra.PlanarArm([0.5, 0.4, 0.3], [3, 2, 1], [0.25, 0.2, 0.15], [0.06, 0.03, 0.01])
    q = np.array([0.3, 0.9, -0.6])
    J, M = g.jac(q), g.M(q)
    N = ra.null_projector(J)
    assert np.allclose(J @ N, 0, atol=1e-12) and np.allclose(N @ N, N)
    Jb = ra.dyn_pinv(M, J)
    assert np.allclose(J @ Jb, np.eye(2), atol=1e-10)
    Nbt = ra.dyn_null_projector(M, J)
    tau0 = RNG.uniform(-3, 3, 3)
    assert np.allclose(J @ np.linalg.solve(M, Nbt @ tau0), 0, atol=1e-10)         # no task acceleration
    assert np.allclose(Nbt @ Nbt, Nbt, atol=1e-10)
    assert np.allclose(ra.damped_pinv(J, 1e-9), np.linalg.pinv(J), atol=1e-6)
    assert ra.manipulability(J) > 0


# ─────────────────────────────────── MuJoCo cross-checks ───────────────────────────────────

mujoco = pytest.importorskip("mujoco")


def _mj_mass(model, data):
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    return M


def test_mujoco_planar_2r_matches_hand_model():
    g = generic_arm()
    a = arm()
    model = mujoco.MjModel.from_xml_string(ra.mj_planar_xml(g.l, g.m, g.lc, g.I))
    data = mujoco.MjData(model)
    for _ in range(4):
        q, qd, qdd = random_state()
        data.qpos[:], data.qvel[:] = q, qd
        mujoco.mj_forward(model, data)
        assert np.allclose(_mj_mass(model, data), a.M(q), atol=1e-10)
        assert np.allclose(data.qfrc_bias, a.bias(q, qd), atol=1e-10)
        data.qacc[:] = qdd
        mujoco.mj_inverse(model, data)
        assert np.allclose(data.qfrc_inverse, a.tau(q, qd, qdd), atol=1e-9)
        assert np.allclose(data.site("tip").xpos[:2], a.fk(q), atol=1e-12)


def test_mujoco_point_mass_matches_payload_theta():
    g = generic_arm()
    model = mujoco.MjModel.from_xml_string(ra.mj_planar_xml(g.l, g.m, g.lc, g.I))
    data = mujoco.MjData(model)
    ra.mj_add_point_mass(model, "link2", 1.7, [PHYS["l2"], 0, 0])
    a = arm().with_payload(1.7)
    q, qd, _ = random_state()
    data.qpos[:], data.qvel[:] = q, qd
    mujoco.mj_forward(model, data)
    assert np.allclose(_mj_mass(model, data), a.M(q), atol=1e-10)
    assert np.allclose(data.qfrc_bias, a.bias(q, qd), atol=1e-10)


def test_mujoco_simulation_matches_python_integration():
    g = generic_arm()
    a = arm()
    model = mujoco.MjModel.from_xml_string(ra.mj_planar_xml(g.l, g.m, g.lc, g.I, timestep=1e-3))
    data = mujoco.MjData(model)
    q0, qd0 = np.array([0.3, 0.8]), np.array([0.2, -0.1])
    data.qpos[:], data.qvel[:] = q0, qd0
    tau = np.array([3.0, -1.0])
    for _ in range(500):
        data.qfrc_applied[:] = tau
        mujoco.mj_step(model, data)
    sim = ra.simulate(a.accel, lambda t, q, qd, z: (tau, None), q0, qd0, 0.5, dt=1e-3)
    assert np.allclose(data.qpos, sim.q[-1], atol=1e-6)


@pytest.fixture(scope="module")
def panda():
    try:
        return ra.mj_panda7()
    except Exception as exc:                                # pragma: no cover - offline without cache
        pytest.skip(f"Panda model unavailable: {exc}")


def test_panda_dynamics_helpers(panda):
    model, data = panda
    assert model.nv == 7
    data.qpos[:] = [0.1, -0.4, 0.2, -1.8, 0.3, 1.6, -0.5]
    data.qvel[:] = RNG.uniform(-1, 1, 7)
    d = ra.mj_dynamics(model, data, site="tcp")
    h = 1e-6
    for k in range(7):
        e = np.zeros(7); e[k] = h
        dq = mujoco.MjData(model)
        dq.qpos[:] = data.qpos + e
        mujoco.mj_forward(model, dq)
        dm = mujoco.MjData(model)
        dm.qpos[:] = data.qpos - e
        mujoco.mj_forward(model, dm)
        assert np.allclose(d["J"][:, k], (dq.site("tcp").xpos - dm.site("tcp").xpos) / (2 * h), atol=1e-6)
    # operational-space inertia of a payload at the tool: Lambda_L = Lambda_0 + m I
    L0 = ra.task_inertia(d["M"], d["J"])
    ra.mj_add_point_mass(model, "hand", 2.0, [0, 0, PANDA_TCP := 0.1034])
    d2 = ra.mj_dynamics(model, data, site="tcp")
    L2 = ra.task_inertia(d2["M"], d2["J"])
    assert np.allclose(L2, L0 + 2.0 * np.eye(3), atol=1e-8)
    ra.mj_add_point_mass(model, "hand", -2.0, [0, 0, PANDA_TCP])           # restore (module-scoped model)


def test_mujoco_sl_terms_match_regressor():
    g = generic_arm()
    a = arm()
    model = mujoco.MjModel.from_xml_string(ra.mj_planar_xml(g.l, g.m, g.lc, g.I))
    data = mujoco.MjData(model)
    for _ in range(4):
        q, qd, _ = random_state()
        qdr, qddr = RNG.uniform(-2, 2, 2), RNG.uniform(-3, 3, 2)
        data.qpos[:], data.qvel[:] = q, qd
        mujoco.mj_forward(model, data)
        tau_mj = ra.mj_sl_terms(model, data, qdr, qddr)
        assert np.allclose(tau_mj, a.regressor(q, qd, qdr, qddr) @ a.theta, atol=1e-9)
        assert np.allclose(data.qvel, qd)                                   # state restored


def test_panda_payload_regressor_is_exact(panda):
    """tau_loaded - tau_nominal = Y_p m with Y_p = J^T (J qddr + Jdot qdr - g_vec) for the Slotine-Li terms."""
    model, data = panda
    q = np.array([0.1, -0.4, 0.2, -1.8, 0.3, 1.6, -0.5])
    qd, qdr, qddr = RNG.uniform(-1, 1, 7), RNG.uniform(-1, 1, 7), RNG.uniform(-2, 2, 7)
    data.qpos[:], data.qvel[:] = q, qd
    nominal = ra.mj_sl_terms(model, data, qdr, qddr, ra.mj_dynamics(model, data, site="tcp")["M"])
    d = ra.mj_dynamics(model, data, site="tcp")
    mujoco.mj_forward(model, data)
    # Jdot (matrix) by finite differences of J along qd
    h = 1e-6
    def J_at(qq):
        dd = mujoco.MjData(model); dd.qpos[:] = qq; mujoco.mj_forward(model, dd)
        return ra.mj_dynamics(model, dd, site="tcp")["J"]
    Jd = (J_at(q + h * qd) - J_at(q - h * qd)) / (2 * h)
    ra.mj_add_point_mass(model, "hand", 1.5, [0, 0, ra.PANDA_TCP])
    try:
        mujoco.mj_forward(model, data)
        loaded = ra.mj_sl_terms(model, data, qdr, qddr)
    finally:
        ra.mj_add_point_mass(model, "hand", -1.5, [0, 0, ra.PANDA_TCP])
    Yp = d["J"].T @ (d["J"] @ qddr + Jd @ qdr - np.array([0, 0, -9.81]))
    assert np.allclose(loaded - nominal, 1.5 * Yp, atol=1e-6)


def test_ur5e_loader():
    try:
        model, data = ra.mj_arm("ur5e")
    except Exception as exc:                                # pragma: no cover
        pytest.skip(f"UR5e unavailable: {exc}")
    assert model.nv == 6 and model.nu == 0
    d = ra.mj_dynamics(model, data, site="tcp")
    assert np.linalg.matrix_rank(d["J"]) == 3

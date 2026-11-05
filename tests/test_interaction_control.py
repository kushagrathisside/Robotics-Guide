"""Tests of rgkit.interaction_control (RMC Chapter 5): analytic limits, exactness of the impedance law,
stability boundaries of the sampled force loop, and consistency of the stiffness formulas."""
import numpy as np
import pytest

from rgkit import interaction_control as ic


# --------------------------------------------------------------------------- environments
def test_wall_is_unilateral_and_linear():
    w = ic.Wall(x_w=0.1, k=5e3, b=10.0)
    assert w.force(0.05) == 0.0
    assert w.force(0.11, 0.0) == pytest.approx(50.0)
    assert w.force(0.11, 1.0) == pytest.approx(60.0)
    assert w.force(0.1001, -10.0) == 0.0          # damping cannot make the wall pull


def test_surface_flat_and_sloped_normal():
    s = ic.Surface(y0=0.0, k=1e4, b=0.0, mu=0.0)
    F = s.force_on_tool(np.array([0.0, -0.002]), np.zeros(2))
    assert np.allclose(F, [0.0, 20.0])
    slope = 0.2
    s2 = ic.Surface(profile=lambda x: slope * x, slope=lambda x: slope, k=1e4, b=0.0, mu=0.0)
    t, n = s2.tangent_normal(0.3)
    assert np.allclose(n, np.array([-slope, 1.0]) / np.hypot(1, slope))
    F2 = s2.force_on_tool(np.array([0.3, slope * 0.3 - 0.001]), np.zeros(2))
    assert np.allclose(F2 / np.linalg.norm(F2), n)          # force along the normal (no friction at rest)


def test_surface_friction_opposes_motion():
    s = ic.Surface(k=1e4, b=0.0, mu=0.3)
    F = s.force_on_tool(np.array([0.0, -0.001]), np.array([0.5, 0.0]))
    assert F[0] == pytest.approx(-0.3 * F[1], rel=1e-3) and F[1] > 0


# --------------------------------------------------------------------------- constraints
def _sliding_block():
    nat = {"x": ("f", 0), "y": ("f", 0), "z": ("v", 0), "rx": ("v", 0), "ry": ("v", 0), "rz": ("f", 0)}
    art = {"x": ("v", 0.1), "y": ("v", 0), "z": ("f", 5), "rx": ("f", 0), "ry": ("f", 0), "rz": ("v", 0)}
    return ic.ConstraintFrame("slide", nat, art)


def test_constraint_frame_selection_and_reciprocity():
    cf = _sliding_block()
    assert cf.validate()
    assert np.array_equal(cf.selection_diagonal(), [1, 1, 0, 0, 0, 1])
    twist = np.array([0.1, 0, 0, 0, 0, 0.0])
    wrench = np.array([0, 0, 5.0, 0, 0, 0])
    assert cf.reciprocal(twist, wrench)
    bad = _sliding_block()
    bad.artificial["z"] = ("v", 0.0)
    with pytest.raises(ValueError):
        bad.validate()


def test_rotate_selection_is_projector():
    S = np.diag([1.0, 1.0, 0.0])
    c, s = np.cos(0.4), np.sin(0.4)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    Sw = ic.rotate_selection(S, R)
    assert np.allclose(Sw @ Sw, Sw) and np.allclose(Sw, Sw.T)
    n_world = R[:, 2]
    assert np.allclose(Sw @ n_world, 0)                    # the contact normal is never position controlled


# --------------------------------------------------------------------------- 1-DOF position control on a wall
def test_position_control_steady_force_matches_series_formula():
    k_s, k_e, x_w, x_d = 2.0e3, 1.0e4, 0.0, 0.003
    wall = ic.Wall(x_w, k_e, 40.0)
    ctrl = ic.PositionPD(kp=k_s, kd=2 * np.sqrt(k_s * 1.0) * 1.0, x_d=x_d)
    out = ic.simulate_1dof(ctrl, wall, m=1.0, x0=-0.01, T=3.0)
    expect = ic.contact_force_position_control(k_s, k_e, x_d, x_w)
    assert out["f"][-1] == pytest.approx(expect, rel=1e-3)
    assert expect == pytest.approx(k_s * k_e / (k_s + k_e) * x_d)


def test_stiff_wall_amplifies_position_error():
    k_s = 1e3
    f_soft = ic.contact_force_position_control(k_s, 1e3, 0.002, 0.0)
    f_stiff = ic.contact_force_position_control(k_s, 1e6, 0.002, 0.0)
    assert f_stiff > 1.9 * f_soft and f_stiff == pytest.approx(k_s * 0.002, rel=2e-3)


def test_impedance_1d_against_wall_steady_force():
    k_d, k_e = 500.0, 5.0e3
    wall = ic.Wall(0.0, k_e, 30.0)
    ctrl = ic.Impedance1D(m=1.0, m_d=1.0, b_d=2 * 0.9 * np.sqrt(k_d), k_d=k_d, x_d=0.004, use_force=False)
    out = ic.simulate_1dof(ctrl, wall, m=1.0, x0=-0.002, T=3.0)
    assert out["f"][-1] == pytest.approx(ic.contact_force_impedance(k_d, k_e, 0.004, 0.0), rel=1e-3)


def test_impedance_1d_with_force_feedback_hits_target_inertia():
    """Compare with a linear simulation of m_d e'' + b_d e' + k_d e = -f for a spring wall."""
    m, md, bd, kd, ke = 1.0, 3.0, 40.0, 800.0, 4000.0
    wall = ic.Wall(0.0, ke, 0.0)
    ctrl = ic.Impedance1D(m=m, m_d=md, b_d=bd, k_d=kd, x_d=0.002, use_force=True)
    out = ic.simulate_1dof(ctrl, wall, m=m, x0=0.0005, T=1.0, dt=2e-5, Ts=2e-5)
    # reference: x'' = (-(ke x) - bd x' - kd (x - xd)) / md  (in contact all the time)
    x, v, xs = 0.0005, 0.0, []
    for _ in range(len(out["t"])):
        a = (-ke * x - bd * v - kd * (x - 0.002)) / md
        v += 2e-5 * a
        x += 2e-5 * v
        xs.append(x)
    assert np.max(np.abs(np.array(xs) - out["x"])) < 2e-5


def test_admittance_1d_steady_force():
    k_a, k_e = 800.0, 8.0e3
    wall = ic.Wall(0.0, k_e, 20.0)
    ctrl = ic.Admittance1D(m_a=2.0, b_a=2 * 0.8 * np.sqrt(2.0 * k_a), k_a=k_a, kp=2e4, kd=300.0, x_d=0.003, m=1.0)
    out = ic.simulate_1dof(ctrl, wall, m=1.0, x0=-0.001, T=3.0)
    assert out["f"][-1] == pytest.approx(ic.contact_force_admittance(k_a, 2e4, k_e, 0.003, 0.0), rel=2e-3)


def test_force_control_pi_reaches_desired_force_independently_of_stiffness():
    for k_e in (2e3, 2e4):
        wall = ic.Wall(0.0, k_e, 20.0)
        ctrl = ic.ForceControl1D(f_d=10.0, k_f=0.2, k_i=2.0, k_v=30.0)
        out = ic.simulate_1dof(ctrl, wall, m=1.0, x0=0.0, v0=0.0, T=3.0)
        assert out["f"][-1] == pytest.approx(10.0, rel=5e-3)


def test_force_loop_matrix_matches_time_domain_simulation():
    """The eigenvalue test must agree with a direct simulation (stable vs unstable) in contact."""
    m, k_e, Ts = 1.0, 2.0e4, 1e-3
    for k_f, stable in ((0.5, True), (80.0, False)):
        rho = ic.force_loop_spectral_radius(m, k_e, k_f, k_v=20.0, Ts=Ts, delay=1)
        assert (rho < 1) == stable
        wall = ic.Wall(0.0, k_e, 0.0)
        ctrl = ic.ForceControl1D(f_d=5.0, k_f=k_f, k_v=20.0, feedforward=False)
        out = ic.simulate_1dof(ctrl, wall, m=m, x0=5.0 / k_e, T=1.0, dt=1e-4, Ts=Ts, force_delay=1)
        grew = np.abs(out["x"][-2000:]).max() > 1e-1 or not np.isfinite(out["x"]).all()
        assert (not grew) == stable


def test_delayed_force_feedback_lowers_stability_limit():
    kw = dict(m=1.0, k_e=5e4, k_v=10.0, Ts=1e-3)
    g0 = ic.max_stable_force_gain(delay=0, **kw)
    g1 = ic.max_stable_force_gain(delay=1, **kw)
    g2 = ic.max_stable_force_gain(delay=2, **kw)
    assert g0 > g1 > g2 > 0


def test_stiffer_environment_lowers_stability_limit_and_damping_raises_it():
    kw = dict(m=1.0, Ts=1e-3, delay=1, b_e=5.0)
    soft = ic.max_stable_force_gain(k_e=1e4, k_v=20.0, **kw)
    stiff = ic.max_stable_force_gain(k_e=1e6, k_v=20.0, **kw)
    assert soft > stiff > 0
    # more velocity (damping) feedback -> larger tolerable force gain at the same stiffness
    assert ic.max_stable_force_gain(k_e=1e6, k_v=60.0, **kw) > stiff
    # the velocity-loop form u = k_vi (k_f e - v) is the direct law with k_f' = k_vi k_f and k_v = k_vi
    kf_v = ic.max_stable_force_gain(k_e=1e6, law="velocity_loop", k_vi=20.0, hi=1.0, **{k: x for k, x in kw.items()})
    assert kf_v * 20.0 == pytest.approx(stiff, rel=1e-3)


def test_admittance_spectral_radius_agrees_with_simulation():
    m, k_e, Ts = 1.0, 5.0e3, 1e-3
    for m_a, stable in ((2.0, True), (0.2, False)):
        pars = dict(m_a=m_a, b_a=2 * np.sqrt(m_a * 500.0), k_a=500.0, kp=2e4, kd=200.0)
        rho = ic.admittance_loop_spectral_radius(m, k_e, Ts=Ts, delay=1, **pars)
        assert (rho < 1) == stable, (m_a, rho)
        ctrl = ic.Admittance1D(x_d=0.01, **pars)
        out = ic.simulate_1dof(ctrl, ic.Wall(0.0, k_e, 0.0), m=m, x0=0.004, T=1.5, dt=1e-4, Ts=Ts, force_delay=1)
        settled = np.ptp(out["x"][-1000:]) < 1e-6
        assert settled == stable


# --------------------------------------------------------------------------- planar 2R arm
def _fd_jacobian(arm, q, h=1e-6):
    J = np.zeros((2, 2))
    for i in range(2):
        dq = np.zeros(2)
        dq[i] = h
        J[:, i] = (arm.fk(q + dq) - arm.fk(q - dq)) / (2 * h)
    return J


def test_2r_kinematics_and_dynamics_identities():
    arm = ic.Planar2R()
    q, qd = np.array([0.4, 0.9]), np.array([0.7, -0.3])
    assert np.allclose(arm.jac(q), _fd_jacobian(arm, q), atol=1e-8)
    M = arm.M(q)
    assert np.allclose(M, M.T) and np.all(np.linalg.eigvalsh(M) > 0)
    # Ndot - 2C skew-symmetric property
    h = 1e-6
    Mdot = (arm.M(q + h * qd) - arm.M(q - h * qd)) / (2 * h)
    N = Mdot - 2 * arm.C(q, qd)
    assert np.allclose(N, -N.T, atol=1e-6)
    # Jdot via finite difference
    Jdot = (arm.jac(q + h * qd) - arm.jac(q - h * qd)) / (2 * h)
    assert np.allclose(arm.jac_dot(q, qd), Jdot, atol=1e-6)
    # gravity = gradient of potential energy
    V = lambda q: arm.m1 * arm.g * arm.lc1 * np.sin(q[0]) + arm.m2 * arm.g * (arm.l1 * np.sin(q[0]) + arm.lc2 * np.sin(q[0] + q[1]))
    g_fd = np.array([(V(q + h * e) - V(q - h * e)) / (2 * h) for e in np.eye(2)])
    assert np.allclose(arm.G(q), g_fd, atol=1e-5)
    # inverse kinematics round trip
    p = arm.fk(q)
    assert np.allclose(arm.fk(arm.ik(p, +1)), p) and np.allclose(arm.fk(arm.ik(p, -1)), p)


def test_task_space_dynamics_identity():
    arm = ic.Planar2R()
    q, qd = np.array([0.5, 1.1]), np.array([0.6, 0.4])
    Lam, mu, p = arm.task_bias(q, qd)
    F = np.array([3.0, -2.0])
    tau = arm.jac(q).T @ F + arm.bias(q, qd) - arm.cj * 0     # tau = J^T F  (ignoring joint friction)
    qdd = np.linalg.solve(arm.M(q), arm.jac(q).T @ F - arm.bias(q, qd))
    xdd = arm.jac(q) @ qdd + arm.jac_dot(q, qd) @ qd
    assert np.allclose(Lam @ xdd + mu + p, F)


def test_full_impedance_control_makes_the_arm_a_linear_mass_spring_damper():
    """The tip obeys M_d e'' + B_d e' + K_d e = F_ext exactly, whatever the configuration (no friction)."""
    arm = ic.Planar2R(cj=0.0)
    M_d, B_d, K_d = np.diag([2.0, 3.0]), np.diag([30.0, 40.0]), np.diag([400.0, 900.0])
    x_d = np.array([0.45, 0.20])
    traj = lambda t: (x_d, np.zeros(2), np.zeros(2))
    ctrl = ic.OperationalSpace2R(arm, traj, mode="impedance", M_d=M_d, Kd=B_d, Kp=K_d)
    Fpush = lambda t: np.array([5.0, -8.0]) * (t > 0.05)
    q0 = arm.ik(x_d + np.array([0.01, -0.02]))
    T, dt = 1.5, 1e-4
    rec = ic.simulate_2r(arm, ctrl, None, q0, T=T, dt=dt, Ts=dt, F_ext_extra=Fpush)
    # linear reference simulation
    e = arm.fk(q0) - x_d
    ed = arm.jac(q0) @ np.zeros(2)
    e_ref = []
    for k in range(len(rec["t"])):
        a = np.linalg.solve(M_d, Fpush(k * dt) - B_d @ ed - K_d @ e)
        # same semi-implicit scheme is not required: compare with loose tolerance
        ed = ed + dt * a
        e = e + dt * ed
        e_ref.append(e.copy())
    e_ref = np.array(e_ref)
    err = np.abs((rec["p"] - x_d) - e_ref).max()
    assert err < 2e-4
    assert np.allclose(rec["p"][-1] - x_d, np.linalg.solve(K_d, [5.0, -8.0]), atol=2e-4)   # static compliance


def test_hybrid_control_force_regulation_on_flat_surface():
    arm = ic.Planar2R()
    surf = ic.Surface(y0=0.30, k=2.0e4, b=20.0, mu=0.2)           # surface at y = 0.30, robot above it
    q0 = arm.ik(np.array([0.35, 0.3005]), -1)
    traj = ic.line_trajectory([0.35, 0.3005], [0.35, 0.3005], 1.0)
    # contact normal is +y; the robot exerts a force -y on the surface, so f (robot on environment) is [0, -f]
    S = np.diag([1.0, 0.0])
    ctrl = ic.OperationalSpace2R(arm, traj, mode="hybrid", S=S, f_d=np.array([0.0, -15.0]),
                                 Kf=0.0, Ki=20.0, Kp=np.diag([1500.0, 1500.0]), Kd=np.diag([80.0, 80.0]))
    rec = ic.simulate_2r(arm, ctrl, surf, q0, T=3.0, dt=5e-4, Ts=1e-3)
    assert rec["F"][-1][1] == pytest.approx(15.0, rel=5e-3)


# --------------------------------------------------------------------------- stiffness
def test_cartesian_stiffness_roundtrip_and_symmetry():
    arm = ic.Planar2R()
    q = np.array([0.3, 1.2])
    J = arm.jac(q)
    Kq = np.diag([800.0, 500.0])
    Kc = ic.cartesian_stiffness(J, Kq)
    Cc = ic.cartesian_compliance(J, Kq)
    assert np.allclose(Kc, Kc.T) and np.allclose(Kc @ Cc, np.eye(2))
    # joint-space check: the torque of a small tip displacement equals J^T Kc dx = Kq dq
    dq = np.array([1e-4, -2e-4])
    dx = J @ dq
    assert np.allclose(J.T @ Kc @ dx, Kq @ dq)


def test_ellipse_points_satisfy_quadratic_form():
    A = np.array([[4.0, 1.0], [1.0, 2.0]])
    P = ic.ellipse_points(A, center=(1.0, 2.0))
    d = P - np.array([1.0, 2.0])
    assert np.allclose(np.einsum("ni,ij,nj->n", d, A, d), 1.0)


# --------------------------------------------------------------------------- series elastic actuation
def test_rigid_impact_peak_force_formula():
    wall = ic.Wall(0.0, 1.0e5, 0.0)
    _, _, peak = ic.simulate_sea_impact(0.1, 1.0, 0.5, 5e3, wall, series=False)
    assert peak == pytest.approx(0.1 * np.sqrt(1.0e5 * 1.5), rel=2e-3)


def test_series_elasticity_reduces_peak_force():
    wall = ic.Wall(0.0, 1.0e5, 0.0)
    _, _, p_rigid = ic.simulate_sea_impact(0.1, 1.0, 0.5, 5e3, wall, series=False)
    _, _, p_sea = ic.simulate_sea_impact(0.1, 1.0, 0.05, 5e3, wall, series=True)
    assert p_sea < 0.5 * p_rigid


def test_sea_force_loop_stability_limited_by_spring_not_wall():
    kw = dict(m_motor=1.0, m_load=0.05, k_s=2e3, k_f=0.5, k_v=30.0, Ts=1e-3, delay=1, b_e=5.0)
    rho_soft = ic.sea_force_loop_spectral_radius(k_e=1e3, **kw)
    rho_rigid = ic.sea_force_loop_spectral_radius(k_e=1e7, **kw)
    assert rho_soft < 1 and rho_rigid < 1                    # stays stable even against a (nearly) rigid wall
    # the same gain applied to a rigid actuator (loop sees k_e directly) is unstable on a stiff wall
    assert ic.force_loop_spectral_radius(1.0, 1e6, 0.5, k_v=30.0, Ts=1e-3, delay=1, b_e=5.0) > 1


def test_guarded_move_then_force_regulation():
    arm = ic.Planar2R()
    surf = ic.Surface(y0=0.30, k=2.0e4, b=20.0, mu=0.2)
    q0 = arm.ik(np.array([0.35, 0.32]), -1)
    traj = ic.line_trajectory([0.35, 0.32], [0.35, 0.32], 1.0)
    ctrl = ic.OperationalSpace2R(arm, traj, mode="hybrid", S=np.diag([1.0, 0.0]), f_d=np.array([0.0, -10.0]),
                                 Kf=0.2, Ki=20.0, Kp=np.diag([1500.0, 1500.0]), Kd=np.diag([80.0, 80.0]),
                                 approach_v=0.03, f_thresh=0.5)
    rec = ic.simulate_2r(arm, ctrl, surf, q0, T=3.0, dt=5e-4, Ts=1e-3)
    assert rec["F"][-1][1] == pytest.approx(10.0, rel=1e-2)
    assert np.abs(rec["F"][:, 1]).max() < 25.0               # no violent impact

"""Tests for rgkit.mobile: kinematics, constraints, controllability tools and controllers."""
import numpy as np
import sympy as sp

from rgkit import mobile as mb

RNG = np.random.default_rng(1)


def test_unicycle_step_matches_numerical_integration_and_arc_formulas():
    q0 = np.array([0.3, -0.2, 0.7])
    v, w, T = 0.8, 0.5, 2.0
    q = mb.unicycle_step(q0, v, w, T)
    # closed form for a circular arc of radius v/w
    R = v / w
    assert np.allclose(q, [q0[0] + R * (np.sin(q0[2] + w * T) - np.sin(q0[2])),
                           q0[1] - R * (np.cos(q0[2] + w * T) - np.cos(q0[2])), q0[2] + w * T])
    # tiny steps (Euler) approach it
    qe = q0.copy()
    n = 200000
    for _ in range(n):
        qe = qe + (T / n) * mb.unicycle_deriv(qe, v, w)
    assert np.allclose(qe, q, atol=1e-4)
    # straight line limit
    assert np.allclose(mb.unicycle_step(q0, v, 0.0, T), [q0[0] + v * T * np.cos(q0[2]), q0[1] + v * T * np.sin(q0[2]), q0[2]])


def test_differential_drive_conversions_are_inverse():
    r, L = 0.033, 0.16
    for _ in range(5):
        v, w = RNG.normal(size=2)
        wr, wl = mb.body_to_diffdrive(v, w, r, L)
        assert np.allclose(mb.diffdrive_to_body(wr, wl, r, L), (v, w))
    # both wheels equal -> straight; opposite -> pure rotation
    assert np.allclose(mb.diffdrive_to_body(5.0, 5.0, r, L), (r * 5.0, 0.0))
    assert np.allclose(mb.diffdrive_to_body(5.0, -5.0, r, L), (0.0, 2 * r * 5.0 / L))


def test_pfaffian_constraint_holds_for_every_input():
    for _ in range(10):
        q = RNG.normal(size=3)
        v, w = RNG.normal(size=2)
        assert abs(mb.pfaffian_residual(q, mb.unicycle_deriv(q, v, w))) < 1e-14
    # a sideways velocity violates it
    q = np.array([0, 0, 0.0])
    assert abs(mb.pfaffian_residual(q, np.array([0.0, 1.0, 0.0]))) == 1.0


def test_lie_bracket_and_commutator_displacement():
    gv = lambda q: np.array([np.cos(q[2]), np.sin(q[2]), 0.0])      # noqa: E731
    gw = lambda q: np.array([0.0, 0.0, 1.0])                         # noqa: E731
    q = np.array([0.2, -0.4, 0.9])
    br = mb.lie_bracket(gv, gw, q)
    assert np.allclose(br, [np.sin(q[2]), -np.cos(q[2]), 0.0], atol=1e-8)
    # the three vector fields span R^3
    assert np.linalg.matrix_rank(np.column_stack([gv(q), gw(q), br])) == 3
    # exact displacement of the four-flow manoeuvre at theta = 0: eps (1 - cos eps, -sin eps, 0)
    for eps in (0.3, 0.1):
        d = mb.commutator_displacement([0, 0, 0], eps)
        assert np.allclose(d, [eps * (1 - np.cos(eps)), -eps * np.sin(eps), 0.0], atol=1e-12)
    d = mb.commutator_displacement(q, 0.01)
    assert np.allclose(d / 0.01 ** 2, br, atol=2e-2)


def test_chained_form_transformation_is_exact():
    # symbolic check: z1' = w1, z2' = w2, z3' = z2 w1
    x, y, th, v, w = sp.symbols("x y theta v omega")
    xd, yd, thd = v * sp.cos(th), v * sp.sin(th), w
    z2 = x * sp.cos(th) + y * sp.sin(th)
    z3 = x * sp.sin(th) - y * sp.cos(th)
    dz2 = sp.diff(z2, x) * xd + sp.diff(z2, y) * yd + sp.diff(z2, th) * thd
    dz3 = sp.diff(z3, x) * xd + sp.diff(z3, y) * yd + sp.diff(z3, th) * thd
    assert sp.simplify(dz2 - (v - z3 * w)) == 0
    assert sp.simplify(dz3 - z2 * w) == 0
    q = RNG.normal(size=3)
    assert np.allclose(mb.chained_to_unicycle(mb.unicycle_to_chained(q)), q)


def test_sinusoidal_steering_shifts_sideways():
    a, b, dt = 1.0, 0.5, 5e-4
    q = np.zeros(3)
    for k in range(int(2 * np.pi / dt)):
        v, w = mb.sinusoidal_parking_input(k * dt, q, a, b)
        q = mb.unicycle_step(q, v, w, dt)
    assert np.allclose(q, [0.0, -np.pi * a * b, 0.0], atol=2e-3)


def test_polar_coordinates_and_closed_loop_kinematics():
    goal = np.array([1.0, 2.0, 0.5])
    q = np.array([-1.0, 0.5, 0.3])
    rho, alpha, beta = mb.polar_coordinates(q, goal)
    assert np.isclose(rho, np.hypot(*(q[:2] - goal[:2])))
    # the derivative relations rho' = -v cos(alpha), alpha' = v sin(alpha)/rho - omega, beta' = -v sin(alpha)/rho
    v, w, h = 0.7, 0.2, 1e-6
    q2 = q + h * mb.unicycle_deriv(q, v, w)
    r2, a2, b2 = mb.polar_coordinates(q2, goal)
    assert np.isclose((r2 - rho) / h, -v * np.cos(alpha), atol=1e-4)
    assert np.isclose((a2 - alpha) / h, v * np.sin(alpha) / rho - w, atol=1e-4)
    assert np.isclose((b2 - beta) / h, -v * np.sin(alpha) / rho, atol=1e-4)


def test_polar_pose_control_reaches_the_goal_from_all_directions():
    goal = np.array([0.5, -0.5, 1.0])
    for a in np.linspace(-np.pi, np.pi, 9)[:-1]:
        for th in (-2.5, 0.0, 2.5):
            q0 = goal + np.array([3 * np.cos(a), 3 * np.sin(a), 0]) + np.array([0, 0, th])
            # stop well before rho reaches rounding level: the polar coordinates are singular at rho = 0
            t, Q, _ = mb.simulate_unicycle(lambda t, q: mb.polar_pose_control(q, goal, 1.0, 3.0, -1.0), q0, 0.01, 12.0)
            assert np.hypot(*(Q[-1, :2] - goal[:2])) < 0.02 and abs(mb.wrap_pi(Q[-1, 2] - goal[2])) < 0.05


def test_lyapunov_pose_control_decreases_V():
    goal = np.array([0.0, 0.0, 0.0])
    q0 = np.array([-3.0, 1.0, 0.4])
    t, Q, _ = mb.simulate_unicycle(lambda t, q: mb.lyapunov_pose_control(q, goal, 1.0, 3.0, 1.0), q0, 0.005, 40.0)
    V = [0.5 * (r ** 2 + a ** 2 + 1.0 * b ** 2) for r, a, b in (mb.polar_coordinates(q, goal) for q in Q)]
    assert np.all(np.diff(V) < 1e-9)
    assert np.hypot(*Q[-1, :2]) < 1e-2 and abs(Q[-1, 2]) < 1e-2


def test_tracking_control_converges_and_kanayama_V_decreases():
    ref = mb.circle_reference(1.0, 0.5)
    q0 = ref(0)[0] + np.array([0.4, -0.3, 0.6])
    ky = 8.0
    t, Q, _ = mb.simulate_unicycle(lambda t, q: mb.tracking_control(q, *ref(t), 2.0, ky, 3.0), q0, 0.005, 25.0)
    V = np.array([0.5 * (mb.tracking_error(Q[k], ref(t[k])[0])[:2] ** 2).sum()
                  + (1 - np.cos(mb.tracking_error(Q[k], ref(t[k])[0])[2])) / ky for k in range(len(t))])
    assert np.all(np.diff(V) < 1e-9)
    assert np.linalg.norm(mb.tracking_error(Q[-1], ref(t[-1])[0])) < 1e-6


def test_figure_eight_reference_is_consistent():
    ref = mb.figure_eight_reference(1.5, 24.0)
    h = 1e-5
    for t in (0.7, 5.0, 11.0):
        pose, v, w = ref(t)
        p2, _, _ = ref(t + h)
        p0, _, _ = ref(t - h)
        vel = (p2[:2] - p0[:2]) / (2 * h)
        assert np.isclose(np.hypot(*vel), v, rtol=1e-6)
        assert np.isclose(mb.wrap_pi(p2[2] - p0[2]) / (2 * h), w, rtol=1e-4)


def test_potential_field_reaches_goal_avoiding_obstacles():
    obst = [(2.0, 0.1, 0.5), (3.5, -0.6, 0.5)]
    goal = np.array([5.0, 0.0])

    def ctrl(t, q):
        f = mb.potential_field_velocity(q[:2], goal, obst, 1.0, 0.8, 1.0, 1.5)
        e = mb.wrap_pi(np.arctan2(f[1], f[0]) - q[2])
        return np.linalg.norm(f) * np.cos(e) * (abs(e) < np.pi / 2), 4 * e

    t, Q, _ = mb.simulate_unicycle(ctrl, np.zeros(3), 0.01, 30.0)
    clear = min(np.hypot(Q[:, 0] - cx, Q[:, 1] - cy).min() - r for cx, cy, r in obst)
    assert clear > 0.2 and np.hypot(*(Q[-1, :2] - goal)) < 0.05


def test_wheel_speed_limit_preserves_curvature():
    v, w = mb.limit_wheel_speeds(2.0, 1.0, 0.05, 0.2, 10.0)
    wr, wl = mb.body_to_diffdrive(v, w, 0.05, 0.2)
    assert max(abs(wr), abs(wl)) <= 10.0 + 1e-9
    assert np.isclose(w / v, 1.0 / 2.0)

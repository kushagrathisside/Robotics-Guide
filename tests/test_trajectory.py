"""Tests for rgkit.trajectory: closed forms, continuity, limits and agreement with SciPy."""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from rgkit import trajectory as tj
from rgkit import transforms as tf

deg = np.deg2rad


def fine(traj, n=4001):
    return np.linspace(traj.t0, traj.t1, n)


# ----------------------------------------------------------------------------- polynomials
def test_cubic_rest_to_rest_closed_form():
    q0, qf, T = deg(20.0), deg(110.0), 2.5
    tr = tj.cubic(q0, qf, T)
    D = qf - q0
    c = tr.segment_coeffs(0)[0]
    assert np.allclose(c, [q0, 0, 3 * D / T ** 2, -2 * D / T ** 3])
    t = fine(tr)
    assert np.isclose(tr.vel(t).max(), 1.5 * D / T)
    assert np.isclose(np.abs(tr.acc(t)).max(), 6 * D / T ** 2)
    assert np.isclose(tr.vel(T / 2), 1.5 * D / T)


def test_cubic_end_velocities_craig_formula():
    q0, qf, T, v0, vf = 0.3, 1.9, 2.0, 0.4, -0.2
    c = tj.cubic(q0, qf, T, v0, vf).segment_coeffs(0)[0]
    D = qf - q0
    assert np.isclose(c[2], 3 * D / T ** 2 - 2 * v0 / T - vf / T)
    assert np.isclose(c[3], -2 * D / T ** 3 + (vf + v0) / T ** 2)


def test_quintic_closed_form_and_peaks():
    D, T = 1.2, 2.0
    tr = tj.quintic(0.0, D, T)
    tau = np.linspace(0, 1, 11)
    assert np.allclose(tr.pos(tau * T), D * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5))
    t = fine(tr, 20001)
    assert np.isclose(tr.vel(t).max(), 15 / 8 * D / T, rtol=1e-6)
    assert np.isclose(np.abs(tr.acc(t)).max(), 10 / np.sqrt(3) * D / T ** 2, rtol=1e-6)
    assert np.isclose(np.abs(tr.jerk(t)).max(), 60 * D / T ** 3, rtol=1e-6)
    assert np.isclose(tr.acc(0.0), 0) and np.isclose(tr.acc(T), 0)


def test_septic_zero_jerk_at_ends_and_coefficients():
    tr = tj.septic(0.0, 1.0, 1.0)
    tau = np.linspace(0, 1, 9)
    assert np.allclose(tr.pos(tau), 35 * tau ** 4 - 84 * tau ** 5 + 70 * tau ** 6 - 20 * tau ** 7)
    assert np.isclose(tr.jerk(0.0), 0) and np.isclose(tr.jerk(1.0), 0)


def test_multijoint_polynomials_independent():
    q0, qf = np.array([0.0, 1.0, -1.0]), np.array([1.0, 0.0, 2.0])
    tr = tj.quintic(q0, qf, 3.0)
    assert tr.n == 3 and not tr.scalar
    t = np.linspace(0, 3, 7)
    assert tr.pos(t).shape == (7, 3)
    assert np.allclose(tr.pos(0.0), q0) and np.allclose(tr.pos(3.0), qf)
    for j in range(3):
        assert np.allclose(tr.pos(t)[:, j], tj.quintic(q0[j], qf[j], 3.0).pos(t))


def test_hold_outside_interval():
    tr = tj.cubic(0.0, 1.0, 2.0)
    assert tr.pos(-1.0) == 0.0 and tr.pos(5.0) == 1.0
    assert tr.vel(-1.0) == 0.0 and tr.vel(5.0) == 0.0 and tr.acc(5.0) == 0.0


def test_minimum_jerk_property_of_quintic():
    """No admissible perturbation lowers the integral of squared jerk below the quintic's 720 D^2 / T^5."""
    D, T = 1.0, 1.0
    base = tj.quintic(0.0, D, T)
    t = np.linspace(0, T, 4001)
    J0 = np.trapezoid(base.jerk(t) ** 2, t)
    assert np.isclose(J0, 720 * D ** 2 / T ** 5, rtol=1e-4)
    from numpy.polynomial import Polynomial as P
    rng = np.random.default_rng(1)
    Q = P(base.segment_coeffs(0)[0])
    for _ in range(5):
        g = P(rng.normal(size=3))
        p = (P([0, 1]) * P([T, -1])) ** 3 * g                # p, p', p'' vanish at both ends
        jerk = (Q + 0.3 * p).deriv(3)
        assert np.trapezoid(jerk(t) ** 2, t) > J0


# ----------------------------------------------------------------------------- via points
def test_heuristic_velocities_rules():
    pts = [0.0, 1.0, 3.0, 2.0, 2.5]
    t = [0.0, 1.0, 2.0, 3.0, 4.0]
    v = tj.heuristic_velocities(pts, t)
    s = np.diff(pts) / np.diff(t)
    assert np.allclose(v, [0, 0.5 * (s[0] + s[1]), 0.0, 0.0, 0])        # slope changes sign at points 2 and 3


def test_via_cubic_c1_and_interpolation():
    pts = [0.0, 1.0, 3.0, 2.0, 2.5]
    t = [0.0, 1.0, 2.5, 3.0, 4.0]
    tr = tj.via_cubic(pts, t)
    assert np.allclose(tr.pos(np.array(t)), pts)
    for tk in t[1:-1]:
        assert np.isclose(tr.pos(tk - 1e-9), tr.pos(tk + 1e-9), atol=1e-7)
        assert np.isclose(tr.vel(tk - 1e-9), tr.vel(tk + 1e-9), atol=1e-7)


def test_spline_matches_scipy_clamped_and_natural():
    pts = np.array([[0.0, 1.0], [1.0, -0.5], [2.5, 0.5], [2.0, 2.0], [3.5, 1.0]])
    t = np.array([0.0, 0.8, 1.9, 3.1, 4.0])
    for bc, scipy_bc in (("clamped", "clamped"), ("natural", "natural")):
        tr = tj.via_cubic(pts, t, velocities="spline" if bc == "clamped" else "natural")
        cs = CubicSpline(t, pts, bc_type=scipy_bc)
        tt = np.linspace(0, 4, 101)
        assert np.allclose(tr.pos(tt), cs(tt), atol=1e-10)
        assert np.allclose(tr.vel(tt), cs(tt, 1), atol=1e-9)
        assert np.allclose(tr.acc(tt), cs(tt, 2), atol=1e-8)


def test_spline_is_c2_but_heuristic_is_not():
    pts = [0.0, 1.0, 3.0, 4.0]
    t = [0.0, 1.0, 2.0, 4.0]
    for tk in t[1:-1]:
        s = tj.via_cubic(pts, t, "spline")
        assert np.isclose(s.acc(tk - 1e-7), s.acc(tk + 1e-7), atol=1e-4)
    h = tj.via_cubic(pts, t, "heuristic")
    assert abs(h.acc(2.0 - 1e-7) - h.acc(2.0 + 1e-7)) > 1e-3


def test_via_quintic_c2_by_construction():
    pts = [0.0, 1.0, 3.0, 2.0]
    t = [0.0, 1.0, 2.0, 3.5]
    v = tj.heuristic_velocities(pts, t)
    tr = tj.via_quintic(pts, t, v)
    for tk in t[1:-1]:
        assert np.isclose(tr.acc(tk - 1e-7), tr.acc(tk + 1e-7), atol=1e-4)
        assert np.isclose(tr.vel(tk), v[t.index(tk)])


def test_stop_at_every_via_point():
    tr = tj.via_stop([0.0, 1.0, 0.5], [0.0, 1.0, 2.0])
    assert np.isclose(tr.vel(1.0), 0.0)


def test_chord_times():
    t = tj.chord_times([[0, 0], [3, 4], [3, 6]], speed=2.0)
    assert np.allclose(t, [0, 2.5, 3.5])


# ----------------------------------------------------------------------------- LSPB
def test_lspb_parameter_relations_and_continuity():
    q0, qf, T = 0.2, 1.7, 3.0
    D = qf - q0
    for kw in (dict(a=1.2), dict(v=0.7), dict(tb=0.6)):
        tr = tj.lspb(q0, qf, T, **kw)
        i = tr.info
        assert np.isclose(i["v"] * (T - i["tb"]), D)
        assert np.isclose(i["a"] * i["tb"], i["v"])
        assert np.isclose(tr.pos(T), qf) and np.isclose(tr.vel(0.0), 0) and np.isclose(tr.vel(T), 0)
        for tk in tr.times[1:-1]:
            assert np.isclose(tr.pos(tk - 1e-9), tr.pos(tk + 1e-9), atol=1e-7)
            assert np.isclose(tr.vel(tk - 1e-9), tr.vel(tk + 1e-9), atol=1e-7)
        assert np.isclose(tr.pos(T / 2), q0 + D / 2)         # symmetric profile


def test_lspb_craig_blend_time_formula():
    q0, qf, T, a = 0.0, 1.0, 2.0, 4.0
    tr = tj.lspb(q0, qf, T, a=a)
    tb = T / 2 - np.sqrt(a ** 2 * T ** 2 - 4 * a * (qf - q0)) / (2 * a)
    assert np.isclose(tr.info["tb"], tb)


def test_lspb_validation():
    with pytest.raises(ValueError):
        tj.lspb(0, 1, 2, a=0.5)                              # needs a >= 4D/T^2 = 1
    with pytest.raises(ValueError):
        tj.lspb(0, 1, 2, v=0.4)                              # v must exceed D/T = 0.5
    with pytest.raises(ValueError):
        tj.lspb(0, 1, 2, v=1.2)                              # v must not exceed 2D/T = 1
    with pytest.raises(ValueError):
        tj.lspb(0, 1, 2, tb=1.2)
    with pytest.raises(ValueError):
        tj.lspb(0, 1, 2)
    assert tj.lspb(0, 1, 2, a=1.0).info["kind"] == "triangle"      # boundary: a = 4D/T^2
    assert tj.lspb(0, 1, 2, v=1.0).info["kind"] == "triangle"      # boundary: v = 2D/T


def test_lspb_negative_motion_and_vector():
    tr = tj.lspb([1.0, 0.0], [0.0, 0.5], 2.0, tb=0.5)
    assert np.allclose(tr.pos(2.0), [0.0, 0.5])
    v = tr.vel(1.0)
    assert np.allclose(v, [-1 / 1.5, 0.5 / 1.5])


def test_min_time_trapezoid_and_bang_bang():
    r = tj.min_time_trapezoid(2.0, vmax=1.0, amax=1.0)               # D >= v^2/a: trapezoid
    assert r["kind"] == "trapezoid" and np.isclose(r["T"], 3.0)
    r = tj.min_time_trapezoid(0.25, vmax=1.0, amax=1.0)              # triangle
    assert r["kind"] == "triangle" and np.isclose(r["T"], 2 * np.sqrt(0.25))
    bb = tj.bang_bang(0.0, 1.0, 2.0)
    assert np.isclose(bb.duration, 2 * np.sqrt(0.5))
    assert np.isclose(np.abs(bb.acc(np.linspace(0, bb.duration, 7))).max(), 2.0)


def test_min_time_is_not_beaten_by_any_feasible_lspb():
    D, vmax, amax = 1.0, 0.8, 1.5
    Tmin = tj.min_time_trapezoid(D, vmax, amax)["T"]
    for T in np.linspace(0.9 * Tmin, 1.5 * Tmin, 41):
        try:
            tr = tj.lspb(0, D, T, tb=min(T / 2, vmax / amax))
        except ValueError:
            continue
        feasible = tr.info["v"] <= vmax * (1 + 1e-9) and tr.info["a"] <= amax * (1 + 1e-9)
        assert (not feasible) or T >= Tmin - 1e-9


def test_lspb_via_properties():
    q = np.array([0.0, 1.0, 0.4, 2.0])
    td = np.array([1.5, 1.5, 1.5])
    tr = tj.lspb_via(q, td, [3.0, 3.0, 3.0, 3.0])
    t = np.linspace(0, td.sum(), 4001)
    assert np.isclose(tr.pos(0.0), q[0]) and np.isclose(tr.pos(td.sum()), q[-1], atol=1e-9)
    assert np.isclose(tr.vel(0.0), 0.0) and np.isclose(tr.vel(td.sum()), 0.0, atol=1e-9)
    assert np.isclose(tr.info["final"], q[-1])
    # position and velocity continuous
    for tk in tr.times[1:-1]:
        assert np.isclose(tr.pos(tk - 1e-9), tr.pos(tk + 1e-9), atol=1e-6)
        assert np.isclose(tr.vel(tk - 1e-9), tr.vel(tk + 1e-9), atol=1e-6)
    # acceleration magnitude never exceeds the blend accelerations
    assert np.abs(tr.acc(t)).max() <= 3.0 + 1e-9
    # interior via points are cut (not reached) but approached to within the corner-cutting distance
    for tv, qv in zip(np.cumsum(td)[:-1], q[1:-1]):
        assert 1e-4 < abs(tr.pos(tv) - qv) < 0.3


def test_lspb_via_infeasible_raises():
    with pytest.raises(ValueError):
        tj.lspb_via([0.0, 1.0, 0.0], [1.0, 1.0], [0.5, 0.5, 0.5])


# ----------------------------------------------------------------------------- S-curve and coordination
@pytest.mark.parametrize("D,vmax,amax,jmax", [(5.0, 1.0, 2.0, 10.0),       # all limits reached
                                              (0.5, 1.0, 2.0, 10.0),       # no cruise
                                              (0.1, 1.0, 2.0, 10.0),       # amax not reached
                                              (3.0, 0.1, 2.0, 10.0)])      # vmax below amax^2/jmax
def test_scurve_respects_limits_and_reaches_target(D, vmax, amax, jmax):
    tr = tj.scurve(0.0, D, vmax, amax, jmax)
    t = fine(tr, 20001)
    assert np.isclose(tr.pos(tr.t1), D) and np.isclose(tr.info["final"], D)
    assert np.abs(tr.vel(t)).max() <= vmax * (1 + 1e-9)
    assert np.abs(tr.acc(t)).max() <= amax * (1 + 1e-9)
    assert np.abs(tr.jerk(t)).max() <= jmax * (1 + 1e-9)
    assert np.isclose(tr.vel(tr.t1), 0.0, atol=1e-9) and np.isclose(tr.acc(tr.t1), 0.0, atol=1e-9)
    for tk in tr.times[1:-1]:                                              # C^2 at the knots
        assert np.isclose(tr.acc(tk - 1e-9), tr.acc(tk + 1e-9), atol=1e-6)


def test_scurve_negative_direction():
    tr = tj.scurve(1.0, -2.0, 1.0, 2.0, 10.0)
    assert np.isclose(tr.pos(tr.t1), -2.0) and tr.vel(tr.t1 / 2) < 0


def test_coordinated_ptp_synchronises_joints():
    q0, qf = np.zeros(3), np.array([1.0, -0.4, 0.2])
    vmax, amax = np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])
    for prof in ("trapezoid", "cubic", "quintic", "scurve"):
        tr = tj.coordinated_ptp(q0, qf, vmax, amax, prof, jmax=20.0)
        rep = tj.check_limits(tr, v_lim=vmax, a_lim=amax)
        assert rep["ok"], prof
        assert np.allclose(tr.pos(tr.t1), qf)
    tr = tj.coordinated_ptp(q0, qf, vmax, amax, "trapezoid")
    assert np.isclose(tr.t1, tj.min_time_trapezoid(1.0, 1.0, 2.0)["T"])    # joint 1 (largest move) sets the time
    assert np.isclose(np.abs(tr.vel(tr.t1 / 2)).max(), 1.0)                # and is exactly at its limit


# ----------------------------------------------------------------------------- Cartesian paths
def test_geodesic_equals_slerp_and_has_constant_angular_speed():
    R0 = tf.rotz(deg(20)) @ tf.rotx(deg(10))
    R1 = tf.roty(deg(70)) @ tf.rotz(deg(-30))
    q0, q1 = tf.rot_to_quat(R0), tf.rot_to_quat(R1)
    for s in (0.0, 0.25, 0.6, 1.0):
        assert np.allclose(tj.rotation_geodesic(R0, R1, np.array([s]))[0], tf.quat_to_rot(tf.slerp(q0, q1, s)), atol=1e-10)
    s = np.linspace(0, 1, 201)
    Rs = tj.rotation_geodesic(R0, R1, s)
    _, th = tf.axis_angle_from_rot(R0.T @ R1)
    step_angles = [tf.axis_angle_from_rot(Rs[i].T @ Rs[i + 1])[1] for i in range(200)]
    assert np.allclose(step_angles, th / 200, atol=1e-9)
    assert all(tf.is_rotation(R) for R in Rs)


def test_matrix_lerp_is_not_a_rotation_but_euler_interp_is():
    R0, R1 = np.eye(3), tf.rotz(deg(120))
    assert not tf.is_rotation(tj.matrix_lerp(R0, R1, 0.5))
    assert tf.is_rotation(tj.euler_interp(R0, R1, np.array([0.5]))[0])
    assert np.allclose(tj.euler_interp(R0, R1, np.array([0.0, 1.0])), [R0, R1], atol=1e-10)


def test_euler_interpolation_is_not_the_geodesic():
    R0 = np.eye(3)
    R1 = tf.rpy_to_rot(deg(80), deg(10), deg(70))
    mid_e = tj.euler_interp(R0, R1, np.array([0.5]))[0]
    mid_g = tj.rotation_geodesic(R0, R1, np.array([0.5]))[0]
    ang_e = tf.axis_angle_from_rot(mid_e)[1]
    ang_g = tf.axis_angle_from_rot(mid_g)[1]
    assert abs(ang_e - ang_g) > 1e-3


def test_cartesian_line_twist_matches_finite_differences():
    T0 = tf.make_T(tf.rotz(deg(10)), [0.3, 0.1, 0.4])
    T1 = tf.make_T(tf.rotx(deg(70)) @ tf.rotz(deg(40)), [0.5, -0.2, 0.6])
    line = tj.CartesianLine(T0, T1, tj.quintic(0.0, 1.0, 2.0))
    assert np.allclose(line.pose(0.0), T0) and np.allclose(line.pose(2.0), T1)
    t, h = 0.7, 1e-6
    v, w = line.twist(t)
    dp = (line.pose(t + h)[:3, 3] - line.pose(t - h)[:3, 3]) / (2 * h)
    assert np.allclose(v, dp, atol=1e-6)
    dR = (line.pose(t + h)[:3, :3] - line.pose(t - h)[:3, :3]) / (2 * h)
    W = dR @ line.pose(t)[:3, :3].T
    assert np.allclose(tf.unskew(0.5 * (W - W.T)), w, atol=1e-6)


def _planar2r():
    L1, L2 = 1.0, 0.8

    def fk(q):
        x = L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1])
        y = L1 * np.sin(q[0]) + L2 * np.sin(q[0] + q[1])
        T = np.eye(4)
        T[:3, :3] = tf.rotz(q[0] + q[1])
        T[:3, 3] = [x, y, 0.0]
        return T

    def jac(q):
        s1, s12, c1, c12 = np.sin(q[0]), np.sin(q[0] + q[1]), np.cos(q[0]), np.cos(q[0] + q[1])
        J = np.zeros((6, 2))
        J[0] = [-L1 * s1 - L2 * s12, -L2 * s12]
        J[1] = [L1 * c1 + L2 * c12, L2 * c12]
        J[5] = [1.0, 1.0]
        return J
    return fk, jac


def test_dls_tracking_converges_and_is_bounded_at_singularity():
    fk, jac = _planar2r()
    q0 = np.array([0.4, 1.0])
    targets = [fk(q0 + np.array([0.05 * k, -0.04 * k])) for k in range(1, 6)]
    Q, res = tj.track_poses(q0, targets, fk, jac)
    assert res.max() < 1e-8
    J = jac(np.array([0.3, 0.0]))                                       # stretched arm: singular
    dq = tj.dls_step(J[:2], np.array([0.0, 0.1]), lam=0.05)
    assert np.all(np.isfinite(dq)) and np.linalg.norm(dq) < 10


def test_pose_error_sign_and_zero():
    T0 = tf.make_T(tf.rotz(0.3), [1, 2, 3])
    assert np.allclose(tj.pose_error(T0, T0), 0)
    T1 = tf.make_T(tf.rotz(0.5), [1, 2, 4])
    assert np.allclose(tj.pose_error(T0, T1), [0, 0, 1, 0, 0, 0.2])


# ----------------------------------------------------------------------------- time scaling
def test_time_stretch_scaling_laws():
    tr = tj.quintic(0.0, 1.0, 1.0)
    st = tj.TimeScaled(tr, 2.0)
    assert np.isclose(st.duration, 2.0)
    t = np.linspace(0, 2, 9)
    assert np.allclose(st.pos(t), tr.pos(t / 2))
    assert np.allclose(st.vel(t), tr.vel(t / 2) / 2)
    assert np.allclose(st.acc(t), tr.acc(t / 2) / 4)
    assert np.allclose(st.jerk(t), tr.jerk(t / 2) / 8)


def test_required_time_stretch_makes_trajectory_feasible_and_tight():
    tr = tj.quintic(np.zeros(2), np.array([1.2, 0.5]), 1.0)
    v_lim, a_lim = np.array([1.0, 1.0]), np.array([2.0, 2.0])
    lam = tj.required_time_stretch(tr, v_lim, a_lim)
    st = tj.TimeScaled(tr, lam)
    rep = tj.check_limits(st, v_lim=v_lim, a_lim=a_lim)
    assert rep["ok"] and max(rep["vel"].max(), rep["acc"].max()) > 1 - 1e-6        # tight


def test_time_optimal_straight_line_reproduces_trapezoid():
    q0, q1 = np.zeros(3), np.array([1.0, -0.5, 0.25])
    vmax, amax = np.array([1.0, 1.0, 1.0]), np.array([3.0, 3.0, 3.0])
    path = tj.Path(lambda s: q0 + s * (q1 - q0), lambda s: q1 - q0, lambda s: 0 * q0)
    sol = tj.time_optimal_scaling(path, vmax, amax, N=400)
    ref = tj.min_time_trapezoid(1.0, 1.0 / 1.0, 3.0 / 1.0)                          # joint 1 limits s
    assert np.isclose(sol.duration, ref["T"], rtol=2e-3)


def test_time_optimal_curved_path_obeys_limits_and_beats_uniform_scaling():
    R = 0.5

    def q(s):
        return np.array([R * np.cos(2 * np.pi * 0.75 * s), R * np.sin(2 * np.pi * 0.75 * s)])

    def dq(s):
        w = 2 * np.pi * 0.75
        return np.array([-R * w * np.sin(w * s), R * w * np.cos(w * s)])

    def ddq(s):
        w = 2 * np.pi * 0.75
        return np.array([-R * w ** 2 * np.cos(w * s), -R * w ** 2 * np.sin(w * s)])

    path = tj.Path(q, dq, ddq)
    vmax, amax = np.array([2.0, 2.0]), np.array([4.0, 4.0])
    sol = tj.time_optimal_scaling(path, vmax, amax, N=500)
    rep = tj.check_limits(sol, v_lim=vmax, a_lim=amax, num=6001)
    assert rep["vel"].max() < 1.01 and rep["acc"].max() < 1.03
    # at every instant some limit is active (time-optimality), except near the ends
    t = np.linspace(0.05, sol.duration - 0.05, 400)
    act = np.maximum(np.abs(sol.vel(t)) / vmax, np.abs(sol.acc(t)) / amax).max(axis=1)
    assert np.median(act) > 0.97
    uni = tj.quintic(0.0, 1.0, 1.0)                                                # uniform scaling of a quintic s(t)
    # a rest-to-rest quintic in s along this path, stretched to the limits, is slower
    class QuinticPath(tj.Trajectory):
        n, scalar, t0, t1 = 2, False, 0.0, 1.0

        def _eval(self, t, d):
            s = uni._eval(t, 0)[:, 0], uni._eval(t, 1)[:, 0], uni._eval(t, 2)[:, 0]
            if d == 0:
                return np.array([q(x) for x in s[0]])
            g = np.array([dq(x) for x in s[0]])
            if d == 1:
                return g * s[1][:, None]
            h = np.array([ddq(x) for x in s[0]])
            return h * (s[1] ** 2)[:, None] + g * s[2][:, None]

    lam = tj.required_time_stretch(QuinticPath(), vmax, amax)
    assert sol.duration < lam


# ----------------------------------------------------------------------------- limits and servo helpers
def test_check_limits_flags_violation():
    tr = tj.cubic(0.0, 2.0, 1.0)
    rep = tj.check_limits(tr, q_lim=(-1.0, 1.0), v_lim=1.0, a_lim=20.0)
    assert not rep["ok"] and rep["pos"].max() > 1 and rep["vel"].max() > 1 and rep["acc"].max() < 1


def test_servo_times_end_exactly_at_T():
    t = tj.servo_times(1.0, rate=125)
    assert np.isclose(t[1] - t[0], 0.008) and t[-1] == 1.0 and len(t) == 126
    t = tj.servo_times(1.0, dt=0.3)
    assert np.allclose(t, [0, 0.3, 0.6, 0.9, 1.0])


def test_online_generator_converges_without_exceeding_limits():
    dt = 0.002
    gen = tj.OnlinePTP([0.0, 0.0], vmax=[1.0, 0.5], amax=[2.0, 4.0], dt=dt)
    target = np.array([1.5, -0.4])
    qs, vs = [], []
    for _ in range(4000):
        q, v = gen.step(target)
        qs.append(q), vs.append(v)
    qs, vs = np.array(qs), np.array(vs)
    assert np.allclose(qs[-1], target, atol=1e-6) and np.allclose(vs[-1], 0)
    assert np.all(np.abs(vs) <= np.array([1.0, 0.5]) * (1 + 1e-9))
    acc = np.diff(vs, axis=0) / dt
    assert np.all(np.abs(acc) <= np.array([2.0, 4.0]) * (1 + 1e-6))
    # no overshoot beyond a few steps' worth of motion
    assert qs[:, 0].max() < 1.5 + 1e-3

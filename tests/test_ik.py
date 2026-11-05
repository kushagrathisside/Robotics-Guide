import os

import numpy as np
import pytest

from rgkit import ik
from rgkit.kinematics import fk_dh, planar_fk
from rgkit.transforms import rotz, rotx

deg = np.deg2rad
rng = np.random.default_rng(7)


# ---- trigonometric toolbox --------------------------------------------------------------------------------
def test_toolbox_equations_are_satisfied():
    for _ in range(200):
        a, b = rng.normal(size=2)
        c = rng.uniform(-1, 1) * np.hypot(a, b)
        for solver in (ik.solve_a_cos_b_sin, ik.solve_half_angle):
            th = solver(a, b, c)
            assert len(th) == 2
            assert np.allclose(a * np.cos(th) + b * np.sin(th), c)
    assert len(ik.solve_a_cos_b_sin(1, 1, 2)) == 0 and len(ik.solve_half_angle(1, 1, 2)) == 0


def test_half_angle_pi_root_and_tangent():
    th = ik.solve_half_angle(1.0, 0.5, -1.0)                     # a + c = 0: theta = pi is a root
    assert np.any(np.isclose(np.abs(th), np.pi)) and np.allclose(np.cos(th) + 0.5 * np.sin(th), -1.0)
    th = ik.solve_a_cos_b_sin(3, 4, 5)                           # tangent: exactly one solution
    assert len(th) == 1 and np.isclose(3 * np.cos(th[0]) + 4 * np.sin(th[0]), 5)


def test_worked_example_half_angle():
    th = np.sort(np.degrees(ik.solve_half_angle(3, 4, 2)))
    assert np.allclose(th, [-13.29, 119.55], atol=0.01)


def test_sin_cos_and_two_equation_forms():
    assert np.allclose(np.sort(ik.solve_sin(0.5)), np.sort([np.arcsin(0.5), np.pi - np.arcsin(0.5)]))
    assert np.allclose(np.sort(ik.solve_cos(0.5)), [-np.arccos(0.5), np.arccos(0.5)])
    t = 2.1
    a, b = 1.3, 0.7
    assert np.isclose(ik.solve_two_equations(a, b, a * np.cos(t) - b * np.sin(t), a * np.sin(t) + b * np.cos(t)), t)


# ---- planar arms ----------------------------------------------------------------------------------------------
def test_2r_both_branches_and_geometric_agree():
    L1, L2 = 1.0, 0.8
    for _ in range(100):
        r, ang = rng.uniform(abs(L1 - L2) + 1e-3, L1 + L2 - 1e-3), rng.uniform(-np.pi, np.pi)
        x, y = r * np.cos(ang), r * np.sin(ang)
        A, G = ik.ik_2r(x, y, L1, L2), ik.ik_2r_geometric(x, y, L1, L2)
        assert A.shape == (2, 2) and np.allclose(A, G)
        assert A[0, 1] > 0 > A[1, 1]
        for q in A:
            assert np.allclose(ik.fk_2r(q, L1, L2), [x, y])


def test_2r_boundary_and_outside():
    assert ik.ik_2r(1.8, 0.0, 1.0, 0.8).shape == (1, 2)
    assert ik.ik_2r(0.2, 0.0, 1.0, 0.8).shape == (1, 2)
    assert ik.ik_2r(1.9, 0.0, 1.0, 0.8).shape == (0, 2)
    assert ik.ik_2r(0.1, 0.0, 1.0, 0.8).shape == (0, 2)


def test_2r_instructor_style_numbers():
    sol = np.degrees(ik.ik_2r(1.2, 0.8, 1.0, 0.8))
    assert np.allclose(sol, [[1.46, 74.04], [65.92, -74.04]], atol=0.01)


def test_3r_planar():
    L = (1.0, 0.8, 0.5)
    for _ in range(100):
        q = rng.uniform(-np.pi, np.pi, 3)
        pose = ik.fk_3r_planar(q, *L)
        sols = ik.ik_3r_planar(*pose, *L)
        assert len(sols) in (1, 2)
        assert any(np.allclose(ik.wrap_angle(s - q), 0, atol=1e-8) for s in sols)
        for s in sols:
            assert np.allclose(ik.fk_3r_planar(s, *L)[:2], pose[:2]) and np.isclose(ik.wrap_angle(s.sum() - pose[2]), 0)
    assert np.allclose(ik.fk_3r_planar([0.3, -0.4, 0.2], 1, 1, 1)[:2], planar_fk([1, 1, 1], [0.3, -0.4, 0.2])[-1])


# ---- articulated arm and SCARA -----------------------------------------------------------------------------------
def test_articulated_four_solutions():
    sols = ik.ik_articulated([1, 1, 1], 1.0, 1.0)
    assert sols.shape == (4, 3)
    assert np.allclose(np.degrees(sols), [[45, 5.26, 60], [45, 65.26, -60], [-135, 114.74, 60], [-135, 174.74, -60]], atol=0.01)
    for q in sols:
        assert np.allclose(ik.fk_articulated(q, 1.0, 1.0), [1, 1, 1])
    for _ in range(100):
        q = np.array([rng.uniform(-3, 3), rng.uniform(-1.5, 1.5), rng.uniform(-3, 3)])
        p = ik.fk_articulated(q, 0.7, 0.5, 0.3)
        S = ik.ik_articulated(p, 0.7, 0.5, 0.3)
        assert any(np.allclose(ik.wrap_angle(s - q), 0, atol=1e-8) for s in S)
        assert all(np.allclose(ik.fk_articulated(s, 0.7, 0.5, 0.3), p) for s in S)


def test_scara():
    q = np.array([0.4, -0.9, 0.12, 0.3])
    pose = ik.fk_scara(q, 0.4, 0.3, 0.1)
    S = ik.ik_scara(*pose, 0.4, 0.3, 0.1)
    assert len(S) == 2 and any(np.allclose(ik.wrap_angle(s - q), 0) for s in S)
    assert all(np.allclose(ik.fk_scara(s, 0.4, 0.3, 0.1), pose) for s in S)


# ---- spherical-wrist arm (PUMA type) -----------------------------------------------------------------------------
@pytest.mark.parametrize("geom", [(0.45, -0.03, 0.15, 0.43), (0.5, 0.1, 0.0, 0.3), (0.3, 0.25, -0.1, 0.4)])
def test_puma_eight_solutions_and_roundtrip(geom):
    table = ik.puma_dh(*geom)
    counts = []
    for _ in range(150):
        q = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3),
                      rng.uniform(0.2, 2.9) * rng.choice([-1, 1]), rng.uniform(-3, 3)])
        T = fk_dh(table, q)[-1]
        S, B = ik.ik_puma(T, *geom, return_branches=True)
        counts.append(len(S))
        assert any(np.allclose(ik.wrap_angle(s - q), 0, atol=1e-7) for s in S)
        for s in S:
            assert np.allclose(fk_dh(table, s)[-1], T, atol=1e-8)
        assert len(set(B)) == len(B)
    assert set(counts) == {8}                       # a wrist-centre point inside the workspace has exactly 8 solutions


def test_puma_out_of_reach_and_wrist_singularity():
    geom = (0.45, -0.03, 0.15, 0.43)
    T = np.eye(4)
    T[:3, 3] = [3, 0, 0]
    assert ik.ik_puma(T, *geom).shape == (0, 6)
    table = ik.puma_dh(*geom)
    q = np.array([0.3, -0.5, 0.8, 0.4, 0.0, -0.2])
    Tq = fk_dh(table, q)[-1]
    S = ik.ik_puma(Tq, *geom)
    assert len(S) >= 4 and all(np.allclose(fk_dh(table, s)[-1], Tq, atol=1e-8) for s in S)


# ---- UR-type arm -------------------------------------------------------------------------------------------------
UR5E = ik.URGeometry(d1=0.163, a2=0.425, a3=0.392, d4=0.134, d5=0.1, d6=0.1)


def test_ur_eight_solutions_and_roundtrip():
    counts = []
    for _ in range(300):
        q = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3),
                      rng.uniform(0.2, 2.9) * rng.choice([-1, 1]), rng.uniform(-3, 3)])
        T = ik.fk_ur(q, UR5E)
        S, B = ik.ik_ur(T, UR5E, return_branches=True)
        counts.append(len(S))
        assert any(np.allclose(ik.wrap_angle(s - q), 0, atol=1e-7) for s in S)
        for s in S:
            assert np.allclose(ik.fk_ur(s, UR5E), T, atol=1e-8)
        assert len(set(B)) == len(B)
    assert 8 in counts and set(counts) <= {2, 4, 6, 8}      # some branches can be out of reach for the non-spherical wrist


def test_ur_wrist_singularity_and_unreachable():
    q = np.array([0.2, -1.1, 1.3, -0.4, 0.0, 0.5])
    T = ik.fk_ur(q, UR5E)
    S = ik.ik_ur(T, UR5E)
    assert len(S) >= 2 and all(np.allclose(ik.fk_ur(s, UR5E), T, atol=1e-8) for s in S)
    T2 = np.eye(4)
    T2[:3, 3] = [5, 0, 0]
    assert ik.ik_ur(T2, UR5E).shape == (0, 6)


# ---- solution selection ---------------------------------------------------------------------------------------------
def test_nearest_and_limits():
    sols = np.array([[0.1, 0.2], [3.0, -3.0], [-3.0, 3.0]])
    i, q = ik.nearest_solution(sols, [3.1, -3.1])
    assert i == 1
    i, q = ik.nearest_solution(sols, [-3.1, 3.1], weights=[1, 1])
    assert i == 2
    mask = ik.within_limits(sols, [-0.5, -0.5], [0.5, 0.5])
    assert mask.tolist() == [True, False, False]


# ---- numerical IK ---------------------------------------------------------------------------------------------------------
def _planar_problem(L=(1.0, 1.0), target=(1.5, 0.5)):
    err = lambda q: np.asarray(target) - planar_fk(L, q)[-1]
    jac = lambda q: ik.jacobian_planar(L, q)
    return err, jac


def test_planar_jacobian_matches_finite_differences():
    L, q = [1.0, 0.8, 0.5], np.array([0.3, -0.7, 1.1])
    Jn = ik.numeric_jacobian(lambda qq: planar_fk(L, qq)[-1], q)
    assert np.allclose(ik.jacobian_planar(L, q), Jn, atol=1e-8)


def test_newton_matches_instructor_style_example():
    err, jac = _planar_problem()
    r = ik.solve_ik(err, jac, [0.0, np.pi / 2], method="newton", tol=1e-8)
    assert r.converged and r.iterations == 4
    assert np.allclose(np.degrees(r.q), [-19.33, 75.52], atol=0.01)
    e = np.array(r.history["err"])
    assert e[1] < e[0] ** 1.5 and e[2] < e[1] ** 1.5                  # super-linear (quadratic) convergence


def test_gradient_descent_is_linear_and_slower():
    err, jac = _planar_problem()
    rn = ik.solve_ik(err, jac, [0.0, np.pi / 2], method="newton", tol=1e-6)
    rg = ik.solve_ik(err, jac, [0.0, np.pi / 2], method="gd", alpha=0.5, tol=1e-6, max_iter=500)
    assert rg.converged and rg.iterations > 5 * rn.iterations
    ro = ik.solve_ik(err, jac, [0.0, np.pi / 2], method="gd", alpha=None, tol=1e-6, max_iter=500)
    assert ro.converged
    assert np.allclose(np.sort(np.degrees(ro.q)), np.sort(np.degrees(rn.q)), atol=0.01)


def test_dls_is_bounded_at_singularity_but_newton_fails():
    err, jac = _planar_problem()
    rn = ik.solve_ik(err, jac, [0.0, 0.0], method="newton", max_iter=20)
    assert not rn.converged
    rd = ik.solve_ik(err, jac, [0.0, 0.0], method="dls", lam=0.2, max_iter=300, tol=1e-6)
    assert rd.converged and max(rd.history["step"]) < 2.0


def test_pinv_redundant_and_nullspace_term():
    L = (1.0, 0.8, 0.6)
    target = np.array([1.2, 0.9])
    err = lambda q: target - planar_fk(L, q)[-1]
    jac = lambda q: ik.jacobian_planar(L, q)
    q0 = np.array([0.3, 0.5, 0.4])
    r1 = ik.solve_ik(err, jac, q0, method="pinv", tol=1e-10)
    pref = np.array([0.0, 0.0, 0.0])
    r2 = ik.solve_ik(err, jac, q0, method="pinv", tol=1e-10, nullspace=lambda q: 0.3 * (pref - q), max_iter=500)
    assert r1.converged and r2.converged
    assert np.allclose(planar_fk(L, r1.q)[-1], target) and np.allclose(planar_fk(L, r2.q)[-1], target)


def test_unknown_method_raises():
    err, jac = _planar_problem()
    with pytest.raises(ValueError):
        ik.solve_ik(err, jac, [0.1, 0.2], method="bogus")


def test_rotation_error_is_zero_for_equal_and_matches_axis_angle():
    R = rotz(0.4) @ rotx(0.3)
    assert np.allclose(ik.rotation_error(R, R), 0)
    assert np.allclose(ik.rotation_error(rotz(0.5) @ R, R), [0, 0, 0.5])


# ---- MuJoCo cross-checks (skipped if MuJoCo or the model cache is missing) ----------------------------------------------------------
def _load(name):
    robots = pytest.importorskip("rgkit.robots")
    pytest.importorskip("mujoco")
    try:
        return robots.load_robot(name, scene="bare")
    except Exception as exc:                     # model not cached and no network
        pytest.skip(f"model {name} unavailable: {exc}")


def test_ur5e_geometry_dh_and_ik_match_mujoco():
    m, d = _load("ur5e")
    arm = ik.MujocoArm(m, d, "attachment_site")
    T_wb = np.eye(4)
    import mujoco
    mujoco.mj_forward(m, d)
    T_wb[:3, :3], T_wb[:3, 3] = d.body("base").xmat.reshape(3, 3), d.body("base").xpos
    Rz_pi = np.eye(4)
    Rz_pi[:3, :3] = rotz(np.pi)
    for _ in range(25):
        q = rng.uniform(-3, 3, 6)
        T_sim = arm.fk(q)
        T_dh = T_wb @ ik.fk_ur(q, UR5E) @ Rz_pi
        assert np.allclose(T_sim, T_dh, atol=1e-9)
        S = ik.ik_ur(np.linalg.inv(T_wb) @ T_sim @ Rz_pi.T, UR5E)
        assert len(S) >= 2
        for s in S:
            assert np.allclose(arm.fk(s), T_sim, atol=1e-8)


def test_mujoco_numerical_ik_converges_on_ur5e():
    m, d = _load("ur5e")
    arm = ik.MujocoArm(m, d, "attachment_site")
    q_true = np.array([0.4, -1.2, 1.5, -1.0, -1.4, 0.3])
    T_d = arm.fk(q_true)
    err = arm.pose_error(T_d)
    r = ik.solve_ik(err, arm.jac, q_true + 0.2, method="dls", lam=0.05, tol=1e-8, max_iter=100)
    assert r.converged and np.allclose(arm.fk(r.q), T_d, atol=1e-6)
    # MuJoCo's Jacobian agrees with finite differences of its own forward kinematics (position rows)
    q = rng.uniform(-1.5, 1.5, 6)
    Jn = ik.numeric_jacobian(lambda qq: arm.fk(qq)[:3, 3], q)
    assert np.allclose(arm.jac(q)[:3], Jn, atol=1e-6)


def test_2r_model_closed_form_matches_mujoco():
    m, d = _load("2r")
    import mujoco
    L1 = L2 = 0.18
    shoulder = np.array([0.0, 0.0, 0.5452])
    arm = ik.MujocoArm(m, d, "end")
    target_uv = np.array([0.15, -0.20])          # (horizontal, vertical) relative to the shoulder
    sols = ik.ik_2r(*target_uv, L1, L2)
    assert len(sols) == 2
    for th1, th2 in sols:
        q = [th1 + np.pi / 2, th2]               # Craig angle -> MuJoCo joint angle (zero = hanging down)
        p = arm.fk(q)[:3, 3]
        assert np.allclose([p[0], p[2]], shoulder[[0, 2]] + target_uv, atol=1e-9)

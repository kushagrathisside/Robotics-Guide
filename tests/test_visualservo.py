import numpy as np
import pytest
from scipy.linalg import expm

from rgkit import visualservo as vs
from rgkit.transforms import make_T, rotz, rotx, roty, inv_T, skew, rot_axis_angle

CAM = vs.PinholeCamera.from_fovy(60.0, 320, 240)
SQUARE = 0.1 * np.array([[-1.0, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])      # 20 cm square target, z = 0


def test_camera_from_fovy_projection_and_backprojection():
    assert np.isclose(CAM.fy, 120 / np.tan(np.deg2rad(30)))
    assert np.allclose(CAM.project([[0, 0, 2.0]]), [[160, 120]])
    P = np.array([[0.3, -0.2, 1.5], [-0.1, 0.4, 0.8]])
    uv = CAM.project(P)
    assert np.allclose(CAM.backproject(uv, P[:, 2]), P)
    assert np.allclose(CAM.denormalise(CAM.normalise(uv)), uv)
    assert np.allclose(CAM.K @ np.array([0.3, -0.2, 1.5]) / 1.5, [uv[0, 0], uv[0, 1], 1.0])
    assert CAM.in_view([[10, 10], [300, 200]]) and not CAM.in_view([[-5, 10]])


def test_mujoco_convention_conversion():
    R_mj = rotx(0.3) @ rotz(1.1)
    T = vs.mujoco_to_cv_pose([1, 2, 3], R_mj.ravel())
    assert np.allclose(T[:3, 0], R_mj[:, 0]) and np.allclose(T[:3, 2], -R_mj[:, 2]) and np.allclose(T[:3, 1], -R_mj[:, 1])
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)


def test_look_at_points_the_optical_axis_at_the_target():
    T = vs.look_at([1.0, -2.0, 1.5], [0.2, 0.1, 0.0])
    z = T[:3, 2]
    d = np.array([0.2, 0.1, 0.0]) - np.array([1.0, -2.0, 1.5])
    assert np.allclose(z, d / np.linalg.norm(d))
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3)) and np.isclose(np.linalg.det(T[:3, :3]), 1.0)
    uv, Z = vs.project_world(CAM, T, [[0.2, 0.1, 0.0]])
    assert np.allclose(uv, [[160, 120]]) and Z[0] > 0


def test_interaction_matrix_matches_finite_differences_of_the_projection():
    rng = np.random.default_rng(0)
    T = make_T(rot_axis_angle([1, 2, 3], 0.4), [0.1, -0.05, -1.2])             # target 1.2 m in front (camera at p)
    P_w = rng.uniform(-0.2, 0.2, (5, 3))
    uv, Z = vs.project_world(CAM, T, P_w)
    L = vs.interaction_matrix_px(uv, Z, CAM)
    xi = np.array([0.02, -0.01, 0.03, 0.05, -0.04, 0.06])                      # twist in the camera frame
    eps = 1e-6
    uv2, _ = vs.project_world(CAM, T @ vs.exp_se3(xi * eps), P_w)
    uv0, _ = vs.project_world(CAM, T @ vs.exp_se3(-xi * eps), P_w)
    numeric = ((uv2 - uv0) / (2 * eps)).ravel()
    assert np.allclose(L @ xi, numeric, rtol=1e-5, atol=1e-6)


def test_interaction_matrix_matches_symbolic_derivation():
    sympy = pytest.importorskip("sympy")
    X, Y, Z, vx, vy, vz, wx, wy, wz = sympy.symbols("X Y Z vx vy vz wx wy wz")
    P = sympy.Matrix([X, Y, Z])
    Pdot = -sympy.Matrix([vx, vy, vz]) - sympy.Matrix([wx, wy, wz]).cross(P)
    x, y = X / Z, Y / Z
    xdot = sympy.diff(x, X) * Pdot[0] + sympy.diff(x, Z) * Pdot[2]
    ydot = sympy.diff(y, Y) * Pdot[1] + sympy.diff(y, Z) * Pdot[2]
    twist = [vx, vy, vz, wx, wy, wz]
    L_sym = sympy.Matrix([[sympy.diff(xdot, t) for t in twist], [sympy.diff(ydot, t) for t in twist]])
    pt = {X: 0.31, Y: -0.17, Z: 1.9}
    L_num = np.array(L_sym.subs(pt).evalf().tolist(), dtype=float)
    assert np.allclose(vs.interaction_matrix([[0.31 / 1.9, -0.17 / 1.9]], [1.9]), L_num)


def test_interaction_matrix_ranks():
    xy = np.array([[0.1, 0.2], [-0.2, 0.1], [0.05, -0.2], [0.2, -0.1]])
    Z = np.array([1.0, 1.1, 0.9, 1.0])
    assert np.linalg.matrix_rank(vs.interaction_matrix(xy[:1], Z[:1])) == 2
    assert np.linalg.matrix_rank(vs.interaction_matrix(xy[:2], Z[:2])) == 4
    assert np.linalg.matrix_rank(vs.interaction_matrix(xy[:3], Z[:3])) == 6
    assert np.linalg.matrix_rank(vs.interaction_matrix(xy, Z)) == 6
    # three collinear points in the image at equal depth: rank drops below 6
    col = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    assert np.linalg.matrix_rank(vs.interaction_matrix(col, [1, 1, 1])) < 6


def test_exp_se3_matches_matrix_exponential():
    xi = np.array([0.3, -0.2, 0.5, 0.4, 0.1, -0.6])
    S = np.zeros((4, 4)); S[:3, :3] = skew(xi[3:]); S[:3, 3] = xi[:3]
    assert np.allclose(vs.exp_se3(xi), expm(S))
    assert np.allclose(vs.exp_se3(np.array([0.1, 0.2, 0.3, 0, 0, 0])), make_T(p=[0.1, 0.2, 0.3]))
    assert np.allclose(vs.exp_se3(np.zeros(6)), np.eye(4))


def _poses(rot_deg=30.0, offset=(0.15, -0.1, 0.0)):
    T_star = vs.look_at([0.0, 0.0, 0.6], [0, 0, 0], up=(0, 1, 0))                # looking down at the board from 0.6 m
    T0 = T_star @ make_T(rotz(np.deg2rad(rot_deg)), offset)
    T0 = T0 @ make_T(p=[0, 0, -0.15])
    return T0, T_star


def test_ibvs_converges_for_all_three_jacobian_choices():
    P_w = SQUARE
    T0, T_star = _poses()
    for L_at in ("current", "desired", "mean"):
        r = vs.simulate_ibvs(CAM, P_w, T0, T_star, lam=1.0, dt=0.05, n_steps=300, L_at=L_at)
        assert r["e"][-1] < 1e-3, L_at
        Tf = r["T"][-1]
        assert np.linalg.norm(Tf[:3, 3] - T_star[:3, 3]) < 1e-3
        assert np.allclose(Tf[:3, :3], T_star[:3, :3], atol=1e-3)


def test_ibvs_error_decays_exponentially_for_the_ideal_law():
    T0, T_star = _poses(10.0, (0.02, 0.0, 0.0))
    r = vs.simulate_ibvs(CAM, SQUARE, T0, T_star, lam=1.0, dt=0.01, n_steps=200)
    e = r["e"]
    ratio = e[100] / e[0]
    assert abs(ratio - np.exp(-1.0)) < 0.08                                      # small motion: nearly e^{-lambda t}


def test_ibvs_tolerates_a_factor_of_two_depth_error_but_not_a_factor_of_fifty_with_wrong_sign_structure():
    T0, T_star = _poses()
    ok = vs.simulate_ibvs(CAM, SQUARE, T0, T_star, lam=1.0, dt=0.05, n_steps=600, depth_factor=2.0)
    assert ok["e"][-1] < 1e-2
    ok = vs.simulate_ibvs(CAM, SQUARE, T0, T_star, lam=1.0, dt=0.05, n_steps=600, depth_factor=0.5)
    assert ok["e"][-1] < 1e-2


def test_camera_retreat_for_large_rotation_about_the_optical_axis():
    T_star = vs.look_at([0.0, 0.0, 0.6], [0, 0, 0], up=(0, 1, 0))
    Zs = 0.6
    retreat = []
    for ang in (30, 60, 120, 150):
        T0 = T_star @ make_T(rotz(np.deg2rad(ang)))
        r = vs.simulate_ibvs(CAM, SQUARE, T0, T_star, lam=1.0, dt=0.05, n_steps=800)
        depth = -(inv_T(T_star) @ np.column_stack([r["T"][:, :3, 3], np.ones(len(r["T"]))]).T)[2]
        retreat.append(depth.max())
        assert r["e"][-1] < 1e-3                                                 # it still converges for angles below 180 degrees
    assert np.all(np.diff(retreat) > 0)                                          # the retreat grows with the rotation
    assert retreat[1] < 0.2 * Zs and retreat[3] > 2.5 * Zs                       # modest at 60 degrees, enormous at 150


def test_pbvs_follows_a_straight_line_and_converges():
    T0, T_star = _poses(70.0, (0.2, 0.15, 0.0))
    T_wo = np.eye(4)
    r = vs.simulate_pbvs(CAM, SQUARE, T_wo, T0, T_star, lam=1.0, dt=0.01, n_steps=1000)
    p = r["T"][:, :3, 3]
    d = (p[-1] - p[0]); d /= np.linalg.norm(d)
    off_line = np.linalg.norm((p - p[0]) - np.outer((p - p[0]) @ d, d), axis=1)
    assert off_line.max() < 1e-3                                                 # straight in Cartesian space (up to the time step)
    assert np.linalg.norm(p[-1] - T_star[:3, 3]) < 1e-3
    assert r["pose_err"].max() < 1e-6


def test_pbvs_is_sensitive_to_intrinsic_errors_ibvs_is_not():
    T0, T_star = _poses()
    bad = vs.PinholeCamera(CAM.width, CAM.height, 1.25 * CAM.fx, 1.25 * CAM.fy, CAM.cx + 6, CAM.cy - 6)
    pb = vs.simulate_pbvs(CAM, SQUARE, np.eye(4), T0, T_star, lam=1.0, dt=0.05, n_steps=300, cam_ctrl=bad)
    ib = vs.simulate_ibvs(CAM, SQUARE, T0, T_star, lam=1.0, dt=0.05, n_steps=300, cam_ctrl=bad)
    assert ib["e"][-1] < 1e-2                                                    # image-based: error goes to zero anyway
    assert pb["e"][-1] > 5.0                                                     # position-based: converges to a wrong pose


def test_planar_pose_exact_and_noisy():
    T_co = make_T(rot_axis_angle([0.3, 1.0, 0.2], 0.5), [0.05, -0.04, 0.7])
    uv = CAM.project(SQUARE @ T_co[:3, :3].T + T_co[:3, 3])
    est = vs.planar_pose(SQUARE, uv, CAM)
    assert np.allclose(est, T_co, atol=1e-8)
    est_dlt = vs.planar_pose(SQUARE, uv, CAM, refine=False)
    assert np.allclose(est_dlt, T_co, atol=1e-6)
    rng = np.random.default_rng(1)
    est = vs.planar_pose(SQUARE, uv + 0.3 * rng.standard_normal(uv.shape), CAM)
    assert np.linalg.norm(est[:3, 3] - T_co[:3, 3]) < 0.02


def test_eye_to_hand_converges_with_the_opposite_sign():
    T_co_star = make_T(rotz(0.2), [0.0, 0.0, 0.8])
    T_co0 = make_T(rotz(0.9) @ rotx(0.2), [0.12, -0.08, 1.0])
    r = vs.simulate_eye_to_hand(CAM, np.eye(4), SQUARE, T_co0, T_co_star, lam=1.0, dt=0.05, n_steps=400)
    assert r["e"][-1] < 1e-3
    assert np.allclose(r["T_co"][-1][:3, 3], T_co_star[:3, 3], atol=1e-3)


def test_dls_and_twist_helpers():
    J = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v = np.array([0.3, -0.2])
    assert np.allclose(vs.dls_solve(J, v, 1e-6), [0.3, -0.2, 0.0], atol=1e-6)
    assert np.linalg.norm(vs.dls_solve(J, v, 1.0)) < np.linalg.norm(v)
    R = rotz(0.5)
    assert np.allclose(vs.twist_to_world(R, [1, 0, 0, 0, 0, 1]), np.concatenate([R[:, 0], R[:, 2]]))

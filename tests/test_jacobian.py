"""Tests of rgkit.jacobian: every closed form is checked against finite differences or MuJoCo."""
import os
from pathlib import Path

import numpy as np
import pytest

from rgkit import jacobian as jc
from rgkit import kinematics as kin
from rgkit import transforms as tf

deg = np.deg2rad
RNG = np.random.default_rng(12345)


def random_table(n=6, prismatic=(), convention_modified=True):
    rows = []
    for i in range(n):
        kind = "P" if i in prismatic else "R"
        rows.append((RNG.uniform(-np.pi, np.pi), RNG.uniform(-0.5, 0.5), RNG.uniform(-0.5, 0.5),
                     RNG.uniform(-np.pi, np.pi), kind))
    return rows


# ---------------------------------------------------------------------------
# Angular velocity
# ---------------------------------------------------------------------------
def test_angular_velocity_space_and_body():
    k, w_true = tf.rot_axis_angle([1, 2, 2], 0.7), np.array([0.3, -0.5, 0.8])
    Rdot = tf.skew(w_true) @ k
    assert np.allclose(jc.angular_velocity_space(k, Rdot), w_true)
    assert np.allclose(jc.angular_velocity_body(k, Rdot), k.T @ w_true)


def test_rotation_vector_roundtrip():
    rv = np.array([0.2, -0.4, 0.9])
    R = tf.rot_axis_angle(rv, np.linalg.norm(rv))
    assert np.allclose(jc.rotation_vector(R), rv)
    assert np.allclose(jc.rotation_vector(np.eye(3)), 0)


# ---------------------------------------------------------------------------
# Geometric Jacobian against finite differences
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("convention", ["modified", "standard"])
@pytest.mark.parametrize("prismatic", [(), (2,), (0, 3)])
def test_geometric_jacobian_matches_finite_differences(convention, prismatic):
    table = random_table(6, prismatic)
    q = RNG.uniform(-1.5, 1.5, 6)
    tool = tf.make_T(tf.rotz(0.4) @ tf.rotx(-0.3), [0.05, -0.02, 0.1])
    J = jc.jacobian_geometric(table, q, convention, tool)
    Jn = jc.jacobian_numeric(lambda x: kin.fk_tool(table, x, convention, tool), q)
    assert np.allclose(J, Jn, atol=1e-6)


def test_jacobian_default_zero_configuration():
    table = [(0, 0, 0, 0, "R"), (0, 1.0, 0, 0, "R"), (0, 0.8, 0, 0, "R")]
    J = jc.jacobian_geometric(table, None)
    assert J.shape == (6, 3)


def test_planar_jacobian_matches_dh_and_closed_form():
    L = [1.0, 0.7, 0.4]
    q = np.array([0.3, -0.8, 0.5])
    table = [(0, 0, 0, 0, "R"), (0, L[0], 0, 0, "R"), (0, L[1], 0, 0, "R"), (0, L[2], 0, 0, "R")]
    Jd = jc.jacobian_geometric(table, np.append(q, 0.0))[:, :3]
    Jp = jc.planar_jacobian(L, q, orientation=True)
    assert np.allclose(Jd[[0, 1, 5]], Jp)
    J2 = jc.planar_jacobian([1.0, 0.8], [0.5, 0.9])
    assert np.isclose(np.linalg.det(J2), 1.0 * 0.8 * np.sin(0.9))       # det J = L1 L2 sin q2


def test_numeric_position_jacobian():
    f = lambda q: np.array([np.sin(q[0]) * q[1], q[0] ** 2 + q[1]])
    q = np.array([0.4, 1.3])
    assert np.allclose(jc.jacobian_position_numeric(f, q), [[np.cos(0.4) * 1.3, np.sin(0.4)], [0.8, 1.0]], atol=1e-8)


# ---------------------------------------------------------------------------
# Frames, reference points, duality
# ---------------------------------------------------------------------------
def test_jacobian_in_tool_frame_and_propagation_agree():
    table = random_table(6, prismatic=(1,))
    q = RNG.uniform(-1, 1, 6)
    tool = tf.make_T(tf.roty(0.3), [0.0, 0.05, 0.12])
    T = kin.fk_tool(table, q, tool=tool)
    J0 = jc.jacobian_geometric(table, q, tool=tool)
    Jt = jc.jacobian_in_frame(J0, T[:3, :3].T)
    assert np.allclose(Jt, jc.jacobian_by_propagation(table, q, tool), atol=1e-12)


def test_velocity_propagation_equals_jacobian_times_rates():
    table = random_table(6)
    q, qd = RNG.uniform(-1, 1, 6), RNG.uniform(-1, 1, 6)
    ws, vs = jc.propagate_velocity(table, q, qd)
    Ts = kin.fk_dh(table, q)
    J0 = jc.jacobian_geometric(table, q)
    twist0 = J0 @ qd
    R = Ts[-1][:3, :3]
    assert np.allclose(vs[-1], R.T @ twist0[:3]) and np.allclose(ws[-1], R.T @ twist0[3:])
    for i in range(1, 7):                       # every intermediate link velocity also matches
        Ji = jc.jacobian_geometric(table[:i], q[:i])
        t_i = Ji @ qd[:i]
        assert np.allclose(vs[i], Ts[i][:3, :3].T @ t_i[:3]) and np.allclose(ws[i], Ts[i][:3, :3].T @ t_i[3:])


def test_shift_reference_point():
    table = random_table(5)
    q = RNG.uniform(-1, 1, 5)
    J = jc.jacobian_geometric(table, q)
    r = np.array([0.1, -0.2, 0.3])                    # offset written in the base frame
    T = kin.fk_tool(table, q)
    tool = tf.make_T(None, T[:3, :3].T @ r)           # same offset as a tool transform
    assert np.allclose(jc.shift_reference_point(J, r), jc.jacobian_geometric(table, q, tool=tool))


def test_velocity_and_wrench_transforms_are_dual_and_correct():
    T_AB = tf.make_T(tf.rot_axis_angle([1, -2, 1], 0.9), [0.3, -0.1, 0.5])
    X, W = jc.velocity_transform(T_AB), jc.wrench_transform(T_AB)
    assert np.allclose(W, np.linalg.inv(X).T)
    # a point-by-point check: body twist at origin of B -> velocity of the body point at origin of A
    wB, vB = np.array([0.2, 0.5, -0.4]), np.array([0.1, -0.3, 0.2])
    pA_in_B = tf.inv_T(T_AB)[:3, 3]
    vA_in_B = vB + np.cross(wB, pA_in_B)
    out = X @ np.r_[vB, wB]
    R = T_AB[:3, :3]
    assert np.allclose(out[:3], R @ vA_in_B) and np.allclose(out[3:], R @ wB)
    # power invariance: F . V is frame independent
    F = RNG.normal(size=6)
    assert np.isclose((W @ F) @ out, F @ np.r_[vB, wB])


# ---------------------------------------------------------------------------
# Statics
# ---------------------------------------------------------------------------
def test_force_propagation_equals_jacobian_transpose():
    table = random_table(6, prismatic=(3,))
    q = RNG.uniform(-1, 1, 6)
    tool = tf.make_T(tf.rotz(0.5), [0.02, 0.0, 0.1])
    f_tool, n_tool = RNG.normal(size=3), RNG.normal(size=3)       # in the tool frame
    tau, _, _ = jc.propagate_force(table, q, f_tool, n_tool, tool)
    T = kin.fk_tool(table, q, tool=tool)
    F0 = np.r_[T[:3, :3] @ f_tool, T[:3, :3] @ n_tool]             # wrench in the base frame at the tool origin
    J0 = jc.jacobian_geometric(table, q, tool=tool)
    assert np.allclose(tau, jc.joint_torques(J0, F0), atol=1e-12)


def test_planar_2r_statics_example():
    J = np.array([[-1.2, -0.6], [0.6, -0.4]])
    assert np.allclose(jc.joint_torques(J, [2.0, 1.0]), [-1.8, -1.6])


# ---------------------------------------------------------------------------
# Euler-angle rates and the analytical Jacobian
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("order,frame", [("xyz", "fixed"), ("zyx", "current"), ("zyz", "current"),
                                         ("zyx", "fixed"), ("xyx", "current"), ("zxz", "fixed")])
def test_euler_rate_matrix_matches_finite_difference(order, frame):
    a = np.array([0.4, 0.7, -0.5])
    rates = np.array([0.3, -0.2, 0.5])
    h = 1e-6
    Rp = tf.euler_to_rot(a + h * rates, order, frame)
    Rm = tf.euler_to_rot(a - h * rates, order, frame)
    R = tf.euler_to_rot(a, order, frame)
    w_fd = jc.angular_velocity_space(R, (Rp - Rm) / (2 * h))
    assert np.allclose(jc.euler_rate_matrix(a, order, frame) @ rates, w_fd, atol=1e-8)
    assert np.allclose(jc.euler_rate_matrix_body(a, order, frame) @ rates, R.T @ w_fd, atol=1e-8)


def test_craig_rpy_rate_matrix_closed_form_and_singularity():
    g, b, a = 0.3, 0.6, -0.4                       # roll, pitch, yaw (X, Y, Z fixed angles)
    E = jc.euler_rate_matrix([g, b, a], "xyz", "fixed")
    closed = np.array([[np.cos(a) * np.cos(b), -np.sin(a), 0], [np.sin(a) * np.cos(b), np.cos(a), 0],
                       [-np.sin(b), 0, 1]])
    assert np.allclose(E, closed)
    assert np.isclose(np.linalg.det(E), np.cos(b))
    assert abs(np.linalg.det(jc.euler_rate_matrix([g, np.pi / 2, a], "xyz", "fixed"))) < 1e-12


def test_analytical_jacobian_matches_finite_difference_of_pose_parameters():
    table = random_table(6)
    q = RNG.uniform(-0.8, 0.8, 6)
    def pose_params(x):
        T = kin.fk_tool(table, x)
        g, b, a = tf.rot_to_rpy(T[:3, :3])
        return np.r_[T[:3, 3], g, b, a]
    Ja_fd = jc.jacobian_position_numeric(pose_params, q)
    Jg = jc.jacobian_geometric(table, q)
    g, b, a = tf.rot_to_rpy(kin.fk_tool(table, q)[:3, :3])
    Ja = jc.analytical_jacobian(Jg, [g, b, a], "xyz", "fixed")
    assert np.allclose(Ja, Ja_fd, atol=1e-6)


# ---------------------------------------------------------------------------
# Manipulability and inverses
# ---------------------------------------------------------------------------
def test_2r_manipulability_and_singular_values():
    L1, L2, q2 = 1.0, 0.8, deg(70)
    J = jc.planar_jacobian([L1, L2], [0.4, q2])
    assert np.isclose(jc.manipulability(J), abs(np.linalg.det(J)))
    assert np.isclose(jc.manipulability(J), L1 * L2 * np.sin(q2))
    s = jc.singular_values(J)
    assert np.isclose(s[0] * s[1], L1 * L2 * np.sin(q2))
    assert np.isclose(jc.condition_number(J), s[0] / s[1]) and np.isclose(jc.isotropy(J), s[1] / s[0])
    Js = jc.planar_jacobian([L1, L2], [0.4, 0.0])                   # stretched: singular
    assert jc.manipulability(Js) < 1e-12 and jc.condition_number(Js) == float("inf")


def test_ellipse_points_have_expected_extent():
    J = jc.planar_jacobian([1.0, 0.8], [0.3, 1.1])
    pts = jc.ellipse_points(J, "velocity", n=361)
    s = jc.singular_values(J)
    assert np.isclose(np.linalg.norm(pts, axis=0).max(), s[0], atol=1e-3)
    assert np.isclose(np.linalg.norm(pts, axis=0).min(), s[1], atol=1e-3)
    fpts = jc.ellipse_points(J, "force", n=361)
    assert np.isclose(np.linalg.norm(fpts, axis=0).max(), 1 / s[1], atol=1e-3)
    # every force on the force ellipse needs unit joint-torque norm
    assert np.allclose(np.linalg.norm(J.T @ fpts, axis=0), 1.0, atol=1e-9)
    X, Y, Z = jc.ellipsoid_surface(RNG.normal(size=(3, 5)), n=10)
    assert X.shape == Y.shape == Z.shape


def test_pseudoinverse_properties_and_minimum_norm():
    J = RNG.normal(size=(2, 3))
    Jp = jc.pinv(J)
    assert np.allclose(Jp, J.T @ np.linalg.inv(J @ J.T))
    for lhs, rhs in [(J @ Jp @ J, J), (Jp @ J @ Jp, Jp), ((J @ Jp).T, J @ Jp), ((Jp @ J).T, Jp @ J)]:
        assert np.allclose(lhs, rhs)
    xd = np.array([0.3, -0.2])
    qd = Jp @ xd
    z = RNG.normal(size=3)
    qd_other = qd + jc.nullspace_projector(J) @ z
    assert np.allclose(J @ qd_other, xd) and np.linalg.norm(qd_other) >= np.linalg.norm(qd) - 1e-12
    assert np.allclose(J @ jc.nullspace_projector(J), 0, atol=1e-12)


def test_tall_pseudoinverse_is_least_squares():
    J = RNG.normal(size=(4, 2))
    xd = RNG.normal(size=4)
    assert np.allclose(jc.pinv(J) @ xd, np.linalg.lstsq(J, xd, rcond=None)[0])
    assert np.allclose(jc.pinv(J), np.linalg.inv(J.T @ J) @ J.T)


def test_dls_matches_textbook_formula_and_limits():
    J = RNG.normal(size=(3, 6))
    lam = 0.3
    assert np.allclose(jc.dls_pinv(J, lam), J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(3)))
    assert np.allclose(jc.dls_pinv(J, 1e-9), jc.pinv(J), atol=1e-6)
    Js = jc.planar_jacobian([1.0, 1.0], [0.3, 0.0])                  # singular
    assert np.abs(jc.dls_pinv(Js, 0.05)).max() <= 1 / (2 * 0.05) + 1e-9
    assert jc.adaptive_damping(Js, eps=0.1, lam_max=0.2) == pytest.approx(0.2)
    assert jc.adaptive_damping(RNG.normal(size=(2, 2)) + 5 * np.eye(2), eps=0.01) == 0.0


def test_weighted_pinv_minimises_weighted_norm():
    J = RNG.normal(size=(2, 4))
    W = np.diag([1.0, 2.0, 5.0, 0.5])
    xd = np.array([0.4, -0.7])
    qd_w = jc.weighted_pinv(J, W) @ xd
    qd_0 = jc.pinv(J) @ xd
    assert np.allclose(J @ qd_w, xd)
    assert qd_w @ W @ qd_w <= qd_0 @ W @ qd_0 + 1e-12
    assert np.allclose(jc.weighted_pinv(J, np.eye(4)), jc.pinv(J))


# ---------------------------------------------------------------------------
# Symbolic spot check of the PUMA-like arm decoupling: det J = det J11 * det J22
# ---------------------------------------------------------------------------
def test_puma_like_arm_block_triangular_at_wrist_centre():
    q = RNG.uniform(-1, 1, 6)
    J = jc.jacobian_geometric(jc.PUMA_LIKE_DH, q)
    Ts = kin.fk_dh(jc.PUMA_LIKE_DH, q)
    pw = Ts[4][:3, 3]                       # the three wrist axes meet at the origin of frame {4}
    Jw = jc.shift_reference_point(J, pw - Ts[-1][:3, 3])
    assert np.allclose(Jw[:3, 3:], 0, atol=1e-12)
    assert np.isclose(np.linalg.det(J), np.linalg.det(Jw[:3, :3]) * np.linalg.det(Jw[3:, 3:]))
    # wrist singularity at q5 = 0
    q[4] = 0.0
    assert abs(np.linalg.det(jc.jacobian_geometric(jc.PUMA_LIKE_DH, q))) < 1e-12


# ---------------------------------------------------------------------------
# MuJoCo cross-checks (skipped when MuJoCo or the cached models are unavailable)
# ---------------------------------------------------------------------------
def _model(name):
    mujoco = pytest.importorskip("mujoco")
    from rgkit import robots
    spec = robots.ROBOTS[name]
    root = robots.cache_dir() / f"menagerie-{robots.MENAGERIE_SHA[:8]}" / spec["dir"] / ".complete"
    local = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if not root.exists() and not (local and (Path(local) / spec["dir"]).exists()):
        pytest.skip("robot model not cached")
    return robots.load_robot(name, scene="bare")


def test_ur5e_jacobian_matches_mujoco():
    import mujoco
    m, d = _model("ur5e")
    R_wb = jc.UR5E_BASE_IN_WORLD[:3, :3]
    for _ in range(8):
        q = RNG.uniform(-np.pi, np.pi, 6)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        T = jc.UR5E_BASE_IN_WORLD @ kin.fk_tool(jc.UR5E_DH, q, tool=jc.UR5E_TOOL)
        assert np.allclose(T[:3, 3], d.site("attachment_site").xpos, atol=1e-9)
        assert np.allclose(T[:3, :3], d.site("attachment_site").xmat.reshape(3, 3), atol=1e-9)
        J = jc.jacobian_in_frame(jc.jacobian_geometric(jc.UR5E_DH, q, tool=jc.UR5E_TOOL), R_wb)
        assert np.allclose(J, jc.mujoco_jacobian(m, d, site="attachment_site"), atol=1e-9)


def test_panda_jacobian_matches_mujoco():
    import mujoco
    m, d = _model("panda")
    for _ in range(8):
        q = np.zeros(m.nq)
        q[:7] = RNG.uniform(-2.0, 2.0, 7)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        T = kin.fk_tool(jc.PANDA_DH, q[:7], tool=jc.PANDA_TOOL)
        Th = tf.make_T(d.body("hand").xmat.reshape(3, 3), d.body("hand").xpos) @ tf.transl(0, 0, 0.1034)
        assert np.allclose(T, Th, atol=1e-6)
        J = jc.jacobian_geometric(jc.PANDA_DH, q[:7], tool=jc.PANDA_TOOL)
        Jm = jc.mujoco_jacobian(m, d, body="hand", point=Th[:3, 3])
        assert np.allclose(J, Jm[:, :7], atol=1e-6) and np.allclose(Jm[:, 7:], 0)


def test_ur5e_static_torque_matches_mujoco_applyFT():
    import mujoco
    m, d = _model("ur5e")
    q = RNG.uniform(-2, 2, 6)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    F = np.r_[RNG.normal(size=3), 0.2 * RNG.normal(size=3)]       # world-frame wrench at the tool origin
    J = jc.mujoco_jacobian(m, d, site="attachment_site")
    out = np.zeros(m.nv)
    mujoco.mj_applyFT(m, d, F[:3], F[3:], d.site("attachment_site").xpos, m.site("attachment_site").bodyid[0], out)
    assert np.allclose(out, J.T @ F, atol=1e-9)

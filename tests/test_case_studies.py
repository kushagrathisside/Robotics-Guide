"""Tests for rgkit.case_studies (Chapter 8 helpers).

Everything here is self-contained: the MuJoCo tests use a small MJCF string, so no robot model has to be
downloaded.  A last group of tests uses the cached UR5e and is skipped when the model is not available.
"""
import numpy as np
import pytest

from rgkit import case_studies as cs
from rgkit import kinematics as kin
from rgkit import transforms as tf

deg = np.deg2rad
rng = np.random.default_rng(7)


def random_table(n=7, special=True):
    rows = []
    for _ in range(n):
        alpha = rng.choice([-np.pi / 2, np.pi / 2, 0.0]) if special else rng.uniform(-np.pi, np.pi)
        a = rng.uniform(0.05, 0.4) * (rng.random() > 0.3)
        d = rng.uniform(-0.3, 0.3) * (rng.random() > 0.3)
        rows.append((alpha, a, d, rng.uniform(-np.pi, np.pi), "R"))
    return rows


# ------------------------------------------------------------------ pose error, Jacobian, IK
def test_pose_error_roundtrip():
    T1 = tf.make_T(tf.rotz(0.3) @ tf.rotx(-0.2), [0.1, 0.2, 0.3])
    T2 = tf.make_T(tf.rotz(0.5) @ tf.rotx(-0.1), [0.3, -0.1, 0.4])
    e = cs.pose_error(T1, T2)
    assert np.allclose(e[:3], [0.2, -0.3, 0.1])
    R = tf.rot_axis_angle(e[3:] / np.linalg.norm(e[3:]), np.linalg.norm(e[3:])) @ T1[:3, :3]
    assert np.allclose(R, T2[:3, :3])                      # the rotation vector carries R1 to R2
    assert np.allclose(cs.pose_error(T1, T1), 0.0)


def test_jacobian_matches_finite_differences():
    table = random_table(7, special=False)
    tool = tf.transl(0, 0, 0.1)
    q = rng.uniform(-1, 1, 7)
    J = cs.jacobian_dh(table, q, tool=tool)
    h = 1e-6
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = h
        T1, T2 = kin.fk_tool(table, q + dq, tool=tool), kin.fk_tool(table, q - dq, tool=tool)
        v = (T1[:3, 3] - T2[:3, 3]) / (2 * h)
        w_skew = (T1[:3, :3] - T2[:3, :3]) / (2 * h) @ kin.fk_tool(table, q, tool=tool)[:3, :3].T
        assert np.allclose(J[:3, i], v, atol=1e-6)
        assert np.allclose(J[3:, i], tf.unskew(w_skew), atol=1e-6)


def test_jacobian_prismatic_column():
    table = [(0, 0, 0.2, 0, "R"), (-np.pi / 2, 0, 0.1, 0, "P")]
    q = [0.4, 0.3]
    J = cs.jacobian_dh(table, q)
    Ts = kin.fk_dh(table, q)
    assert np.allclose(J[:3, 1], Ts[2][:3, 2]) and np.allclose(J[3:, 1], 0.0)


def test_ik_dls_reaches_reachable_pose():
    table = [(0, 0, 0.16, 0, "R"), (np.pi / 2, 0, 0, 0, "R"), (0, -0.42, 0, 0, "R"),
             (0, -0.39, 0.13, 0, "R"), (np.pi / 2, 0, 0.1, 0, "R"), (-np.pi / 2, 0, 0.1, 0, "R")]
    q_true = np.array([0.4, -1.1, 1.5, -0.9, -1.2, 0.3])
    T = kin.fk_tool(table, q_true)
    res = cs.ik_dls(lambda q: kin.fk_tool(table, q), lambda q: cs.jacobian_dh(table, q), T, q_true + 0.3)
    assert res.converged and res.iterations < 100
    assert np.allclose(kin.fk_tool(table, res.q), T, atol=1e-5)
    assert res.errors[-1] < res.errors[0]


def test_ik_dls_reports_failure_for_unreachable_pose():
    table = [(0, 0, 0, 0, "R"), (0, 0.5, 0, 0, "R")]
    T = tf.make_T(np.eye(3), [5.0, 0, 0])
    res = cs.ik_dls(lambda q: kin.fk_tool(table, q), lambda q: cs.jacobian_dh(table, q), T, [0.1, 0.1], max_iter=30)
    assert not res.converged


def test_sigma_min_and_manipulability_of_planar_2r():
    L1, L2 = 1.0, 0.8
    table = [(0, 0, 0, 0, "R"), (0, L1, 0, 0, "R")]
    tool = tf.transl(L2)
    for q2 in (0.0, 0.7, np.pi):
        J = cs.jacobian_dh(table, [0.3, q2], tool=tool)[[0, 1, 5]]       # x, y, and rotation about z
        assert np.isclose(abs(np.linalg.det(J[:2, :2])), L1 * L2 * abs(np.sin(q2)))
    J0 = cs.jacobian_dh(table, [0.3, 0.0], tool=tool)
    assert cs.sigma_min(J0[:2]) < 1e-12            # stretched arm is singular
    assert cs.manipulability(J0[:2]) < 1e-12


# ------------------------------------------------------------------ reading a D-H table off axes
@pytest.mark.parametrize("special", [True, False])
def test_dh_from_axes_reproduces_random_chain(special):
    for _ in range(5):
        table = random_table(7, special)
        Ts = kin.fk_dh(table, np.zeros(7))
        omegas = np.array([T[:3, 2] for T in Ts[1:]])
        points = np.array([T[:3, 3] for T in Ts[1:]])
        fit = cs.dh_from_axes(omegas, points)
        n = len(table)
        T_tool = tf.inv_T(fit.T_base_0 @ kin.fk_dh(fit.table, np.zeros(n))[-1]) @ Ts[-1]
        for _ in range(5):
            q = rng.uniform(-1.5, 1.5, n)
            assert np.allclose(kin.fk_dh(table, q)[-1], fit.T_base_0 @ kin.fk_dh(fit.table, q)[-1] @ T_tool, atol=1e-9)


def test_dh_from_axes_parallel_and_collinear_axes():
    # a UR-like chain: axes 2, 3, 4 parallel; axes 1 and 5 collinear with each other's perpendicular
    table = [(0, 0, 0.16, 0, "R"), (np.pi / 2, 0, 0, 0, "R"), (0, -0.42, 0, 0, "R"),
             (0, -0.39, 0.13, 0, "R"), (np.pi / 2, 0, 0.1, 0, "R"), (-np.pi / 2, 0, 0.1, 0, "R")]
    Ts = kin.fk_dh(table, np.zeros(6))
    fit = cs.dh_from_axes(np.array([T[:3, 2] for T in Ts[1:]]), np.array([T[:3, 3] for T in Ts[1:]]))
    a = [abs(r[1]) for r in fit.table]
    assert np.allclose(sorted(a), sorted([0, 0, 0, 0.42, 0.39, 0]), atol=1e-9)
    # collinear consecutive axes: two stacked joints about the same vertical line
    tab2 = [(0, 0, 0.1, 0.0, "R"), (0, 0, 0.2, 0.0, "R"), (np.pi / 2, 0.1, 0, 0.0, "R")]
    Ts = kin.fk_dh(tab2, np.zeros(3))
    fit = cs.dh_from_axes(np.array([T[:3, 2] for T in Ts[1:]]), np.array([T[:3, 3] for T in Ts[1:]]))
    q = [0.3, -0.4, 0.5]
    T_tool = tf.inv_T(fit.T_base_0 @ kin.fk_dh(fit.table, np.zeros(3))[-1]) @ Ts[-1]
    assert np.allclose(kin.fk_dh(tab2, q)[-1], fit.T_base_0 @ kin.fk_dh(fit.table, q)[-1] @ T_tool, atol=1e-9)


# ------------------------------------------------------------------ product of exponentials
def test_poe_matches_dh():
    table = random_table(6, special=False)
    Ts = kin.fk_dh(table, np.zeros(6))
    om = np.array([T[:3, 2] for T in Ts[1:]])
    pt = np.array([T[:3, 3] for T in Ts[1:]])
    S = cs.screw_axes(om, pt)
    assert np.allclose(S[:, :3], om) and np.allclose(S[:, 3:], np.cross(pt, om))
    for _ in range(5):
        q = rng.uniform(-1.5, 1.5, 6)
        assert np.allclose(cs.poe_fk(S, Ts[-1], q), kin.fk_dh(table, q)[-1], atol=1e-10)


def test_exp_screw_pure_rotation_and_translation():
    T = cs.exp_screw([0, 0, 1, 0, 0, 0], 0.5)
    assert np.allclose(T[:3, :3], tf.rotz(0.5)) and np.allclose(T[:3, 3], 0)
    T = cs.exp_screw([0, 0, 1, 1, 0, 0], np.pi / 2)         # rotation about an axis through (0, 1, 0)... v = -w x p
    assert tf.is_transform(T)


# ------------------------------------------------------------------ effective mass
def test_effective_mass_of_point_mass_on_a_2_link_arm():
    # the 1-D case: one prismatic joint of mass m seen at the tip -> m_eff = m in the joint direction
    M = np.array([[2.0]])
    J = np.array([[1.0], [0.0], [0.0]])
    assert np.isclose(cs.effective_mass(M, J, [1, 0, 0]), 2.0)
    # two joints driving one point along x with J = [1, 2]: m_eff = 1 / (J M^-1 J^T)
    M = np.diag([1.0, 4.0])
    J = np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
    assert np.isclose(cs.effective_mass(M, J, [1, 0, 0]), 1.0 / (1.0 + 4.0 / 4.0))
    Lam = cs.operational_inertia(np.diag([1.0, 4.0]), np.array([[1.0, 2.0], [0.0, 1.0]]))
    assert np.allclose(Lam, np.linalg.inv(np.array([[1.0, 2.0], [0.0, 1.0]]) @ np.diag([1, 0.25]) @ np.array([[1.0, 2.0], [0.0, 1.0]]).T))


# ------------------------------------------------------------------ support polygon
def test_support_polygon_and_margin():
    pts = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5), (0.2, 0.7)]
    poly = cs.support_polygon(pts)
    assert len(poly) == 4
    area = 0.5 * sum(poly[i][0] * poly[(i + 1) % 4][1] - poly[(i + 1) % 4][0] * poly[i][1] for i in range(4))
    assert np.isclose(area, 1.0)                          # counter-clockwise, unit square
    assert np.isclose(cs.polygon_margin([0.5, 0.5], poly), 0.5)
    assert np.isclose(cs.polygon_margin([0.9, 0.5], poly), 0.1)
    assert np.isclose(cs.polygon_margin([1.5, 0.5], poly), -0.5)
    assert np.isclose(cs.polygon_margin([2.0, 2.0], poly), -np.sqrt(2.0))      # outside, nearest to a corner
    assert cs.polygon_margin([1.0, 0.5], poly) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ trajectories
def test_quintic_boundary_conditions():
    s, ds = cs.quintic(np.array([0.0, 0.5, 1.0]))
    assert np.allclose(s, [0, 0.5, 1]) and np.allclose(ds[[0, 2]], 0)
    h = 1e-6
    s1, _ = cs.quintic(1e-4)
    assert s1 < 1e-9                                       # flat start: s ~ 10 tau^3


def test_rest_to_rest_path():
    path = cs.RestToRestPath([[0, 0], [1, 2], [1, 2], [0, 0]], [1.0, 0.5, 2.0])
    assert path.duration == pytest.approx(3.5)
    q, qd = path(0.0)
    assert np.allclose(q, 0) and np.allclose(qd, 0)
    q, qd = path(1.0)
    assert np.allclose(q, [1, 2]) and np.allclose(qd, 0, atol=1e-9)
    q, qd = path(1.25)                                      # dwell segment: stays put
    assert np.allclose(q, [1, 2]) and np.allclose(qd, 0)
    q, qd = path(10.0)
    assert np.allclose(q, 0)
    # velocity is the derivative of position
    h = 1e-6
    t = 0.37
    assert np.allclose((path(t + h)[0] - path(t - h)[0]) / (2 * h), path(t)[1], atol=1e-5)


def test_save_gif_only_when_asked(tmp_path, monkeypatch):
    frames = [np.zeros((20, 30, 3), np.uint8), np.full((20, 30, 3), 255, np.uint8)]
    monkeypatch.delenv("RG_SAVE_ASSETS", raising=False)
    assert cs.save_gif(frames, tmp_path / "a.gif") is False and not (tmp_path / "a.gif").exists()
    monkeypatch.setenv("RG_SAVE_ASSETS", "1")
    assert cs.save_gif(frames, tmp_path / "a.gif") is True and (tmp_path / "a.gif").exists()


# ------------------------------------------------------------------ MuJoCo (small inline model, no download)
mujoco = pytest.importorskip("mujoco")

ARM = """
<mujoco><option gravity="0 0 -9.81"/>
<worldbody>
  <body name="b1" pos="0 0 0.1"><joint name="j1" axis="0 0 1" armature="0.02"/>
    <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.03" mass="2"/>
    <body name="b2" pos="0.5 0 0"><joint name="j2" axis="0 1 0" armature="0.03"/>
      <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.03" mass="1.5"/>
      <body name="b3" pos="0.4 0 0"><joint name="j3" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.5"/>
        <site name="tip" pos="0.2 0 0"/></body></body></body>
</worldbody></mujoco>"""


def _arm():
    m = mujoco.MjModel.from_xml_string(ARM)
    d = mujoco.MjData(m)
    d.qpos[:] = [0.4, -0.5, 0.8]
    mujoco.mj_forward(m, d)
    return m, d


def test_site_jacobian_matches_finite_differences_of_mujoco_fk():
    m, d = _arm()
    J = cs.site_jacobian(m, d, "tip")
    sid = m.site("tip").id
    h = 1e-6
    for i in range(3):
        qp = d.qpos.copy()
        d.qpos[i] += h
        mujoco.mj_forward(m, d)
        p1 = d.site_xpos[sid].copy()
        d.qpos[:] = qp
        d.qpos[i] -= h
        mujoco.mj_forward(m, d)
        p2 = d.site_xpos[sid].copy()
        d.qpos[:] = qp
        assert np.allclose(J[:3, i], (p1 - p2) / (2 * h), atol=1e-6)
    mujoco.mj_forward(m, d)
    Jp = cs.point_jacobian(m, d, "b3", d.site_xpos[sid])
    assert np.allclose(Jp, J, atol=1e-9)


def test_mass_matrix_symmetric_positive_and_effective_mass_by_dynamics():
    m, d = _arm()
    M = cs.mass_matrix(m, d)
    assert np.allclose(M, M.T) and np.all(np.linalg.eigvalsh(M) > 0)
    # push the tip with a force F along u and compare the extra acceleration with F / m_eff
    J = cs.site_jacobian(m, d, "tip")
    u = np.array([0.3, 0.5, 0.8])
    u /= np.linalg.norm(u)
    sid = m.site("tip").id
    mujoco.mj_forward(m, d)
    qacc0 = d.qacc.copy()
    F = 2.0
    d.xfrc_applied[m.site_bodyid[sid], :3] = F * u          # applied at the body origin ...
    # ... so apply it at the tip instead: use the Jacobian transpose through qfrc_applied
    d.xfrc_applied[:] = 0
    d.qfrc_applied[:] = J[:3].T @ (F * u)
    mujoco.mj_forward(m, d)
    a_extra = J[:3] @ (d.qacc - qacc0)
    assert np.isclose(a_extra @ u, F / cs.effective_mass(M, J[:3], u), rtol=1e-6)
    assert cs.effective_mass(M, J[:3], u) < m.body_subtreemass[1]       # lighter than the whole arm


def test_chain_joint_ids_and_axes():
    m, d = _arm()
    ids = cs.chain_joint_ids(m, "b3")
    assert ids == [0, 1, 2]
    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    om, pt = cs.joint_axes(m, d, ids)
    assert np.allclose(om, [[0, 0, 1], [0, 1, 0], [0, 1, 0]])
    assert np.allclose(pt, [[0, 0, 0.1], [0.5, 0, 0.1], [0.9, 0, 0.1]])


def test_dh_fit_of_mujoco_model_reproduces_forward_kinematics():
    m, d = _arm()
    ids = cs.chain_joint_ids(m, "b3")
    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    om, pt = cs.joint_axes(m, d, ids)
    fit = cs.dh_from_axes(om, pt)
    sid = m.site("tip").id

    def T_model(q):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        return tf.make_T(d.site_xmat[sid].reshape(3, 3), d.site_xpos[sid])

    T0 = T_model(np.zeros(3))
    T_tool = tf.inv_T(fit.T_base_0 @ kin.fk_dh(fit.table, np.zeros(3))[-1]) @ T0
    for _ in range(5):
        q = rng.uniform(-1.5, 1.5, 3)
        assert np.allclose(T_model(q), fit.T_base_0 @ kin.fk_dh(fit.table, q)[-1] @ T_tool, atol=1e-9)
    S = cs.screw_axes(om, pt)
    q = rng.uniform(-1, 1, 3)
    assert np.allclose(T_model(q), cs.poe_fk(S, T0, q), atol=1e-9)


# ------------------------------------------------------------------ real model (skipped if not cached)
def _ur5e():
    try:
        from rgkit import robots
        return robots.load_robot("ur5e", scene="bare")
    except Exception:                                    # no network and no cache
        pytest.skip("UR5e model not available")


def test_ur5e_modified_dh_table_matches_the_model():
    m, d = _ur5e()
    d1, a2, a3, d4, d5, d6 = 0.163, -0.425, -0.392, 0.134, 0.1, 0.1
    table = [(0, 0, d1, 0, "R"), (np.pi / 2, 0, 0, 0, "R"), (0, a2, 0, 0, "R"),
             (0, a3, d4, 0, "R"), (np.pi / 2, 0, d5, 0, "R"), (-np.pi / 2, 0, d6, 0, "R")]
    sid = m.site("attachment_site").id
    for _ in range(5):
        q = rng.uniform(-3, 3, 6)
        d.qpos[:6] = q
        mujoco.mj_forward(m, d)
        T = tf.make_T(d.site_xmat[sid].reshape(3, 3), d.site_xpos[sid])
        assert np.allclose(T, kin.fk_dh(table, q)[-1], atol=1e-9)
        J = cs.site_jacobian(m, d, "attachment_site")
        assert np.allclose(J, cs.jacobian_dh(table, q), atol=1e-9)
        c2, c23, s234 = np.cos(q[1]), np.cos(q[1] + q[2]), np.sin(q[1] + q[2] + q[3])
        det = a2 * a3 * np.sin(q[2]) * np.sin(q[4]) * (a2 * c2 + a3 * c23 + d5 * s234)
        assert np.isclose(np.linalg.det(J), det, rtol=1e-8, atol=1e-12)


# ------------------------------------------------------------------ self-motion of a redundant arm
def test_self_motion_keeps_the_tool_pose_and_swivel_circle():
    # a spherical-revolute-spherical 7R arm without offsets: elbow must move on an exact circle
    h = np.pi / 2
    L2, L4, L6 = 0.36, 0.37, 0.2
    table = [(0, 0, 0, 0, "R"), (-h, 0, 0, 0, "R"), (h, 0, L2, 0, "R"), (-h, 0, 0, 0, "R"),
             (h, 0, L4, 0, "R"), (-h, 0, 0, 0, "R"), (h, 0, 0, 0, "R")]
    tool = tf.transl(0, 0, L6)
    fk = lambda q: kin.fk_tool(table, q, tool=tool)
    jac = lambda q: cs.jacobian_dh(table, q, tool=tool)
    q0 = np.array([0.3, 0.5, -0.4, 0.9, 0.6, -0.7, 0.2])
    J = jac(q0)
    assert np.linalg.matrix_rank(J) == 6
    n = cs.null_vector(J)
    assert np.allclose(J @ n, 0, atol=1e-12) and np.isclose(np.linalg.norm(n), 1.0)
    path = cs.self_motion(fk, jac, q0, n_steps=80, ds=0.03)
    T0 = fk(q0)
    assert max(np.linalg.norm(cs.pose_error(fk(q), T0)) for q in path) < 1e-9
    assert np.linalg.norm(path[-1] - path[0]) > 0.5                      # it really moved
    radii, phis = [], []
    for q in path:
        Ts = kin.fk_dh(table, q)
        phi, r, dsw = cs.swivel_angle(Ts[2][:3, 3], Ts[4][:3, 3], Ts[6][:3, 3])
        radii.append(r)
        phis.append(phi)
    assert np.ptp(radii) < 1e-9                                           # exact circle
    assert np.ptp(np.unwrap(phis)) > 0.5                                  # and it is swept through a large angle
    back = cs.self_motion(fk, jac, q0, n_steps=10, ds=-0.03)
    assert np.sign(np.unwrap(phis)[10] - phis[0]) != np.sign(np.unwrap([phis[0]] + [cs.swivel_angle(
        *(kin.fk_dh(table, q)[i][:3, 3] for i in (2, 4, 6)))[0] for q in back[1:]])[-1] - phis[0])


def test_swivel_angle_geometry():
    s, w = np.array([0, 0, 0.0]), np.array([0.5, 0, 0.0])
    phi, r, d = cs.swivel_angle(s, [0.25, 0.0, 0.2], w)                    # elbow straight up
    assert np.isclose(phi, 0.0) and np.isclose(r, 0.2) and np.isclose(d, 0.5)
    phi, r, _ = cs.swivel_angle(s, [0.25, 0.2, 0.0], w)
    assert np.isclose(abs(phi), np.pi / 2)

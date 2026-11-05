import numpy as np
import pytest

from rgkit import dh_tools as dh
from rgkit import kinematics as kin
from rgkit import transforms as tf

deg = np.deg2rad
RNG = np.random.default_rng(7)

# classic tables used throughout Chapter 3 (modified convention, rows alpha_{i-1}, a_{i-1}, d_i, theta_i)
PUMA = [(0, 0, 0, 0, "R"), (-np.pi / 2, 0, 0, 0, "R"), (0, 0.4318, 0.15, 0, "R"),
        (-np.pi / 2, 0.0203, 0.4318, 0, "R"), (np.pi / 2, 0, 0, 0, "R"), (-np.pi / 2, 0, 0, 0, "R")]
SCARA = [(0, 0, 0, 0, "R"), (0, 0.4, 0, 0, "R"), (np.pi, 0.3, 0.1, 0, "P"), (0, 0, 0.05, 0, "R")]
SPHERICAL = [(0, 0, 0, 0, "R"), (-np.pi / 2, 0, 0, 0, "R"), (np.pi / 2, 0, 0.5, 0, "P")]
CYLINDRICAL = [(0, 0, 0, 0, "R"), (0, 0, 0.2, 0, "P"), (-np.pi / 2, 0, 0.3, 0, "P")]
PLANAR3 = [(0, 0, 0, 0, "R"), (0, 0.5, 0, 0, "R"), (0, 0.4, 0, 0, "R")]


def random_table(n, kinds=None):
    kinds = kinds or ["R"] * n
    return [(RNG.uniform(-np.pi, np.pi), RNG.uniform(0.0, 0.6), RNG.uniform(-0.5, 0.5), RNG.uniform(-1, 1), k)
            for k in kinds]


def random_q(table):
    return np.array([RNG.uniform(-np.pi, np.pi) if r[4] == "R" else RNG.uniform(-0.2, 0.2) for r in table])


def axes_of(table, convention="modified"):
    """World joint axes (point, direction) of a table at q = 0."""
    Ts = kin.fk_dh(table, None, convention)
    pts, dirs = [], []
    for i in range(1, len(table) + 1):
        if convention == "modified":
            T = Ts[i]
        else:
            T = Ts[i - 1]
        pts.append(T[:3, 3]); dirs.append(T[:3, 2])
    return pts, dirs


# ---------------------------------------------------------------------------
def test_parameter_roundtrip_modified_and_standard():
    for _ in range(50):
        al, a, d, th = RNG.uniform(-np.pi, np.pi), RNG.uniform(-1, 1), RNG.uniform(-1, 1), RNG.uniform(-np.pi, np.pi)
        assert np.allclose(dh.dh_params_modified(kin.dh_modified(al, a, d, th)), (al, a, d, th))
        assert np.allclose(dh.dh_params_standard(kin.dh_standard(al, a, d, th)), (al, a, d, th))


def test_non_dh_transform_is_rejected_and_violations_measured():
    T = tf.make_T(tf.rotx(0.4) @ tf.roty(0.3), [0.2, 0.1, 0.3])      # a generic pose: not of D-H form
    with pytest.raises(ValueError):
        dh.dh_params_modified(T)
    with pytest.raises(ValueError):
        dh.dh_params_standard(T)
    assert max(dh.dh_violations(T, "modified")) > 1e-3
    good = kin.dh_modified(0.7, 0.3, 0.2, -0.4)
    assert max(dh.dh_violations(good, "modified")) < 1e-12
    good = kin.dh_standard(0.7, 0.3, 0.2, -0.4)
    assert max(dh.dh_violations(good, "standard")) < 1e-12
    # the same transform read with the other convention's rules generally violates them
    assert max(dh.dh_violations(kin.dh_modified(0.7, 0.3, 0.2, -0.4), "standard")) > 1e-3


def test_modified_and_standard_tables_are_a_row_shift_of_each_other():
    for _ in range(10):
        tab = random_table(5)
        tab[0] = (0.0, 0.0, tab[0][2], tab[0][3], "R")                 # alpha_0 = a_0 = 0
        std = dh.modified_to_standard(tab)
        q = random_q(tab)
        assert np.allclose(kin.fk_dh(tab, q, "modified")[-1], kin.fk_dh(std, q, "standard")[-1])
        back, tool = dh.standard_to_modified(std)
        assert np.allclose([r[:4] for r in back], [r[:4] for r in tab]) and np.allclose(tool, 0)
    with pytest.raises(ValueError):
        dh.modified_to_standard([(0.3, 0.0, 0, 0, "R")])


@pytest.mark.parametrize("convention", ["modified", "standard"])
@pytest.mark.parametrize("x_sign", ["craig", "continuous"])
@pytest.mark.parametrize("name,table", [("generic6", None), ("puma", PUMA), ("scara", SCARA), ("spherical", SPHERICAL),
                                         ("cylindrical", CYLINDRICAL), ("planar3", PLANAR3)])
def test_frame_assignment_reproduces_the_robot(convention, x_sign, name, table):
    table = random_table(6) if table is None else table
    table = [(0.0, 0.0, 0.0, 0.0, table[0][4])] + list(table[1:]) if convention == "modified" else table
    kinds = [r[4] for r in table]
    pts, dirs = axes_of(table, "modified")
    # reference robot: modified table, world frame = base frame {0}; tool = frame {N}
    tool_point = kin.fk_dh(table, None, "modified")[-1][:3, 3]
    tab2, frames = dh.dh_from_axes(pts, dirs, convention, kinds, tool_point=tool_point, x_sign=x_sign)
    T_ref0 = kin.fk_dh(table, None, "modified")[-1]
    C = tf.inv_T(frames[-1]) @ T_ref0                                # constant tool offset between the two {N}
    for _ in range(25):
        q = random_q(table)
        A = kin.fk_dh(table, q, "modified")[-1]
        B = frames[0] @ kin.fk_dh(tab2, q, convention)[-1] @ C
        assert np.allclose(A, B, atol=1e-9), (name, convention, x_sign)
    # the assigned frames obey the D-H rules
    for i in range(1, len(frames)):
        assert max(dh.dh_violations(tf.inv_T(frames[i - 1]) @ frames[i], convention)) < 1e-9
    if x_sign == "craig":
        assert all(r[1] >= -1e-12 for r in (tab2 if convention == "modified" else tab2))


def test_craig_signs_give_nonnegative_link_lengths():
    table = random_table(5)
    table[0] = (0, 0, 0, 0, "R")
    pts, dirs = axes_of(table)
    tab2, _ = dh.dh_from_axes(pts, dirs, "modified")
    assert all(r[1] >= -1e-12 for r in tab2)


def test_first_row_and_offsets_for_known_planar_arm():
    pts, dirs = axes_of(PLANAR3)
    tab, frames = dh.dh_from_axes(pts, dirs, "modified", tool_point=[0.9, 0, 0])
    assert np.allclose([r[:4] for r in tab], [(0, 0, 0, 0), (0, 0.5, 0, 0), (0, 0.4, 0, 0)], atol=1e-12)
    # moving {0} down along axis 1 gives d_1 != 0
    tab, frames = dh.dh_from_axes(pts, dirs, "modified", base_point=[0, 0, -0.2])
    assert np.isclose(tab[0][2], 0.2)


def test_common_normal_cases():
    f1, f2, kind = dh.common_normal([0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0])      # skew
    assert kind == "skew" and np.allclose(f1, [0, 0, 0]) and np.allclose(f2, [1, 0, 0])
    f1, f2, kind = dh.common_normal([0, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0])      # intersecting
    assert kind == "intersecting" and np.allclose(f1, f2)
    f1, f2, kind = dh.common_normal([0, 0, 0], [0, 0, 1], [1, 0, 2], [0, 0, 1])      # parallel
    assert kind == "parallel" and np.allclose(f2, [1, 0, 0])


def test_pose_error():
    A = tf.make_T(tf.rotz(0.3), [1, 2, 3])
    B = tf.make_T(tf.rotz(0.3 + 1e-6), [1, 2, 3 + 2e-6])
    dt, dr = dh.pose_error(A, B)
    assert np.isclose(dt, 2e-6) and np.isclose(dr, 1e-6, rtol=1e-4)
    dt, dr = dh.pose_error(np.stack([A, A]), np.stack([A, B]))
    assert dt[0] == 0 and dr[0] == 0 and dt[1] > 0


def test_wrap_angle():
    assert np.isclose(dh.wrap_angle(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(dh.wrap_angle(np.pi), np.pi)


# ---------------------------------------------------------------------------
# symbolic
# ---------------------------------------------------------------------------
def test_symbolic_link_transform_matches_numeric():
    import sympy as sp
    al, a, d, th = sp.symbols("alpha a d theta")
    vals = {al: 0.7, a: 0.3, d: 0.2, th: -0.4}
    num = np.array(dh.sym_dh_modified(al, a, d, th).subs(vals).evalf(), dtype=float)
    assert np.allclose(num, kin.dh_modified(0.7, 0.3, 0.2, -0.4))
    num = np.array(dh.sym_dh_standard(al, a, d, th).subs(vals).evalf(), dtype=float)
    assert np.allclose(num, kin.dh_standard(0.7, 0.3, 0.2, -0.4))


def test_symbolic_planar_3r_and_cs_notation():
    import sympy as sp
    t1, t2, t3, L1, L2 = sp.symbols("theta_1 theta_2 theta_3 L_1 L_2")
    table = [(0, 0, 0, t1), (0, L1, 0, t2), (0, L2, 0, t3)]
    T = dh.sym_fk(table)[-1]
    assert sp.simplify(T[0, 3] - (L1 * sp.cos(t1) + L2 * sp.cos(t1 + t2))) == 0
    s = dh.latex_cs(T[0, 3], [t1, t2, t3])
    assert "c_{12}" in s and "c_{1}" in s
    s = dh.latex_cs(sp.sin(t1 + t2 + t3) + sp.cos(t1 - t3), [t1, t2, t3])
    assert "s_{123}" in s and "c_{1-3}" in s
    assert dh.latex_cs(sp.sin(t2 + sp.pi / 3), [t1, t2, t3]) == sp.latex(sp.sin(t2 + sp.pi / 3))   # offsets are left alone


def test_symbolic_puma_position_matches_closed_form():
    import sympy as sp
    th = sp.symbols("theta_1:7")
    a2, a3, d3, d4 = sp.symbols("a_2 a_3 d_3 d_4")
    t = sp.pi / 2
    table = [(0, 0, 0, th[0]), (-t, 0, 0, th[1]), (0, a2, d3, th[2]), (-t, a3, d4, th[3]), (t, 0, 0, th[4]), (-t, 0, 0, th[5])]
    T = dh.sym_fk(table)[-1]
    c1, s1, c2, s2, c23, s23 = sp.cos(th[0]), sp.sin(th[0]), sp.cos(th[1]), sp.sin(th[1]), sp.cos(th[1] + th[2]), sp.sin(th[1] + th[2])
    px = c1 * (a2 * c2 + a3 * c23 - d4 * s23) - d3 * s1
    py = s1 * (a2 * c2 + a3 * c23 - d4 * s23) + d3 * c1
    pz = -a3 * s23 - a2 * s2 - d4 * c23
    assert sp.simplify(T[0, 3] - px) == 0 and sp.simplify(T[1, 3] - py) == 0 and sp.simplify(T[2, 3] - pz) == 0
    f = dh.lambdify_fk(T, list(th) + [a2, a3, d3, d4])
    q = RNG.uniform(-3, 3, 6)
    num = f(*q, 0.4318, 0.0203, 0.15, 0.4318)
    assert np.allclose(num, kin.fk_dh(PUMA, q)[-1])


# ---------------------------------------------------------------------------
# MuJoCo
# ---------------------------------------------------------------------------
mujoco = pytest.importorskip("mujoco")


@pytest.mark.parametrize("convention", ["modified", "standard"])
@pytest.mark.parametrize("name,table", [("puma", PUMA), ("scara", SCARA), ("spherical", SPHERICAL),
                                         ("cylindrical", CYLINDRICAL)])
def test_mjcf_from_dh_agrees_with_fk_dh(convention, name, table):
    tab = table if convention == "modified" else dh.modified_to_standard(table)
    xml = dh.mjcf_from_dh(tab, convention)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    qs = np.array([random_q(tab) for _ in range(20)])
    Tm = dh.mujoco_poses(model, data, qs, site="tool")
    Td = np.array([kin.fk_dh(tab, q, convention)[-1] for q in qs])
    dt, dr = dh.pose_error(Tm, Td)
    assert dt.max() < 1e-10 and dr.max() < 1e-10


def test_joint_axes_from_mujoco_on_a_small_model():
    xml = """<mujoco><worldbody><body pos="0 0 0.5"><joint name="a" axis="0 0 1"/><geom size="0.05"/>
    <body pos="0.3 0 0"><joint name="b" type="slide" axis="1 0 0"/><geom size="0.05"/></body></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    pts, dirs, kinds = dh.joint_axes_from_mujoco(model, data)
    assert kinds == ["R", "P"] and np.allclose(pts[0], [0, 0, 0.5]) and np.allclose(dirs[1], [1, 0, 0])

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from rgkit import joint_census as jc  # noqa: E402

ARM = """
<mujoco><option jacobian="dense"/><worldbody>
 <body name="b1"><joint name="j1" type="hinge" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 0 0 .3" size=".02" contype="0" conaffinity="0"/>
  <body name="b2" pos="0 0 .3"><joint name="j2" type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .4 0 0" size=".02" contype="0" conaffinity="0"/>
   <body name="b3" pos=".4 0 0"><joint name="j3" type="slide" axis="1 0 0" range="0 .2"/><geom size=".03" contype="0" conaffinity="0"/>
    <body name="b4"><joint name="j4" type="ball"/><geom size=".03" contype="0" conaffinity="0"/></body></body></body></body>
</worldbody></mujoco>"""

FOURBAR = """
<mujoco><option jacobian="dense"/><worldbody>
 <body name="crank"><joint name="j2" type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 1 0 0" size=".02" contype="0" conaffinity="0"/>
  <body name="coupler" pos="1 0 0"><joint name="j3" type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 2.5 0 0" size=".02" contype="0" conaffinity="0"/>
   <site name="B" pos="2.5 0 0"/></body></body>
 <body name="rocker" pos="2.2 0 0"><joint name="j4" type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 2 0 0" size=".02" contype="0" conaffinity="0"/>
   <site name="B2" pos="2 0 0"/></body>
</worldbody><equality><connect site1="B" site2="B2"/></equality></mujoco>"""


def _load(xml):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_joint_table_and_chain():
    m, d = _load(ARM)
    rows = jc.joint_table(m, d)
    assert [r["symbol"] for r in rows] == ["R", "R", "P", "S"]
    assert rows[2]["range"] == (0.0, 0.2) and rows[0]["range"] is None
    assert np.allclose(rows[1]["axis"], [0, 1, 0]) and np.allclose(rows[1]["anchor"], [0, 0, 0.3])
    assert "".join(r["symbol"] for r in jc.chain_joints(m, d, "b4")) == "RRPS"
    assert (m.nq, m.nv) == (1 + 1 + 1 + 4, 1 + 1 + 1 + 3)
    assert jc.describe_dofs(m)[:2] == (7, 6)
    tree = jc.kinematic_tree(m)
    assert tree["b1"] == ["b2"] and tree["world"] == ["b1"]


def test_axis_relation():
    d, ang = jc.axis_relation([0, 0, 0], [0, 0, 1], [0, 0, 0.3], [0, 1, 0])
    assert np.isclose(d, 0.0) and np.isclose(ang, np.pi / 2)
    assert jc.axis_kind(d, ang) == "perpendicular, intersecting"
    d, ang = jc.axis_relation([0, 0, 0], [0, 1, 0], [0.4, 0, 0.0], [0, 1, 0])
    assert np.isclose(d, 0.4) and np.isclose(ang, 0.0) and jc.axis_kind(d, ang) == "parallel"
    d, ang = jc.axis_relation([0, 0, 0], [1, 0, 0], [0, 0.5, 0], [0, 0, 1])
    assert np.isclose(d, 0.5) and jc.axis_kind(d, ang) == "perpendicular, offset"


def test_model_mobility_counts_loops():
    m, d = _load(ARM)
    assert jc.model_mobility(m, d) == 6                    # no equality constraints
    # assemble the four-bar numerically (a=1, b=2.5, c=2, d=2.2), then count with the simulator
    a, b, c, dd = 1.0, 2.5, 2.0, 2.2
    from rgkit import mechanisms as mech
    th2 = 0.7
    th3, th4 = mech.fourbar_closure(a, b, c, dd, th2)
    m, d = _load(FOURBAR)
    d.qpos[:] = [-th2, -(th3 - th2), -th4]
    mujoco.mj_forward(m, d)
    assert np.abs(d.efc_pos[:3]).max() < 1e-9               # the loop is closed
    assert jc.equality_rank(m, d) == 2                      # 3 connect equations, one redundant (planar motion)
    assert jc.model_mobility(m, d) == 1

import numpy as np
import pytest
from scipy.linalg import expm, logm

from rgkit import kinematics as kin
from rgkit import poe
from rgkit import transforms as tf

RNG = np.random.default_rng(11)


def random_table(n, kinds=None):
    kinds = kinds or ["R"] * n
    return [(RNG.uniform(-np.pi, np.pi), RNG.uniform(0.0, 0.6), RNG.uniform(-0.5, 0.5), RNG.uniform(-1, 1), k)
            for k in kinds]


def test_screw_axes():
    S = poe.screw_axis([0, 0, 2], [1, 0, 0])
    assert np.allclose(S, [0, 0, 1, 0, -1, 0])
    assert np.allclose(poe.screw_axis_prismatic([0, 3, 0]), [0, 0, 0, 0, 1, 0])


def test_exp_screw_matches_matrix_exponential():
    for _ in range(20):
        w = RNG.normal(size=3); w /= np.linalg.norm(w)
        S = np.concatenate([w, RNG.normal(size=3)])
        th = RNG.uniform(-3, 3)
        assert np.allclose(poe.exp_screw(S, th), expm(poe.twist_hat(S) * th))
    S = np.array([0, 0, 0, 0, 0, 1.0])
    assert np.allclose(poe.exp_screw(S, 0.7)[:3, 3], [0, 0, 0.7])
    with pytest.raises(ValueError):
        poe.exp_screw([0, 0, 2.0, 0, 0, 0], 0.1)


def test_log_inverts_exp_and_matches_scipy():
    for _ in range(20):
        w = RNG.normal(size=3); w /= np.linalg.norm(w)
        S = np.concatenate([w, RNG.normal(size=3)])
        th = RNG.uniform(0.05, 3.0)
        T = poe.exp_screw(S, th)
        tw = poe.log_se3(T)
        assert np.allclose(tw, S * th, atol=1e-9)
        assert np.allclose(poe.twist_hat(tw), np.real(logm(T)), atol=1e-8)
        S2, th2 = poe.screw_from_twist(tw)
        assert np.allclose(S2, S) and np.isclose(th2, th)
    assert np.allclose(poe.log_se3(tf.make_T(np.eye(3), [1, 2, 3])), [0, 0, 0, 1, 2, 3])


def test_adjoint_properties():
    T1 = tf.make_T(tf.rotx(0.4) @ tf.rotz(1.1), [0.1, 0.2, 0.3])
    T2 = tf.make_T(tf.roty(-0.7), [-0.3, 0.0, 0.5])
    assert np.allclose(poe.adjoint(T1 @ T2), poe.adjoint(T1) @ poe.adjoint(T2))
    assert np.allclose(poe.adjoint(tf.inv_T(T1)), np.linalg.inv(poe.adjoint(T1)))
    S = np.concatenate([[0, 0, 1.0], RNG.normal(size=3)])
    # T e^{[S]th} T^{-1} = e^{[Ad_T S] th}
    assert np.allclose(T1 @ poe.exp_screw(S, 0.8) @ tf.inv_T(T1), poe.exp_screw(poe.adjoint(T1) @ S, 0.8))


@pytest.mark.parametrize("convention", ["modified", "standard"])
@pytest.mark.parametrize("kinds", [["R"] * 6, ["R", "R", "P", "R"], ["P", "R", "P", "R", "R"]])
def test_poe_equals_dh_forward_kinematics(convention, kinds):
    table = random_table(len(kinds), kinds)
    table[0] = (0.0, 0.0) + tuple(table[0][2:]) if convention == "modified" else table[0]
    M, Slist = poe.screws_from_table(table, convention)
    Blist = poe.body_from_space(M, Slist)
    assert np.allclose(poe.space_from_body(M, Blist), Slist)
    for _ in range(20):
        q = np.array([RNG.uniform(-3, 3) if k == "R" else RNG.uniform(-0.3, 0.3) for k in kinds])
        T_dh = kin.fk_dh(table, q, convention)[-1]
        assert np.allclose(poe.fk_poe_space(M, Slist, q), T_dh, atol=1e-10)
        assert np.allclose(poe.fk_poe_body(M, Blist, q), T_dh, atol=1e-10)


def test_planar_2r_screws():
    L1, L2 = 1.0, 0.8
    M = tf.make_T(np.eye(3), [L1 + L2, 0, 0])
    S = np.array([poe.screw_axis([0, 0, 1], [0, 0, 0]), poe.screw_axis([0, 0, 1], [L1, 0, 0])])
    assert np.allclose(S[1], [0, 0, 1, 0, -L1, 0])
    t1, t2 = 0.5, -0.9
    T = poe.fk_poe_space(M, S, [t1, t2])
    assert np.allclose(T[:2, 3], [L1 * np.cos(t1) + L2 * np.cos(t1 + t2), L1 * np.sin(t1) + L2 * np.sin(t1 + t2)])


mujoco = pytest.importorskip("mujoco")


def test_screws_from_mujoco_small_arm():
    xml = """<mujoco><worldbody><body pos="0 0 0.2"><joint name="a" axis="0 0 1"/><geom size="0.05"/>
    <body pos="0.5 0 0"><joint name="b" axis="0 1 0"/><geom size="0.05"/><body pos="0.4 0 0">
    <joint name="c" type="slide" axis="1 0 0"/><geom size="0.05"/><site name="tip" pos="0.1 0 0"/></body></body></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    M, S = poe.screws_from_mujoco(model, data, site="tip")
    for _ in range(10):
        q = RNG.uniform(-1, 1, 3)
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        T_mj = tf.make_T(data.site("tip").xmat.reshape(3, 3), data.site("tip").xpos)
        assert np.allclose(poe.fk_poe_space(M, S, q), T_mj, atol=1e-12)

import numpy as np
from rgkit import kinematics as kin
from rgkit import transforms as tf

deg = np.deg2rad


def test_planar_2r_matches_closed_form():
    L1, L2 = 1.0, 0.8
    table = [(0, 0, 0, 0, "R"), (0, L1, 0, 0, "R"), (0, L2, 0, 0, "R")]   # last row = fixed tool offset
    t1, t2 = deg(30), deg(45)
    T = kin.fk_dh(table, [t1, t2, 0.0])[-1]
    assert np.allclose(T[:2, 3], [L1 * np.cos(t1) + L2 * np.cos(t1 + t2), L1 * np.sin(t1) + L2 * np.sin(t1 + t2)])
    assert np.allclose(T[:2, 3], kin.planar_fk([L1, L2], [t1, t2])[-1])
    assert np.allclose(T[:2, 3], [1.0731, 1.2727], atol=5e-5)            # value quoted in the notes


def test_modified_and_standard_agree_on_planar_chain():
    # for a planar arm the two conventions give the same tip pose when each is given its own table
    L = [1.0, 0.7, 0.4]
    q = [0.3, -0.8, 0.5]
    mod = [(0, 0, 0, 0, "R"), (0, L[0], 0, 0, "R"), (0, L[1], 0, 0, "R"), (0, L[2], 0, 0, "R")]
    std = [(0, L[0], 0, 0, "R"), (0, L[1], 0, 0, "R"), (0, L[2], 0, 0, "R")]
    Tm = kin.fk_dh(mod, q + [0.0], "modified")[-1]
    Ts = kin.fk_dh(std, q, "standard")[-1]
    assert np.allclose(Tm, Ts)


def test_results_are_rigid_transforms():
    table = [(0, 0, 0.3, 0, "R"), (-np.pi / 2, 0.1, 0, 0, "R"), (0, 0.4, 0.05, 0, "P")]
    for T in kin.fk_dh(table, [0.4, -0.9, 0.2]):
        assert tf.is_transform(T)


def test_prismatic_adds_to_d():
    T = kin.fk_dh([(0, 0, 0.5, 0, "P")], [0.25])[-1]
    assert np.allclose(T[:3, 3], [0, 0, 0.75])

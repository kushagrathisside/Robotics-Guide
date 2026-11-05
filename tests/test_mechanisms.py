import numpy as np
import pytest

from rgkit import mechanisms as mech

deg = np.deg2rad


def test_grubler_textbook_examples():
    # four-bar and slider-crank: 1 dof; serial n-R arm: n dof; Stewart platform: 6 dof
    assert mech.grubler_planar(4, [1, 1, 1, 1]) == 1
    assert mech.grubler_planar(4, [1, 1, 1, 1]) == mech.grubler(4, [1] * 4, m=3)
    assert mech.grubler_spatial(7, [1] * 6) == 6
    assert mech.grubler_spatial(14, [2] * 6 + [1] * 6 + [3] * 6) == 6        # 6 U, 6 P, 6 S
    # the spatial formula wrongly applied to a planar four-bar
    assert mech.grubler_spatial(4, [1] * 4) == -2
    # Delta robot: 11 links, 3 R + 12 S
    assert mech.grubler_spatial(11, [1] * 3 + [3] * 12) == 9


def test_lower_pairs_table():
    assert {k: v[1] for k, v in mech.LOWER_PAIRS.items()} == dict(
        revolute=1, prismatic=1, helical=1, cylindrical=2, spherical=3, planar=3)


def test_independent_loops():
    assert mech.independent_loops(4, 4) == 1            # four-bar
    assert mech.independent_loops(14, 18) == 5          # Stewart platform
    assert mech.independent_loops(7, 6) == 0            # serial 6R arm


def test_rigidity_counts_general_position():
    rng = np.random.default_rng(3)
    for n in range(3, 9):
        assert mech.rigidity_dof(rng.normal(size=(n, 3))) == (6 if n >= 4 else 6)
        assert mech.rigidity_dof(rng.normal(size=(n, 2))) == 3
    assert mech.rigidity_dof(rng.normal(size=(1, 3))) == 3
    assert mech.rigidity_dof(rng.normal(size=(2, 3))) == 5


def test_fourbar_closure_and_mobility():
    a, b, c, d = 1.0, 2.5, 2.0, 2.2
    for th in np.linspace(0.1, 6.0, 7):
        for br in (+1, -1):
            sol = mech.fourbar_closure(a, b, c, d, th, br)
            assert sol is not None
            pts = mech.fourbar_points(a, b, c, d, th, br)
            O2, A, B, O4 = pts
            assert np.isclose(np.linalg.norm(A - O2), a) and np.isclose(np.linalg.norm(B - A), b)
            assert np.isclose(np.linalg.norm(B - O4), c)
    L, q = mech.fourbar_linkage(a, b, c, d, 0.7)
    assert np.abs(L.residual(q)).max() < 1e-12
    assert L.mobility(q)[0] == 1 == L.grubler()
    # tracking the motion with the Newton solver agrees with the closed form
    q2 = L.solve(q, fixed={2: 0.9})
    assert np.isclose(q2[8], mech.fourbar_closure(a, b, c, d, 0.9)[1])


def test_fourbar_unassemblable():
    assert mech.fourbar_closure(1.0, 1.0, 1.0, 5.0, 0.0) is None


def test_slider_crank():
    r, l = 1.0, 3.0
    L, q = mech.slider_crank_linkage(r, l, 0.8)
    assert np.abs(L.residual(q)).max() < 1e-12
    assert L.mobility(q)[0] == 1 == L.grubler()
    q2 = L.solve(q, fixed={2: 1.4})
    assert np.isclose(q2[6], mech.slider_crank_x(r, l, 1.4))
    assert np.isclose(mech.slider_crank_x(r, l, 0.0), r + l)         # top dead centre
    assert np.isclose(mech.slider_crank_x(r, l, np.pi), l - r)       # bottom dead centre


def test_coupling_rod_is_paradoxical():
    L, q = mech.coupling_rod_linkage()
    assert np.abs(L.residual(q)).max() < 1e-12
    assert L.grubler() == 0
    assert L.mobility(q)[0] == 1


def test_parallelogram_bifurcation():
    L, q = mech.fourbar_linkage(1.0, 2.0, 1.0, 2.0, 0.7)
    assert L.mobility(q)[0] == 1
    L, q = mech.fourbar_linkage(1.0, 2.0, 1.0, 2.0, 0.0)             # all links collinear
    assert L.mobility(q)[0] == 2


def test_grashof():
    assert mech.grashof(1.0, 2.5, 2.0, 2.2)
    assert not mech.grashof(2.0, 2.5, 2.0, 1.0 + 3.0)
    assert "crank-rocker" in mech.grashof_type(1.0, 2.5, 2.0, 2.2)
    assert "double-rocker" in mech.grashof_type(3.0, 4.0, 3.5, 5.5)


def test_stewart_platform():
    base, plat = mech.stewart_geometry()
    pos, rv = np.array([0.0, 0.0, 1.0]), np.array([0.1, 0.05, 0.2])
    assert mech.stewart_mobility(base, plat, pos, rv) == 6
    assert [mech.stewart_mobility(base, plat, pos, rv, locked_legs=range(k)) for k in range(7)] == [6, 5, 4, 3, 2, 1, 0]
    # all legs equal at the symmetric pose
    l = mech.stewart_leg_lengths(base, plat, [0, 0, 1.0], [0, 0, 0])
    assert np.allclose(l, l[0])


def test_bennett_paradox():
    params = mech.bennett_parameters()
    a, b, al, be = params
    assert np.isclose(a / np.sin(al), b / np.sin(be))
    # known closed branch: theta3 = -theta1, theta4 = -theta2 with tan(t1/2) tan(t2/2) = sin((be+al)/2)/sin((be-al)/2)
    k = np.sin((be + al) / 2) / np.sin((be - al) / 2)
    t1 = 0.8
    t2 = 2 * np.arctan(k / np.tan(t1 / 2))
    th = np.array([t1, t2, -t1, -t2])
    assert np.abs(mech.bennett_residual(th, params)).max() < 1e-10
    dof, _ = mech.numerical_mobility(lambda t: mech.bennett_residual(t, params), th)
    assert dof == 1
    assert mech.grubler_spatial(4, [1] * 4) == -2

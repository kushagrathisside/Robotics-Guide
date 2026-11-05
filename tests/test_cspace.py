import numpy as np

from rgkit import cspace as cs
from rgkit import transforms as tf

deg = np.deg2rad


def test_wrap_and_circle_distance():
    assert np.isclose(cs.wrap_to_pi(3 * np.pi), np.pi)
    assert np.isclose(cs.wrap_to_pi(-np.pi / 2 - 2 * np.pi), -np.pi / 2)
    assert np.isclose(cs.angle_diff(deg(350), deg(10)), deg(20))
    assert np.isclose(cs.circle_distance(deg(10), deg(350)), deg(20))
    assert np.isclose(cs.torus_distance([deg(350), 0], [deg(10), deg(30)]), np.hypot(deg(20), deg(30)))
    mid = cs.torus_interpolate([deg(350), 0.0], [deg(10), 0.0], 0.5)
    assert np.isclose(mid[0], 0.0)


def test_so3_ball_identification():
    n = np.array([0.0, 0.6, 0.8])
    w1, w2 = np.pi * n, -np.pi * n
    R1, R2 = cs.rotation_from_ball(w1), cs.rotation_from_ball(w2)
    assert np.allclose(R1, R2)                            # antipodal boundary points are the same rotation
    w = cs.canonical_ball(1.3 * np.pi * n)                # a vector outside the ball is mapped back inside
    assert np.linalg.norm(w) <= np.pi + 1e-9
    assert np.allclose(cs.rotation_from_ball(w), cs.rotation_from_ball(1.3 * np.pi * n))


def test_manifold_dimension_of_so3():
    rng = np.random.default_rng(0)
    R = tf.rot_axis_angle(rng.normal(size=3), 0.7)
    assert cs.manifold_dimension(cs.so3_constraint, R.ravel()) == 3        # 9 numbers - 6 constraints
    assert cs.manifold_dimension(lambda x: np.array([x @ x - 1.0]), np.array([0.6, 0.0, 0.8])) == 2  # S^2


def test_unicycle_lie_bracket_and_chow():
    q = np.array([0.0, 0.0, 0.3])
    br = cs.lie_bracket(cs.unicycle_field_g1, cs.unicycle_field_g2, q)
    assert np.allclose(br, [np.sin(0.3), -np.cos(0.3), 0.0], atol=1e-7)
    assert cs.distribution_rank([cs.unicycle_field_g1, cs.unicycle_field_g2], q) == 3
    assert cs.distribution_rank([cs.unicycle_field_g1, cs.unicycle_field_g2], q, depth=1) == 2


def test_parking_manoeuvre_moves_sideways():
    eps = 0.1
    path = cs.parking_manoeuvre(eps)
    final = path[-1]
    assert np.isclose(final[2], 0.0)
    assert np.allclose(final[:2], [eps * (1 - np.cos(eps)), -eps * np.sin(eps)], atol=1e-12)
    assert np.isclose(final[1], -eps ** 2, rtol=2e-2)       # second-order prediction eps^2 [g1, g2]
    # exact integration of a circle: radius v/w
    q = cs.unicycle_step([0, 0, 0], 1.0, 2.0, np.pi)
    assert np.allclose(q, [0, 0, 2 * np.pi], atol=1e-12) or np.allclose(q[:2], [0, 0], atol=1e-12)

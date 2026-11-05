"""Property and worked-example tests for rgkit.transforms.

The numerical answers quoted in the Chapter 2 notes are asserted here, so a
mistake in either the notes' code or the library is caught automatically.
"""

import itertools

import numpy as np
import pytest

from rgkit import transforms as tf

rng = np.random.default_rng(0)
deg = np.deg2rad


def random_rot():
    q = rng.normal(size=4)
    return tf.quat_to_rot(q / np.linalg.norm(q))


@pytest.mark.parametrize("f", [tf.rotx, tf.roty, tf.rotz])
def test_elementary_rotations_are_in_SO3(f):
    for th in np.linspace(-np.pi, np.pi, 13):
        assert tf.is_rotation(f(th))


def test_skew_is_cross_product():
    w, p = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(tf.skew(w) @ p, np.cross(w, p))
    assert np.allclose(tf.unskew(tf.skew(w)), w)


def test_rodrigues_matches_elementary():
    for th in np.linspace(-3, 3, 7):
        assert np.allclose(tf.rot_axis_angle([1, 0, 0], th), tf.rotx(th))
        assert np.allclose(tf.rot_axis_angle([0, 1, 0], th), tf.roty(th))
        assert np.allclose(tf.rot_axis_angle([0, 0, 1], th), tf.rotz(th))


def test_axis_angle_round_trip_including_pi():
    for _ in range(50):
        k = rng.normal(size=3); k /= np.linalg.norm(k)
        th = rng.uniform(0.01, np.pi - 0.01)
        k2, th2 = tf.axis_angle_from_rot(tf.rot_axis_angle(k, th))
        assert np.isclose(th, th2) and np.allclose(k, k2)
    k2, th2 = tf.axis_angle_from_rot(tf.rot_axis_angle([0, 1, 1], np.pi))
    assert np.isclose(th2, np.pi)
    assert np.allclose(tf.rot_axis_angle(k2, th2), tf.rot_axis_angle([0, 1, 1], np.pi))


def test_inverse_transform():
    for _ in range(20):
        T = tf.make_T(random_rot(), rng.normal(size=3))
        assert np.allclose(tf.inv_T(T) @ T, np.eye(4))
        assert np.allclose(tf.inv_T(T), np.linalg.inv(T))


def test_rpy_round_trip_and_equivalence():
    for _ in range(50):
        g, b, a = rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5), rng.uniform(-np.pi, np.pi)
        R = tf.rpy_to_rot(g, b, a)
        assert np.allclose(R, tf.euler_zyx_to_rot(a, b, g))  # fixed XYZ == Euler ZYX
        assert np.allclose(R, tf.euler_to_rot([g, b, a], "xyz", "fixed"))
        assert np.allclose(R, tf.euler_to_rot([a, b, g], "zyx", "current"))
        assert np.allclose(tf.rot_to_rpy(R), (g, b, a))


def test_zyz_round_trip_and_singularity():
    for _ in range(50):
        a, b, g = rng.uniform(-np.pi, np.pi), rng.uniform(0.05, np.pi - 0.05), rng.uniform(-np.pi, np.pi)
        R = tf.euler_zyz_to_rot(a, b, g)
        assert np.allclose(tf.euler_zyz_to_rot(*tf.rot_to_euler_zyz(R)), R)
        assert np.allclose(tf.rot_to_euler_zyz(R), (a, b, g))
    for b in (0.0, np.pi):
        R = tf.euler_zyz_to_rot(0.3, b, 0.5)
        assert np.allclose(tf.euler_zyz_to_rot(*tf.rot_to_euler_zyz(R)), R)


def test_rpy_gimbal_lock_reconstructs():
    for b in (np.pi / 2, -np.pi / 2):
        R = tf.rpy_to_rot(0.4, b, 0.9)
        assert np.allclose(tf.rpy_to_rot(*tf.rot_to_rpy(R)), R)


def test_all_24_conventions_are_rotations():
    orders = ["".join(p) for p in itertools.product("xyz", repeat=3) if p[0] != p[1] and p[1] != p[2]]
    assert len(orders) == 12
    for o, frame in itertools.product(orders, ("fixed", "current")):
        assert tf.is_rotation(tf.euler_to_rot([0.3, -0.7, 1.1], o, frame))
    # fixed-axis sequence equals current-axis sequence in reverse order
    for o in orders:
        assert np.allclose(tf.euler_to_rot([0.3, -0.7, 1.1], o, "fixed"),
                           tf.euler_to_rot([1.1, -0.7, 0.3], o[::-1], "current"))


def test_quaternion_algebra():
    for _ in range(50):
        R1, R2 = random_rot(), random_rot()
        q1, q2 = tf.rot_to_quat(R1), tf.rot_to_quat(R2)
        assert np.allclose(tf.quat_to_rot(q1), R1)
        assert np.allclose(tf.quat_to_rot(tf.quat_mul(q1, q2)), R1 @ R2)
        p = rng.normal(size=3)
        assert np.allclose(tf.quat_rotate(q1, p), R1 @ p)
    # 180-degree rotations, where Craig's 1/(4 w) formula breaks down
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]):
        R = tf.rot_axis_angle(axis, np.pi)
        assert np.allclose(tf.quat_to_rot(tf.rot_to_quat(R)), R)


def test_slerp_endpoints_and_constant_speed():
    q0 = tf.quat_from_axis_angle([0, 0, 1], 0.0)
    q1 = tf.quat_from_axis_angle([0, 0, 1], deg(90))
    assert np.allclose(tf.slerp(q0, q1, 0), q0) and np.allclose(tf.slerp(q0, q1, 1), q1)
    _, th = tf.axis_angle_from_rot(tf.quat_to_rot(tf.slerp(q0, q1, 0.25)))
    assert np.isclose(th, deg(22.5))


def test_orthonormalize():
    R = random_rot() + 1e-3 * rng.normal(size=(3, 3))
    assert tf.is_rotation(tf.orthonormalize(R))


# --- Worked examples quoted in the Chapter 2 notes --------------------------
def test_craig_example_2_1():
    P = tf.rotz(deg(30)) @ [0.0, 2.0, 0.0]
    assert np.allclose(P, [-1.0, 1.7320508, 0.0])


def test_craig_example_2_2():
    T = tf.make_T(tf.rotz(deg(30)), [10.0, 5.0, 0.0])
    P = (T @ [3.0, 7.0, 0.0, 1.0])[:3]
    assert np.allclose(P, [9.098, 12.562, 0.0], atol=5e-4)


def test_order_matters_example():
    assert np.allclose(tf.rotz(deg(90)) @ tf.rotx(deg(90)), [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert np.allclose(tf.rotx(deg(90)) @ tf.rotz(deg(90)), [[0, -1, 0], [0, 0, -1], [1, 0, 0]])


def test_rz_rx_is_120_deg_about_111():
    k, th = tf.axis_angle_from_rot(tf.rotz(deg(90)) @ tf.rotx(deg(90)))
    assert np.isclose(th, deg(120)) and np.allclose(k, np.ones(3) / np.sqrt(3))
    assert np.allclose(tf.rot_to_quat(tf.rotz(deg(90)) @ tf.rotx(deg(90))), [0.5, 0.5, 0.5, 0.5])

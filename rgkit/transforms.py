"""Reference implementations of the spatial-description mathematics.

Every function here is derived step by step in the MFoR notebooks
(Chapter 2). The notebooks re-derive and re-implement the formulas in the
open so that students can read them; this module holds the same formulas
in one tested place so that later chapters can simply import them.

Conventions (J. J. Craig, *Introduction to Robotics*, 3rd ed., Ch. 2)
---------------------------------------------------------------------
* ``R_AB`` (written :math:`{}^{A}_{B}R` in the notes) is the orientation of
  frame {B} described in frame {A}; its columns are the unit axes of {B}
  expressed in {A}.
* ``T_AB`` (:math:`{}^{A}_{B}T`) is the 4x4 homogeneous transform of {B}
  relative to {A}.
* Angles are in **radians** unless a function name ends in ``_deg``.
* Quaternions are stored scalar-first, ``q = [w, x, y, z]`` (Craig calls
  these Euler parameters ``epsilon_4, epsilon_1..3``).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rotx", "roty", "rotz", "skew", "unskew", "is_rotation",
    "rot_axis_angle", "axis_angle_from_rot",
    "transl", "make_T", "rot_of", "pos_of", "inv_T", "is_transform",
    "rpy_to_rot", "rot_to_rpy", "euler_zyx_to_rot", "rot_to_euler_zyx",
    "euler_zyz_to_rot", "rot_to_euler_zyz", "euler_to_rot",
    "quat_mul", "quat_conj", "quat_from_axis_angle", "quat_to_rot",
    "rot_to_quat", "quat_rotate", "slerp", "orthonormalize",
]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Elementary rotations and the skew-symmetric matrix
# ---------------------------------------------------------------------------
def rotx(theta: float) -> np.ndarray:
    """Rotation by ``theta`` about the X axis (Craig eq. 2.77)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def roty(theta: float) -> np.ndarray:
    """Rotation by ``theta`` about the Y axis. Note the +sin at top right."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotz(theta: float) -> np.ndarray:
    """Rotation by ``theta`` about the Z axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(w) -> np.ndarray:
    """Return the skew-symmetric matrix ``[w]x`` such that ``[w]x @ p = w x p``."""
    wx, wy, wz = np.asarray(w, dtype=float).ravel()
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])


def unskew(S) -> np.ndarray:
    """Inverse of :func:`skew`: recover ``w`` from ``[w]x``."""
    S = np.asarray(S, dtype=float)
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def is_rotation(R, tol: float = 1e-9) -> bool:
    """True if ``R`` is a member of SO(3): ``R^T R = I`` and ``det R = +1``."""
    R = np.asarray(R, dtype=float)
    return (R.shape == (3, 3)
            and np.allclose(R.T @ R, np.eye(3), atol=tol)
            and abs(np.linalg.det(R) - 1.0) < tol)


def orthonormalize(R) -> np.ndarray:
    """Project a nearly-orthonormal matrix back onto SO(3) using the SVD."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=float))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


# ---------------------------------------------------------------------------
# Angle-axis (Rodrigues) representation
# ---------------------------------------------------------------------------
def rot_axis_angle(axis, theta: float) -> np.ndarray:
    """Rodrigues' formula: rotation by ``theta`` about unit vector ``axis``.

    R = I + sin(theta) [k]x + (1 - cos(theta)) [k]x^2   (Craig eq. 2.80)
    """
    k = np.asarray(axis, dtype=float).ravel()
    k = k / np.linalg.norm(k)
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def axis_angle_from_rot(R):
    """Return ``(axis, theta)`` with ``theta`` in [0, pi] (Craig eq. 2.81).

    At ``theta = 0`` the axis is undefined and ``[0, 0, 1]`` is returned.
    Near ``theta = pi`` the formula with ``sin(theta)`` in the denominator
    is ill-conditioned, so the axis is recovered from ``(R + I) / 2``.
    """
    R = np.asarray(R, dtype=float)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    # atan2 of (sin, cos) is accurate everywhere; arccos alone loses ~8 digits near pi.
    theta = float(np.arctan2(0.5 * np.linalg.norm(v), (np.trace(R) - 1.0) / 2.0))
    if theta < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if np.pi - theta < 1e-6:
        B = (R + np.eye(3)) / 2.0          # = k k^T when theta = pi
        i = int(np.argmax(np.diag(B)))
        k = B[:, i] / np.sqrt(B[i, i])
        return k / np.linalg.norm(k), theta
    return v / (2.0 * np.sin(theta)), theta


# ---------------------------------------------------------------------------
# Homogeneous transforms
# ---------------------------------------------------------------------------
def make_T(R=None, p=None) -> np.ndarray:
    """Assemble the 4x4 homogeneous transform [[R, p], [0 0 0 1]]."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float)
    if p is not None:
        T[:3, 3] = np.asarray(p, dtype=float).ravel()
    return T


def transl(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """Pure translation as a 4x4 homogeneous transform."""
    return make_T(p=[x, y, z])


def rot_of(T) -> np.ndarray:
    """Rotation block of a homogeneous transform."""
    return np.asarray(T)[:3, :3]


def pos_of(T) -> np.ndarray:
    """Translation column of a homogeneous transform."""
    return np.asarray(T)[:3, 3]


def inv_T(T) -> np.ndarray:
    """Inverse of a homogeneous transform using its structure (Craig eq. 2.45).

    inv([[R, p], [0, 1]]) = [[R^T, -R^T p], [0, 1]]
    """
    R, p = rot_of(T), pos_of(T)
    return make_T(R.T, -R.T @ p)


def is_transform(T, tol: float = 1e-9) -> bool:
    """True if ``T`` is a valid element of SE(3)."""
    T = np.asarray(T, dtype=float)
    return (T.shape == (4, 4) and is_rotation(T[:3, :3], tol)
            and np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=tol))


# ---------------------------------------------------------------------------
# Three-angle representations (Craig Sec. 2.8 and Appendix B)
# ---------------------------------------------------------------------------
_ELEM = {"x": rotx, "y": roty, "z": rotz}


def euler_to_rot(angles, order: str = "zyx", frame: str = "current") -> np.ndarray:
    """General three-angle rotation.

    ``order`` names the axes in the order the rotations are *applied*.
    ``frame='current'`` (Euler angles) post-multiplies each new rotation;
    ``frame='fixed'`` pre-multiplies it. All 24 conventions are covered.
    """
    order = order.lower()
    if len(order) != 3 or any(a not in _ELEM for a in order):
        raise ValueError("order must be three letters from 'xyz', e.g. 'zyz'")
    R = np.eye(3)
    for axis, ang in zip(order, angles):
        step = _ELEM[axis](ang)
        R = R @ step if frame == "current" else step @ R
    return R


def rpy_to_rot(gamma: float, beta: float, alpha: float) -> np.ndarray:
    """X-Y-Z *fixed* angles (roll gamma, pitch beta, yaw alpha), Craig eq. 2.64.

    R = Rz(alpha) Ry(beta) Rx(gamma)
    """
    return rotz(alpha) @ roty(beta) @ rotx(gamma)


def rot_to_rpy(R):
    """Inverse of :func:`rpy_to_rot`; returns ``(gamma, beta, alpha)``.

    Uses Craig eq. 2.66. At the singularity ``beta = +/-pi/2`` the
    convention ``alpha = 0`` is adopted (gimbal lock).
    """
    R = np.asarray(R, dtype=float)
    beta = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    cb = np.cos(beta)
    if abs(cb) < 1e-9:
        alpha = 0.0
        gamma = np.sign(np.sin(beta)) * np.arctan2(R[0, 1], R[1, 1])
    else:
        alpha = np.arctan2(R[1, 0] / cb, R[0, 0] / cb)
        gamma = np.arctan2(R[2, 1] / cb, R[2, 2] / cb)
    return float(gamma), float(beta), float(alpha)


def euler_zyx_to_rot(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Z-Y-X Euler angles: Rz(alpha) Ry(beta) Rx(gamma) — same matrix as RPY."""
    return rotz(alpha) @ roty(beta) @ rotx(gamma)


def rot_to_euler_zyx(R):
    """Inverse of :func:`euler_zyx_to_rot`; returns ``(alpha, beta, gamma)``."""
    gamma, beta, alpha = rot_to_rpy(R)
    return alpha, beta, gamma


def euler_zyz_to_rot(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Z-Y-Z Euler angles: Rz(alpha) Ry(beta) Rz(gamma), Craig eq. 2.72."""
    return rotz(alpha) @ roty(beta) @ rotz(gamma)


def rot_to_euler_zyz(R):
    """Inverse of :func:`euler_zyz_to_rot` with ``beta`` in [0, pi] (Craig eq. 2.74)."""
    R = np.asarray(R, dtype=float)
    beta = np.arctan2(np.hypot(R[2, 0], R[2, 1]), R[2, 2])
    sb = np.sin(beta)
    if abs(sb) < 1e-9:
        alpha = 0.0
        if R[2, 2] > 0:      # beta = 0: only alpha + gamma is defined
            gamma = np.arctan2(-R[0, 1], R[0, 0])
        else:                # beta = pi: only alpha - gamma is defined
            gamma = np.arctan2(R[0, 1], -R[0, 0])
    else:
        alpha = np.arctan2(R[1, 2] / sb, R[0, 2] / sb)
        gamma = np.arctan2(R[2, 1] / sb, -R[2, 0] / sb)
    return float(alpha), float(beta), float(gamma)


# ---------------------------------------------------------------------------
# Unit quaternions (Euler parameters), scalar-first [w, x, y, z]
# ---------------------------------------------------------------------------
def quat_mul(q1, q2) -> np.ndarray:
    """Hamilton product q1 (x) q2 = (s1 s2 - v1.v2,  s1 v2 + s2 v1 + v1 x v2)."""
    s1, v1 = q1[0], np.asarray(q1[1:], dtype=float)
    s2, v2 = q2[0], np.asarray(q2[1:], dtype=float)
    return np.concatenate([[s1 * s2 - v1 @ v2], s1 * v2 + s2 * v1 + np.cross(v1, v2)])


def quat_conj(q) -> np.ndarray:
    """Conjugate (s, -v); equals the inverse for a unit quaternion."""
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_axis_angle(axis, theta: float) -> np.ndarray:
    """q = (cos(theta/2), k sin(theta/2))."""
    k = np.asarray(axis, dtype=float).ravel()
    k = k / np.linalg.norm(k)
    return np.concatenate([[np.cos(theta / 2.0)], k * np.sin(theta / 2.0)])


def quat_to_rot(q) -> np.ndarray:
    """Rotation matrix of a unit quaternion (Craig eq. 2.91)."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R) -> np.ndarray:
    """Unit quaternion of a rotation matrix (Shepperd's method, w >= 0).

    Craig eq. 2.92 divides by ``4 w``, which fails near 180 degrees; this
    version divides by the largest of the four candidates instead.
    """
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    cands = [tr, R[0, 0], R[1, 1], R[2, 2]]
    i = int(np.argmax(cands))
    if i == 0:
        s = 2.0 * np.sqrt(1.0 + tr)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif i == 1:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif i == 2:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_rotate(q, p) -> np.ndarray:
    """Rotate vector ``p`` by unit quaternion ``q``: (0, p') = q (x) (0, p) (x) q*."""
    return quat_mul(quat_mul(q, np.concatenate([[0.0], p])), quat_conj(q))[1:]


def slerp(q0, q1, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions (shortest path)."""
    q0 = np.asarray(q0, dtype=float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
    d = float(q0 @ q1)
    if d < 0.0:              # q and -q are the same rotation; take the short arc
        q1, d = -q1, -d
    if d > 1.0 - 1e-9:       # nearly identical: linear interpolation is exact enough
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    omega = np.arccos(d)
    return (np.sin((1 - t) * omega) * q0 + np.sin(t * omega) * q1) / np.sin(omega)

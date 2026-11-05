"""Product of exponentials: screw axes, the exponential of a twist and PoE forward kinematics.

Implements the formulation of Lynch & Park (*Modern Robotics*, Ch. 3-4) used in notebook 3.4. A serial robot
is described by

* the home pose ``M`` of the end-effector (all joint variables zero), and
* one **screw axis** per joint, ``S_i = (omega_i, v_i)``, written either in the fixed *space* frame (``Slist``)
  or in the end-effector (*body*) frame (``Blist``), at the home configuration.

For a revolute joint with unit axis ``omega`` through the point ``q``: ``S = (omega, -omega x q)``; for a prismatic
joint with unit direction ``v``: ``S = (0, v)``. Then

    space form:  T(theta) = e^{[S_1] theta_1} ... e^{[S_n] theta_n} M
    body  form:  T(theta) = M e^{[B_1] theta_1} ... e^{[B_n] theta_n}

No link frames, common normals or joint offsets are needed. Twists are ordered ``(angular, linear)``.
"""

from __future__ import annotations

import numpy as np

from .transforms import skew, unskew, make_T, inv_T, axis_angle_from_rot
from .kinematics import fk_dh

__all__ = ["adjoint", "screw_axis", "screw_axis_prismatic", "twist_hat", "twist_vee", "exp_screw", "log_se3",
           "screw_from_twist", "fk_poe_space", "fk_poe_body", "body_from_space", "space_from_body",
           "screws_from_axes", "screws_from_table", "screws_from_mujoco"]


def adjoint(T) -> np.ndarray:
    """6x6 adjoint ``[[R, 0], [[p]_x R, R]]`` mapping a twist expressed in one frame to another: ``V_a = Ad_{T_ab} V_b``."""
    T = np.asarray(T, dtype=float)
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


def screw_axis(omega, point) -> np.ndarray:
    """Screw axis ``(omega, -omega x q)`` of a revolute joint with unit axis ``omega`` through ``point``."""
    w = np.asarray(omega, dtype=float).ravel()
    w = w / np.linalg.norm(w)
    q = np.asarray(point, dtype=float).ravel()
    return np.concatenate([w, -np.cross(w, q)])


def screw_axis_prismatic(direction) -> np.ndarray:
    """Screw axis ``(0, v)`` of a prismatic joint sliding along the unit direction ``v``."""
    v = np.asarray(direction, dtype=float).ravel()
    return np.concatenate([np.zeros(3), v / np.linalg.norm(v)])


def twist_hat(S) -> np.ndarray:
    """The 4x4 matrix ``[S] = [[ [omega]_x, v ], [0, 0]]`` of a twist ``S = (omega, v)``."""
    S = np.asarray(S, dtype=float).ravel()
    M = np.zeros((4, 4))
    M[:3, :3] = skew(S[:3])
    M[:3, 3] = S[3:]
    return M


def twist_vee(M) -> np.ndarray:
    """Inverse of :func:`twist_hat`."""
    M = np.asarray(M, dtype=float)
    return np.concatenate([unskew(M[:3, :3]), M[:3, 3]])


def exp_screw(S, theta: float) -> np.ndarray:
    """Matrix exponential ``e^{[S] theta}`` in closed form (Rodrigues' formula), as a 4x4 transform.

    ``S = (omega, v)`` must have ``|omega| = 1`` (a rotation, any ``v``) or ``omega = 0`` and ``|v| = 1`` (a pure
    translation); then ``theta`` is the angle or the distance. For ``|omega| = 1``::

        R = I + sin(theta) [omega] + (1 - cos(theta)) [omega]^2
        p = (I theta + (1 - cos(theta)) [omega] + (theta - sin(theta)) [omega]^2) v
    """
    S = np.asarray(S, dtype=float).ravel()
    w, v = S[:3], S[3:]
    nw = np.linalg.norm(w)
    if nw < 1e-12:
        return make_T(np.eye(3), v * theta)
    if abs(nw - 1.0) > 1e-9:
        raise ValueError("a screw axis needs |omega| = 1 (or omega = 0)")
    K = skew(w)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    p = (np.eye(3) * theta + (1 - np.cos(theta)) * K + (theta - np.sin(theta)) * (K @ K)) @ v
    return make_T(R, p)


def log_se3(T) -> np.ndarray:
    """Exponential coordinates (the twist ``S theta`` as a 6-vector) of a transform: the inverse of the exponential.

    Uses ``omega theta`` from the matrix logarithm of ``R`` and ``v theta = G(theta)^{-1} p`` with
    ``G^{-1} = I/theta - [omega]/2 + (1/theta - cot(theta/2)/2) [omega]^2``; for ``R = I`` the result is ``(0, p)``.
    """
    T = np.asarray(T, dtype=float)
    R, p = T[:3, :3], T[:3, 3]
    k, th = axis_angle_from_rot(R)
    if th < 1e-9:
        return np.concatenate([np.zeros(3), p])
    K = skew(k)
    Ginv = np.eye(3) / th - K / 2.0 + (1.0 / th - 0.5 / np.tan(th / 2.0)) * (K @ K)
    return np.concatenate([k * th, Ginv @ p * th])


def screw_from_twist(twist):
    """Split exponential coordinates ``S theta`` into the unit screw axis ``S`` and the distance ``theta``."""
    twist = np.asarray(twist, dtype=float).ravel()
    n = np.linalg.norm(twist[:3])
    if n < 1e-12:
        n = np.linalg.norm(twist[3:])
    return twist / n, float(n)


def fk_poe_space(M, Slist, thetas) -> np.ndarray:
    """``e^{[S_1] th_1} ... e^{[S_n] th_n} M`` for screw axes given as the rows of ``Slist`` ``(n, 6)``."""
    T = np.eye(4)
    for S, th in zip(np.asarray(Slist, dtype=float), np.asarray(thetas, dtype=float)):
        T = T @ exp_screw(S, th)
    return T @ np.asarray(M, dtype=float)


def fk_poe_body(M, Blist, thetas) -> np.ndarray:
    """``M e^{[B_1] th_1} ... e^{[B_n] th_n}`` for body-frame screw axes (rows of ``Blist``)."""
    T = np.asarray(M, dtype=float).copy()
    for B, th in zip(np.asarray(Blist, dtype=float), np.asarray(thetas, dtype=float)):
        T = T @ exp_screw(B, th)
    return T


def body_from_space(M, Slist) -> np.ndarray:
    """Body screw axes ``B_i = Ad_{M^{-1}} S_i`` (rows) from space screw axes."""
    Ad = adjoint(inv_T(M))
    return np.array([Ad @ S for S in np.asarray(Slist, dtype=float)])


def space_from_body(M, Blist) -> np.ndarray:
    """Space screw axes ``S_i = Ad_M B_i`` (rows) from body screw axes."""
    Ad = adjoint(M)
    return np.array([Ad @ B for B in np.asarray(Blist, dtype=float)])


def screws_from_axes(points, directions, kinds=None) -> np.ndarray:
    """Space screw axes (rows) from joint axes given as lines ``(point, direction)`` at the home configuration."""
    n = len(points)
    kinds = ["R"] * n if kinds is None else kinds
    return np.array([screw_axis(d, p) if k == "R" else screw_axis_prismatic(d)
                     for p, d, k in zip(points, directions, kinds)])


def screws_from_table(table, convention: str = "modified"):
    """``(M, Slist)`` of the robot described by a D-H table, expressed in the base frame ``{0}``.

    The axes are read from the frames at ``q = 0``: in the modified convention joint ``i`` turns about ``Z_i``
    through ``O_i``; in the standard convention about ``Z_{i-1}`` through ``O_{i-1}``. ``M`` is the pose of the
    last frame ``{N}`` at ``q = 0``.
    """
    Ts = fk_dh(table, None, convention)
    pts, dirs, kinds = [], [], []
    for i, row in enumerate(table, start=1):
        T = Ts[i] if convention == "modified" else Ts[i - 1]
        pts.append(T[:3, 3]); dirs.append(T[:3, 2]); kinds.append(row[4])
    return Ts[-1], screws_from_axes(pts, dirs, kinds)


def screws_from_mujoco(model, data, site=None, body=None, joint_names=None):
    """``(M, Slist)`` of a MuJoCo serial chain in the world frame, from the home configuration ``qpos = 0``.

    ``site`` (or ``body``) names the end-effector frame. ``data`` is restored afterwards.
    """
    import mujoco
    from .dh_tools import joint_axes_from_mujoco

    saved = data.qpos.copy()
    data.qpos[:] = 0.0
    mujoco.mj_kinematics(model, data)
    pts, dirs, kinds = joint_axes_from_mujoco(model, data, joint_names)
    if site is not None:
        s = data.site(site)
        M = make_T(s.xmat.reshape(3, 3).copy(), s.xpos.copy())
    else:
        b = data.body(body)
        M = make_T(b.xmat.reshape(3, 3).copy(), b.xpos.copy())
    data.qpos[:] = saved
    mujoco.mj_kinematics(model, data)
    return M, screws_from_axes(pts, dirs, kinds)

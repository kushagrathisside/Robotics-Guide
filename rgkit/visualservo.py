"""Camera geometry and visual servoing (Chapter 8 of the Robot Motion Control notebooks).

Everything here is plain NumPy so that each formula can be tested against a derivation done
independently (finite differences, SymPy, SciPy's matrix exponential).

Frames and conventions
----------------------
* The **camera frame** {C} follows the computer-vision convention used by Chaumette & Hutchinson and by
  Corke: ``x`` to the right, ``y`` down, ``z`` along the optical axis (forward). A MuJoCo camera looks
  along its own ``-z`` with ``y`` up; :func:`mujoco_to_cv_pose` converts (a rotation by 180 degrees about
  ``x``, i.e. multiplication by ``diag(1, -1, -1)``).
* ``T_wc`` is the pose of {C} in the world {W} (Craig's :math:`{}^{W}_{C}T`). A point in the camera frame is
  ``P_c = inv(T_wc) @ P_w``.
* Pixels use continuous image coordinates: pixel ``(row i, col j)`` covers ``[j, j+1) x [i, i+1)`` and its
  centre is ``(j + 0.5, i + 0.5)``; the principal point of a MuJoCo camera is ``(W/2, H/2)``.
* A **camera twist** is ``[v, omega]`` expressed in {C}: the velocity of the point of the camera that is at
  the origin of {C}, and the angular velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .transforms import axis_angle_from_rot, inv_T, make_T, rot_axis_angle, skew

__all__ = [
    "PinholeCamera", "MUJOCO_TO_CV", "mujoco_to_cv_pose", "project_world", "interaction_matrix",
    "interaction_matrix_px", "stack_interaction", "exp_se3", "rotation_vector", "ibvs_velocity",
    "pbvs_velocity", "planar_pose", "dls_solve", "twist_to_world", "simulate_ibvs", "simulate_pbvs",
    "simulate_eye_to_hand", "look_at",
]

#: Rotation from a MuJoCo camera frame to the computer-vision camera frame (columns of the former, flipped).
MUJOCO_TO_CV = np.diag([1.0, -1.0, -1.0])


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PinholeCamera:
    """Ideal pinhole camera with focal lengths ``fx, fy`` and principal point ``(cx, cy)`` in pixels."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fovy(cls, fovy_deg: float, width: int, height: int):
        """Camera with the vertical field of view ``fovy_deg`` (MuJoCo's ``fovy``) and square pixels."""
        f = 0.5 * height / np.tan(0.5 * np.deg2rad(fovy_deg))
        return cls(width, height, f, f, 0.5 * width, 0.5 * height)

    @property
    def K(self) -> np.ndarray:
        """Intrinsic matrix."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def project(self, P_c):
        """Pixel coordinates ``(u, v)`` of points ``P_c`` (N x 3, camera frame, ``z > 0``)."""
        P_c = np.atleast_2d(np.asarray(P_c, dtype=float))
        return np.column_stack([self.fx * P_c[:, 0] / P_c[:, 2] + self.cx,
                                self.fy * P_c[:, 1] / P_c[:, 2] + self.cy])

    def normalise(self, uv):
        """Normalised image coordinates ``(x, y) = ((u - cx)/fx, (v - cy)/fy)``."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        return np.column_stack([(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy])

    def denormalise(self, xy):
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        return np.column_stack([self.fx * xy[:, 0] + self.cx, self.fy * xy[:, 1] + self.cy])

    def backproject(self, uv, Z):
        """3-D points (camera frame) at depth ``Z`` that project to the pixels ``uv``."""
        xy = self.normalise(uv)
        Z = np.broadcast_to(np.asarray(Z, dtype=float), (len(xy),))
        return np.column_stack([xy[:, 0] * Z, xy[:, 1] * Z, Z])

    def in_view(self, uv, margin: float = 0.0):
        uv = np.atleast_2d(uv)
        return bool(np.all((uv[:, 0] >= margin) & (uv[:, 0] <= self.width - margin)
                           & (uv[:, 1] >= margin) & (uv[:, 1] <= self.height - margin)))


def mujoco_to_cv_pose(pos, xmat) -> np.ndarray:
    """Pose ``T_wc`` (computer-vision convention) of a MuJoCo camera or site from ``cam_xpos`` / ``cam_xmat``."""
    return make_T(np.asarray(xmat, dtype=float).reshape(3, 3) @ MUJOCO_TO_CV, pos)


def look_at(eye, target, up=(0.0, 0.0, 1.0), roll: float = 0.0) -> np.ndarray:
    """Pose ``T_wc`` of a camera at ``eye`` whose optical axis points at ``target`` (``y`` down, image 'up' = ``up``)."""
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=float))          # right = forward x up   (y down, x right, z forward)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, [1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z]) @ rot_axis_angle([0, 0, 1], roll)
    return make_T(R, eye)


def project_world(cam: PinholeCamera, T_wc, P_w):
    """Pixels of world points ``P_w`` (N x 3) seen from a camera at pose ``T_wc``; also returns the depths."""
    P_w = np.atleast_2d(np.asarray(P_w, dtype=float))
    T_cw = inv_T(T_wc)
    P_c = P_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    return cam.project(P_c), P_c[:, 2]


# ---------------------------------------------------------------------------
# Interaction matrix
# ---------------------------------------------------------------------------
def interaction_matrix(xy, Z):
    """Interaction matrix (2N x 6) of N image points in *normalised* coordinates with depths ``Z``.

    ``[x_dot; y_dot] = L @ [vx vy vz wx wy wz]`` with the twist in the camera frame:

    ``L = [[-1/Z, 0, x/Z, x y, -(1+x^2), y], [0, -1/Z, y/Z, 1+y^2, -x y, -x]]``.
    """
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    Z = np.broadcast_to(np.asarray(Z, dtype=float), (len(xy),))
    L = np.zeros((2 * len(xy), 6))
    for i, ((x, y), z) in enumerate(zip(xy, Z)):
        L[2 * i] = [-1 / z, 0.0, x / z, x * y, -(1 + x * x), y]
        L[2 * i + 1] = [0.0, -1 / z, y / z, 1 + y * y, -x * y, -x]
    return L


def interaction_matrix_px(uv, Z, cam: PinholeCamera):
    """Interaction matrix of points given in *pixels*: ``[u_dot; v_dot] = diag(fx, fy) [x_dot; y_dot]``."""
    L = interaction_matrix(cam.normalise(uv), Z)
    L[0::2] *= cam.fx
    L[1::2] *= cam.fy
    return L


def stack_interaction(uv, Z, cam: PinholeCamera):
    """Alias of :func:`interaction_matrix_px` (kept for notebook readability)."""
    return interaction_matrix_px(uv, Z, cam)


# ---------------------------------------------------------------------------
# Rigid-body motion helpers
# ---------------------------------------------------------------------------
def exp_se3(xi) -> np.ndarray:
    """Homogeneous transform ``exp([xi])`` of the twist ``xi = [v, w]`` (already multiplied by the time step)."""
    xi = np.asarray(xi, dtype=float)
    v, w = xi[:3], xi[3:]
    th = np.linalg.norm(w)
    W = skew(w)
    if th < 1e-9:
        R, V = np.eye(3) + W, np.eye(3) + 0.5 * W
    else:
        R = np.eye(3) + np.sin(th) / th * W + (1 - np.cos(th)) / th**2 * (W @ W)
        V = np.eye(3) + (1 - np.cos(th)) / th**2 * W + (th - np.sin(th)) / th**3 * (W @ W)
    return make_T(R, V @ v)


def rotation_vector(R) -> np.ndarray:
    """Rotation vector ``theta * u`` of a rotation matrix (angle in [0, pi])."""
    u, th = axis_angle_from_rot(R)
    return th * u


def dls_solve(J, v, mu: float = 0.05):
    """Damped least-squares solution of ``J qdot = v``: ``qdot = J^T (J J^T + mu^2 I)^{-1} v``."""
    J = np.asarray(J, dtype=float)
    return J.T @ np.linalg.solve(J @ J.T + mu**2 * np.eye(J.shape[0]), np.asarray(v, dtype=float))


def twist_to_world(R_wc, twist) -> np.ndarray:
    """Express a camera-frame twist ``[v, w]`` in the world frame (rotate both halves by ``R_wc``)."""
    twist = np.asarray(twist, dtype=float)
    return np.concatenate([R_wc @ twist[:3], R_wc @ twist[3:]])


# ---------------------------------------------------------------------------
# Control laws
# ---------------------------------------------------------------------------
def ibvs_velocity(s, s_star, Z, Z_star, cam: PinholeCamera, lam: float = 0.5, L_at: str = "current",
                  rcond: float = 1e-9):
    """Image-based law ``v = -lam * pinv(L_hat) @ (s - s*)``; returns ``(twist, e)``.

    ``s`` and ``s_star`` are N x 2 pixel arrays; ``Z`` and ``Z_star`` are the depths (current/desired) used
    to build the interaction matrix; ``L_at`` selects ``L`` evaluated at the current features
    (``"current"``), at the desired features (``"desired"``) or their average (``"mean"``).
    """
    e = (np.asarray(s, dtype=float) - np.asarray(s_star, dtype=float)).ravel()
    if L_at == "current":
        L = interaction_matrix_px(s, Z, cam)
    elif L_at == "desired":
        L = interaction_matrix_px(s_star, Z_star, cam)
    elif L_at == "mean":
        L = 0.5 * (interaction_matrix_px(s, Z, cam) + interaction_matrix_px(s_star, Z_star, cam))
    else:
        raise ValueError("L_at must be 'current', 'desired' or 'mean'")
    return -lam * np.linalg.pinv(L, rcond=rcond) @ e, e


def pbvs_velocity(T_wc, T_wc_star, lam: float = 0.5):
    """Position-based law with the feature ``s = (c*_t_c, theta u)``.

    ``c*_T_c = inv(T_wc*) T_wc`` is the current camera in the desired camera frame. The law
    ``v = -lam R^T t``, ``w = -lam theta u`` makes ``t`` and ``theta u`` decay exponentially, so the camera
    travels on a straight line. Returns ``(twist, t, theta_u)``.
    """
    Tcc = inv_T(T_wc_star) @ np.asarray(T_wc, dtype=float)
    R, t = Tcc[:3, :3], Tcc[:3, 3]
    tu = rotation_vector(R)
    return np.concatenate([-lam * R.T @ t, -lam * tu]), t, tu


# ---------------------------------------------------------------------------
# Pose from a planar target
# ---------------------------------------------------------------------------
def planar_pose(P_obj, uv, cam: PinholeCamera, refine: bool = True, T0=None):
    """Pose ``T_co`` of a planar target (object frame with all points at ``z = 0``) from its pixels.

    A homography is estimated by the direct linear transform and decomposed into ``[r1 r2 t]``
    (Zhang's method); the result is optionally refined by minimising the reprojection error with
    Levenberg-Marquardt. At least 4 points are required; returns a 4 x 4 transform of the object in the camera.
    """
    P = np.asarray(P_obj, dtype=float)
    xy = cam.normalise(uv)
    if T0 is None:
        n = len(P)
        A = np.zeros((2 * n, 9))
        for i in range(n):
            X, Y = P[i, 0], P[i, 1]
            x, y = xy[i]
            A[2 * i] = [X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x]
            A[2 * i + 1] = [0, 0, 0, X, Y, 1, -y * X, -y * Y, -y]
        H = np.linalg.svd(A)[2][-1].reshape(3, 3)
        if (H @ np.array([P[:, 0].mean(), P[:, 1].mean(), 1.0]))[2] < 0:       # points must be in front of the camera
            H = -H
        lam_ = 0.5 * (np.linalg.norm(H[:, 0]) + np.linalg.norm(H[:, 1]))
        r1, r2, t = H[:, 0] / lam_, H[:, 1] / lam_, H[:, 2] / lam_
        U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
        R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        T0 = make_T(R, t)
    if not refine:
        return T0

    def resid(p):
        R = rot_axis_angle(p[:3], np.linalg.norm(p[:3])) if np.linalg.norm(p[:3]) > 1e-12 else np.eye(3)
        Pc = P @ R.T + p[3:]
        return (cam.project(Pc) - np.asarray(uv)).ravel()

    p0 = np.concatenate([rotation_vector(T0[:3, :3]), T0[:3, 3]])
    sol = least_squares(resid, p0, method="lm", xtol=1e-12, ftol=1e-12)
    p = sol.x
    th = np.linalg.norm(p[:3])
    return make_T(rot_axis_angle(p[:3], th) if th > 1e-12 else np.eye(3), p[3:])


# ---------------------------------------------------------------------------
# Kinematic simulation of free-flying cameras
# ---------------------------------------------------------------------------
def simulate_ibvs(cam: PinholeCamera, P_w, T_wc0, T_wc_star, lam: float = 0.5, dt: float = 0.05, n_steps: int = 200,
                  L_at: str = "current", depth_factor: float = 1.0, depth_const: float | None = None,
                  cam_ctrl: PinholeCamera | None = None, noise_px: float = 0.0, rng=None, v_max: float | None = None):
    """Eye-in-hand IBVS of a free-flying camera observing the world points ``P_w`` (N x 3).

    The control law sees pixels from the *true* camera ``cam`` (plus Gaussian noise ``noise_px``) but builds its
    interaction matrix with ``cam_ctrl`` (default: the same) and with depths equal to ``depth_factor`` times the
    true depths, or the constant ``depth_const``. The twist is applied for ``dt`` through the matrix exponential.
    Returns a dict with ``t``, ``T`` (poses, n+1 x 4 x 4), ``s`` (n+1 x N x 2 pixels), ``Z`` (n+1 x N depths),
    ``e`` (error norms) and ``v`` (twists).
    """
    cam_ctrl = cam_ctrl or cam
    rng = rng or np.random.default_rng(0)
    s_star, Z_star = project_world(cam, T_wc_star, P_w)
    T = np.array(T_wc0, dtype=float)
    Ts, ss, Zs, es, vs = [T.copy()], [], [], [], []
    s, Z = project_world(cam, T, P_w)
    ss.append(s); Zs.append(Z)
    for _ in range(n_steps):
        s_meas = s + noise_px * rng.standard_normal(s.shape)
        Zh = np.full_like(Z, depth_const) if depth_const is not None else depth_factor * Z
        Zh_star = np.full_like(Z_star, depth_const) if depth_const is not None else depth_factor * Z_star
        v, e = ibvs_velocity(s_meas, s_star, Zh, Zh_star, cam_ctrl, lam, L_at)
        if v_max is not None and np.linalg.norm(v[:3]) > v_max:
            v = v * v_max / np.linalg.norm(v[:3])
        es.append(np.linalg.norm(s - s_star)); vs.append(v)
        T = T @ exp_se3(v * dt)
        Ts.append(T.copy())
        s, Z = project_world(cam, T, P_w)
        ss.append(s); Zs.append(Z)
    es.append(np.linalg.norm(s - s_star))
    return dict(t=np.arange(n_steps + 1) * dt, T=np.array(Ts), s=np.array(ss), Z=np.array(Zs), e=np.array(es),
                v=np.array(vs), s_star=s_star)


def simulate_pbvs(cam: PinholeCamera, P_obj, T_wo, T_wc0, T_wc_star, lam: float = 0.5, dt: float = 0.05,
                  n_steps: int = 200, cam_ctrl: PinholeCamera | None = None, noise_px: float = 0.0, rng=None):
    """Eye-in-hand PBVS: the camera pose is *estimated* from the image of the planar target, then servoed.

    ``P_obj`` are the target points in the object frame (``z = 0``) and ``T_wo`` the target pose. The pose is
    recovered by :func:`planar_pose` with the (possibly wrong) intrinsics ``cam_ctrl`` from noisy pixels;
    the control law :func:`pbvs_velocity` is evaluated on the *estimated* camera pose relative to the object,
    against the desired one. Returns the same dictionary as :func:`simulate_ibvs` (plus ``pose_err``: the
    translation error of the estimate).
    """
    cam_ctrl = cam_ctrl or cam
    rng = rng or np.random.default_rng(0)
    P_w = (np.asarray(P_obj) @ T_wo[:3, :3].T) + T_wo[:3, 3]
    s_star, _ = project_world(cam, T_wc_star, P_w)
    T_oc_star = inv_T(T_wo) @ T_wc_star
    T = np.array(T_wc0, dtype=float)
    Ts, ss, Zs, es, vs, perr = [T.copy()], [], [], [], [], []
    s, Z = project_world(cam, T, P_w)
    ss.append(s); Zs.append(Z)
    for _ in range(n_steps):
        s_meas = s + noise_px * rng.standard_normal(s.shape)
        T_co_est = planar_pose(P_obj, s_meas, cam_ctrl)
        T_oc_est = inv_T(T_co_est)
        v, _, _ = pbvs_velocity(T_oc_est, T_oc_star, lam)
        true_oc = inv_T(T_wo) @ T
        perr.append(np.linalg.norm(T_oc_est[:3, 3] - true_oc[:3, 3]))
        es.append(np.linalg.norm(s - s_star)); vs.append(v)
        T = T @ exp_se3(v * dt)
        Ts.append(T.copy())
        s, Z = project_world(cam, T, P_w)
        ss.append(s); Zs.append(Z)
    es.append(np.linalg.norm(s - s_star))
    return dict(t=np.arange(n_steps + 1) * dt, T=np.array(Ts), s=np.array(ss), Z=np.array(Zs), e=np.array(es),
                v=np.array(vs), s_star=s_star, pose_err=np.array(perr))


def simulate_eye_to_hand(cam: PinholeCamera, T_wc, P_obj, T_co0, T_co_star, lam: float = 0.5, dt: float = 0.05,
                         n_steps: int = 200):
    """Eye-to-hand servoing: a *fixed* camera watches points ``P_obj`` rigidly attached to a moving object.

    The object pose in the camera frame is ``T_co``. Moving the object with the twist ``V_o`` (expressed in the
    camera frame) changes the image by ``s_dot = -L V_o``, so the law is ``V_o = +lam * pinv(L) (s - s*)`` — the
    *opposite* sign to the eye-in-hand law. Returns a dict with ``t``, ``T_co``, ``s`` and ``e``.
    """
    P_obj = np.asarray(P_obj, dtype=float)
    pts = lambda T: P_obj @ T[:3, :3].T + T[:3, 3]
    s_star = cam.project(pts(T_co_star))
    T = np.array(T_co0, dtype=float)
    Ts, ss, es = [T.copy()], [cam.project(pts(T))], [np.linalg.norm(cam.project(pts(T)) - s_star)]
    for _ in range(n_steps):
        Pc = pts(T)
        s = cam.project(Pc)
        L = interaction_matrix_px(s, Pc[:, 2], cam)
        V = lam * np.linalg.pinv(L) @ (s - s_star).ravel()
        v, w = V[:3], V[3:]
        R_new = rot_axis_angle(w, np.linalg.norm(w) * dt) @ T[:3, :3] if np.linalg.norm(w) > 1e-12 else T[:3, :3]
        t_new = T[:3, 3] + (v + np.cross(w, T[:3, 3])) * dt
        T = make_T(R_new, t_new)
        Ts.append(T.copy())
        s = cam.project(pts(T))
        ss.append(s)
        es.append(np.linalg.norm(s - s_star))
    return dict(t=np.arange(n_steps + 1) * dt, T_co=np.array(Ts), s=np.array(ss), e=np.array(es), s_star=s_star)

"""Wheeled mobile robots: kinematic models, non-holonomic constraints and motion control.

Pose convention: ``q = [x, y, theta]`` -- position of the reference point (the middle of the rear/drive axle)
in the world frame and heading ``theta`` measured counter-clockwise from the world ``x`` axis.  The inputs of
the unicycle are the forward speed ``v`` and the turning rate ``omega``.  All functions accept a single pose
``(3,)`` or a batch ``(3, N)``.

Contents

* kinematics: :func:`unicycle_deriv`, :func:`unicycle_step` (exact circular arcs), differential-drive
  wheel/body conversions, the bicycle model;
* constraint and controllability tools: :func:`pfaffian_residual`, :func:`lie_bracket`,
  :func:`sinusoidal_parking_input`;
* controllers: :func:`go_to_point`, :func:`polar_pose_control` (Siegwart-Aicardi polar coordinates),
  :func:`lyapunov_pose_control`, :func:`tracking_control` (Kanayama), reference trajectories,
  :func:`potential_field_velocity` for obstacle avoidance;
* a small simulator, :func:`simulate_unicycle`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "wrap_pi", "unicycle_deriv", "unicycle_step", "diffdrive_to_body", "body_to_diffdrive", "bicycle_deriv",
    "pfaffian_residual", "lie_bracket", "unicycle_to_chained", "chained_to_unicycle",
    "sinusoidal_parking_input", "commutator_displacement",
    "polar_coordinates", "go_to_point", "polar_pose_control", "lyapunov_pose_control", "pose_in_goal_frame",
    "tracking_error", "tracking_control", "circle_reference", "figure_eight_reference",
    "potential_field_velocity", "limit_wheel_speeds", "simulate_unicycle",
]


def wrap_pi(a):
    """Wrap an angle (or array) to ``[-pi, pi)``."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


# --------------------------------------------------------------------------------------------------
# kinematic models
# --------------------------------------------------------------------------------------------------
def unicycle_deriv(q, v, omega):
    """``[xdot, ydot, thetadot] = [v cos(theta), v sin(theta), omega]``."""
    q = np.asarray(q, dtype=float)
    return np.stack([v * np.cos(q[2]), v * np.sin(q[2]), omega * np.ones_like(q[2])])


def unicycle_step(q, v, omega, dt):
    """Exact integration over ``dt`` of the unicycle with constant ``(v, omega)`` (a circular arc or a line)."""
    q = np.asarray(q, dtype=float)
    x, y, th = q
    th1 = th + omega * dt
    small = np.abs(omega) < 1e-9
    safe = np.where(small, 1.0, omega)
    dx = np.where(small, v * dt * np.cos(th), v / safe * (np.sin(th1) - np.sin(th)))
    dy = np.where(small, v * dt * np.sin(th), -v / safe * (np.cos(th1) - np.cos(th)))
    return np.stack([x + dx, y + dy, th1])


def diffdrive_to_body(omega_r, omega_l, r, track):
    """Wheel angular speeds -> ``(v, omega)`` for wheel radius ``r`` and track width ``track`` (wheel separation).

    ``v = r (omega_r + omega_l) / 2`` and ``omega = r (omega_r - omega_l) / track``.
    """
    return 0.5 * r * (omega_r + omega_l), r * (omega_r - omega_l) / track


def body_to_diffdrive(v, omega, r, track):
    """Inverse of :func:`diffdrive_to_body`: ``(omega_r, omega_l)``."""
    return (v + 0.5 * track * omega) / r, (v - 0.5 * track * omega) / r


def bicycle_deriv(q, v, delta, wheelbase):
    """Rear-axle bicycle (car-like) model: ``thetadot = v tan(delta) / wheelbase``."""
    q = np.asarray(q, dtype=float)
    return np.stack([v * np.cos(q[2]), v * np.sin(q[2]), v * np.tan(delta) / wheelbase])


def pfaffian_residual(q, qdot):
    """Residual of the no-side-slip constraint ``xdot sin(theta) - ydot cos(theta) = 0`` (zero if satisfied)."""
    q, qdot = np.asarray(q, dtype=float), np.asarray(qdot, dtype=float)
    return qdot[0] * np.sin(q[2]) - qdot[1] * np.cos(q[2])


def lie_bracket(f, g, q, eps=1e-6):
    """Numerical Lie bracket ``[f, g](q) = (dg/dq) f - (df/dq) g`` of two vector fields ``f(q), g(q)``."""
    q = np.asarray(q, dtype=float)
    n = q.size
    Jf, Jg = np.empty((n, n)), np.empty((n, n))
    for i in range(n):
        d = np.zeros(n)
        d[i] = eps
        Jf[:, i] = (f(q + d) - f(q - d)) / (2 * eps)
        Jg[:, i] = (g(q + d) - g(q - d)) / (2 * eps)
    return Jg @ f(q) - Jf @ g(q)


def unicycle_to_chained(q):
    """Chained-form coordinates ``z = [theta, x cos(theta) + y sin(theta), x sin(theta) - y cos(theta)]``.

    With inputs ``w1 = omega`` and ``w2 = v - z3 omega`` the unicycle becomes
    ``z1' = w1, z2' = w2, z3' = z2 w1``.
    """
    x, y, th = np.asarray(q, dtype=float)
    return np.array([th, x * np.cos(th) + y * np.sin(th), x * np.sin(th) - y * np.cos(th)])


def chained_to_unicycle(z):
    """Inverse of :func:`unicycle_to_chained`."""
    th, z2, z3 = np.asarray(z, dtype=float)
    return np.array([z2 * np.cos(th) + z3 * np.sin(th), z2 * np.sin(th) - z3 * np.cos(th), th])


def sinusoidal_parking_input(t, q, a=1.0, b=0.5, period=2 * np.pi):
    """Sinusoidal steering input ``(v, omega)`` that shifts the unicycle sideways by ``-pi a b T^2 / (2 pi)^2``.

    In chained coordinates ``w1 = a sin(2 pi t / T)``, ``w2 = b cos(2 pi t / T)``; over one period
    ``z1, z2`` return to their initial values and ``z3`` changes by ``pi a b (T / 2 pi)^2``.
    The state-dependent term ``v = w2 + z3 omega`` is evaluated from the measured pose ``q``.
    """
    s = 2 * np.pi / period
    w1, w2 = a * np.sin(s * t), b * np.cos(s * t)
    z3 = unicycle_to_chained(q)[2]
    return w2 + z3 * w1, w1


def commutator_displacement(q0, eps, v_gain=1.0, w_gain=1.0, n=2000):
    """Net displacement after the four-flow motion ``(+v) (+omega) (-v) (-omega)``, each for time ``eps``.

    This is the "parallel parking" manoeuvre of Lie-bracket theory; for small ``eps`` the displacement is
    ``eps^2 [g_v, g_omega](q0)`` plus higher order terms. Returns the final pose minus ``q0``.
    """
    q = np.asarray(q0, dtype=float).copy()
    for v, w in [(v_gain, 0.0), (0.0, w_gain), (-v_gain, 0.0), (0.0, -w_gain)]:
        for _ in range(n):
            q = unicycle_step(q, v, w, eps / n)
    return q - np.asarray(q0, dtype=float)


# --------------------------------------------------------------------------------------------------
# controllers
# --------------------------------------------------------------------------------------------------
def pose_in_goal_frame(q, goal):
    """Express pose ``q`` in the frame of ``goal = [xg, yg, thetag]`` (goal at the origin, heading 0)."""
    q, goal = np.asarray(q, dtype=float), np.asarray(goal, dtype=float)
    c, s = np.cos(goal[2]), np.sin(goal[2])
    dx, dy = q[0] - goal[0], q[1] - goal[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap_pi(q[2] - goal[2])])


def polar_coordinates(q, goal):
    """Polar coordinates of Siegwart *et al.*: ``rho`` (distance to the goal), ``alpha`` (angle between the
    robot heading and the line to the goal) and ``beta`` (angle between that line and the goal heading).

    ``alpha = -theta + atan2(dy, dx)`` and ``beta = -theta - alpha`` in the goal frame, wrapped to ``[-pi, pi)``.
    """
    x, y, th = pose_in_goal_frame(q, goal)
    rho = np.hypot(x, y)
    gamma = np.arctan2(-y, -x)                # direction from the robot to the goal
    alpha = wrap_pi(gamma - th)
    beta = wrap_pi(-th - alpha)
    return rho, alpha, beta


def go_to_point(q, point, k_v=1.0, k_w=3.0, v_max=np.inf):
    """Go-to-point law: ``v = k_v rho cos(alpha)`` (never negative), ``omega = k_w alpha``."""
    q, point = np.asarray(q, dtype=float), np.asarray(point, dtype=float)
    dx, dy = point[0] - q[0], point[1] - q[1]
    rho = np.hypot(dx, dy)
    alpha = wrap_pi(np.arctan2(dy, dx) - q[2])
    v = np.minimum(k_v * rho * np.maximum(np.cos(alpha), 0.0), v_max)
    return v, k_w * alpha


def polar_pose_control(q, goal, k_rho=1.0, k_alpha=3.0, k_beta=-1.0, reverse=True):
    """Pose regulation in polar coordinates: ``v = k_rho rho``, ``omega = k_alpha alpha + k_beta beta``.

    Local exponential stability needs ``k_rho > 0``, ``k_beta < 0`` and ``k_alpha > k_rho``.  The law assumes
    the goal is in front of the robot (``|alpha| < pi/2``).  If ``reverse`` is true and it is behind, the robot
    is treated as a unicycle whose front is its back: heading and goal heading are both rotated by ``pi``,
    the law is evaluated, and the forward speed changes sign (``omega`` is unchanged).
    """
    q, goal = np.asarray(q, dtype=float), np.asarray(goal, dtype=float)
    rho, alpha, beta = polar_coordinates(q, goal)
    if reverse and abs(alpha) > np.pi / 2:
        rho, alpha, beta = polar_coordinates(q + np.array([0.0, 0.0, np.pi]), goal + np.array([0.0, 0.0, np.pi]))
        return -k_rho * rho, k_alpha * alpha + k_beta * beta
    return k_rho * rho, k_alpha * alpha + k_beta * beta


def lyapunov_pose_control(q, goal, gamma=1.0, k=3.0, h=1.0):
    """Aicardi-type Lyapunov law:  ``v = gamma rho cos(alpha)``,
    ``omega = k alpha + gamma (sin(alpha) cos(alpha) / alpha) (alpha - h beta)``.

    With ``V = (lam rho^2 + alpha^2 + h beta^2) / 2`` one finds ``Vdot = -lam gamma rho^2 cos^2(alpha) - k alpha^2 <= 0``
    (``lam = 1``).  Valid for ``|alpha| < pi/2``.
    """
    rho, alpha, beta = polar_coordinates(q, goal)
    sinc_like = np.sinc(alpha / np.pi) * np.cos(alpha)          # sin(alpha) cos(alpha) / alpha
    v = gamma * rho * np.cos(alpha)
    omega = k * alpha + gamma * sinc_like * (alpha - h * beta)
    return v, omega


def tracking_error(q, q_ref):
    """Reference minus actual pose, expressed in the robot frame: ``[e_x, e_y, e_theta]``."""
    q, q_ref = np.asarray(q, dtype=float), np.asarray(q_ref, dtype=float)
    c, s = np.cos(q[2]), np.sin(q[2])
    dx, dy = q_ref[0] - q[0], q_ref[1] - q[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap_pi(q_ref[2] - q[2])])


def tracking_control(q, q_ref, v_ref, w_ref, kx=2.0, ky=8.0, kth=3.0):
    """Kanayama's feedforward-plus-feedback tracking law.

    ``v = v_ref cos(e_th) + kx e_x``,  ``omega = w_ref + ky v_ref e_y + kth sin(e_th)``.
    With ``V = (e_x^2 + e_y^2)/2 + (1 - cos e_th)/ky`` it gives ``Vdot = -kx e_x^2 - (kth/ky) sin^2(e_th)``.
    """
    ex, ey, eth = tracking_error(q, q_ref)
    return v_ref * np.cos(eth) + kx * ex, w_ref + ky * v_ref * ey + kth * np.sin(eth)


def circle_reference(radius=1.0, speed=0.5, centre=(0.0, 0.0)):
    """Reference ``ref(t) -> (pose, v, omega)`` for a counter-clockwise circle starting at ``centre + (radius, 0)``."""
    Om = speed / radius

    def ref(t):
        a = Om * t
        pose = np.array([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a), a + np.pi / 2])
        return pose, speed, Om

    return ref


def figure_eight_reference(a=1.5, period=24.0):
    """Reference for the lemniscate-like figure eight ``x = a sin(Om t)``, ``y = (a/2) sin(2 Om t)``."""
    Om = 2 * np.pi / period

    def ref(t):
        xd, yd = a * Om * np.cos(Om * t), a * Om * np.cos(2 * Om * t)
        xdd, ydd = -a * Om ** 2 * np.sin(Om * t), -2 * a * Om ** 2 * np.sin(2 * Om * t)
        v = np.hypot(xd, yd)
        pose = np.array([a * np.sin(Om * t), 0.5 * a * np.sin(2 * Om * t), np.arctan2(yd, xd)])
        return pose, v, (xd * ydd - yd * xdd) / v ** 2

    return ref


def potential_field_velocity(p, goal, obstacles, k_att=1.0, k_rep=0.5, d0=1.0, f_max=2.0):
    """Negative gradient of the artificial potential (Khatib): attractive paraboloid plus repulsive barrier.

    ``obstacles`` is a list of ``(cx, cy, radius)``.  The repulsive potential of an obstacle at clearance
    ``d`` is ``(k_rep / 2) (1/d - 1/d0)^2`` for ``d < d0`` and zero otherwise.  Returns the desired velocity
    vector ``-grad U`` clipped to magnitude ``f_max``.
    """
    p, goal = np.asarray(p, dtype=float), np.asarray(goal, dtype=float)
    f = -k_att * (p - goal)
    for cx, cy, r in obstacles:
        diff = p - np.array([cx, cy])
        dist = np.linalg.norm(diff)
        d = max(dist - r, 1e-6)
        if d < d0:
            f = f + k_rep * (1.0 / d - 1.0 / d0) / d ** 2 * diff / dist
    n = np.linalg.norm(f)
    return f if n <= f_max else f * (f_max / n)


def limit_wheel_speeds(v, omega, r, track, omega_max):
    """Scale ``(v, omega)`` uniformly so that neither wheel exceeds ``omega_max`` (keeps the path curvature)."""
    wr, wl = body_to_diffdrive(v, omega, r, track)
    s = max(abs(wr), abs(wl)) / omega_max
    if s > 1.0:
        return v / s, omega / s
    return v, omega


# --------------------------------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------------------------------
def simulate_unicycle(controller, q0, dt, T, v_max=np.inf, w_max=np.inf):
    """Simulate the unicycle under ``(v, omega) = controller(t, q)`` (zero-order hold, exact arcs).

    Returns ``t`` ``(K+1,)``, ``Q`` ``(K+1, 3)`` and ``U`` ``(K, 2)``.
    """
    q = np.array(q0, dtype=float)
    K = int(round(T / dt))
    t = np.arange(K + 1) * dt
    Q, U = np.empty((K + 1, 3)), np.empty((K, 2))
    Q[0] = q
    for k in range(K):
        v, w = controller(t[k], q)
        v, w = float(np.clip(v, -v_max, v_max)), float(np.clip(w, -w_max, w_max))
        U[k] = v, w
        q = unicycle_step(q, v, w, dt)
        Q[k + 1] = q
    return t, Q, U

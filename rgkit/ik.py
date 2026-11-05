"""Inverse kinematics: closed-form solvers, a trigonometric toolbox and iterative (numerical) solvers.

This module backs Chapter 4 of the MFoR notebooks. It follows the structure of Craig, *Introduction to
Robotics*, Chapter 4: the algebraic and geometric closed-form solutions of planar arms, the *kinematic
decoupling* of arms with a spherical wrist, and (beyond Craig) the standard iterative methods used when no
closed form exists. Everything is written for clarity first; all functions are exercised by
``tests/test_ik.py`` and, where a MuJoCo robot model is available, cross-checked against the simulator.

Conventions
-----------
* Angles are in **radians**; the solvers return angles wrapped to ``(-pi, pi]``.
* Closed-form solvers return a NumPy array with **one row per solution** (shape ``(k, n)``; ``k = 0`` means
  that the goal is outside the workspace). The order of the rows is documented for each function.
* D-H tables use ``rgkit.kinematics`` (Craig's *modified* convention): a row is ``(alpha, a, d, theta, "R")``.
* Numerical solvers use the error ``e(q) = x_d - F(q)`` and the Jacobian ``J(q) = dF/dq``. Chapter 5 develops
  Jacobians in depth; here they are only used as a tool (finite differences, planar closed form, or MuJoCo).

Contents
--------
Trigonometric toolbox
    :func:`wrap_angle`, :func:`solve_sin`, :func:`solve_cos`, :func:`solve_a_cos_b_sin`,
    :func:`solve_half_angle`, :func:`solve_two_equations`
Planar and simple spatial arms
    :func:`fk_2r`, :func:`ik_2r`, :func:`ik_2r_geometric`, :func:`fk_3r_planar`, :func:`ik_3r_planar`,
    :func:`fk_articulated`, :func:`ik_articulated`, :func:`fk_scara`, :func:`ik_scara`
Arms with a spherical wrist and the UR family
    :func:`puma_dh`, :func:`ik_puma`, :class:`URGeometry`, :func:`ik_ur`
Choosing among solutions
    :func:`nearest_solution`, :func:`within_limits`
Numerical inverse kinematics
    :func:`rotation_error`, :func:`numeric_jacobian`, :func:`jacobian_planar`, :func:`solve_ik`,
    :class:`IKResult`, :class:`MujocoArm`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .kinematics import fk_dh
from .transforms import axis_angle_from_rot, rotx, rotz

__all__ = [
    "wrap_angle", "solve_sin", "solve_cos", "solve_a_cos_b_sin", "solve_half_angle", "solve_two_equations",
    "fk_2r", "ik_2r", "ik_2r_geometric", "fk_3r_planar", "ik_3r_planar",
    "fk_articulated", "ik_articulated", "fk_scara", "ik_scara",
    "puma_dh", "ik_puma", "URGeometry", "ur_dh", "fk_ur", "ik_ur",
    "nearest_solution", "within_limits",
    "rotation_error", "numeric_jacobian", "jacobian_planar", "solve_ik", "IKResult", "MujocoArm",
]

_TOL = 1e-9
_TANGENT = 1e-7      # |sqrt(1 - c^2)| below this means a double root (rounding turns an exact 0 into ~1e-8)


# ---------------------------------------------------------------------------------------------------------
# Trigonometric toolbox (Craig, Appendix C)
# ---------------------------------------------------------------------------------------------------------
def wrap_angle(a):
    """Wrap an angle (or array of angles) to ``(-pi, pi]``."""
    a = np.asarray(a, dtype=float)
    return np.arctan2(np.sin(a), np.cos(a))


def solve_sin(a: float, tol: float = _TOL) -> np.ndarray:
    """All ``theta`` in ``(-pi, pi]`` with ``sin(theta) = a`` (none if ``|a| > 1``)."""
    if abs(a) > 1 + tol:
        return np.empty(0)
    a = float(np.clip(a, -1.0, 1.0))
    c = np.sqrt(max(0.0, 1.0 - a * a))
    if c < tol:
        return np.array([np.arctan2(a, 0.0)])
    return np.array([np.arctan2(a, c), np.arctan2(a, -c)])


def solve_cos(b: float, tol: float = _TOL) -> np.ndarray:
    """All ``theta`` in ``(-pi, pi]`` with ``cos(theta) = b`` (none if ``|b| > 1``)."""
    if abs(b) > 1 + tol:
        return np.empty(0)
    b = float(np.clip(b, -1.0, 1.0))
    s = np.sqrt(max(0.0, 1.0 - b * b))
    if s < tol:
        return np.array([np.arctan2(0.0, b)])
    return np.array([np.arctan2(s, b), np.arctan2(-s, b)])


def solve_a_cos_b_sin(a: float, b: float, c: float, tol: float = _TOL) -> np.ndarray:
    """All ``theta`` with ``a cos(theta) + b sin(theta) = c``  (Craig App. C, ``atan2`` form).

    Writing ``a = rho cos(psi)``, ``b = rho sin(psi)`` gives ``cos(theta - psi) = c / rho``, hence
    ``theta = atan2(b, a) +- atan2(sqrt(a^2 + b^2 - c^2), c)``. Returns 0, 1 (tangent) or 2 angles.
    """
    rho = np.hypot(a, b)
    if rho < tol:
        return np.empty(0)
    d = c / rho
    if abs(d) > 1 + tol:
        return np.empty(0)
    psi = np.arctan2(b, a)
    root = np.sqrt(max(0.0, rho * rho - c * c))
    delta = np.arctan2(root, c)
    if root / rho < _TANGENT:
        return np.array([wrap_angle(psi + delta)])
    return np.array([wrap_angle(psi + delta), wrap_angle(psi - delta)])


def solve_half_angle(a: float, b: float, c: float, tol: float = _TOL) -> np.ndarray:
    """Solve ``a cos(theta) + b sin(theta) = c`` with ``u = tan(theta/2)`` (Craig §4.5).

    ``cos(theta) = (1 - u^2)/(1 + u^2)`` and ``sin(theta) = 2u/(1 + u^2)`` turn the equation into the
    quadratic ``(a + c) u^2 - 2 b u + (c - a) = 0``. The solution ``theta = pi`` (``u -> infinity``) is
    lost by the substitution and is added back when ``a + c = 0``.
    """
    sols = []
    if abs(a + c) < tol:                       # leading coefficient vanishes: theta = pi is a root
        sols.append(np.pi)
        if abs(b) > tol:
            sols.append(2.0 * np.arctan((c - a) / (2.0 * b)))
    else:
        disc = b * b - (a + c) * (c - a)       # = a^2 + b^2 - c^2
        if disc < -tol:
            return np.empty(0)
        sq = np.sqrt(max(disc, 0.0))
        for u in ((b + sq) / (a + c), (b - sq) / (a + c)):
            sols.append(2.0 * np.arctan(u))
        if sq < _TANGENT * max(abs(a + c), abs(b), 1e-300):
            sols = sols[:1]
    return wrap_angle(np.array(sols))


def solve_two_equations(a: float, b: float, c: float, d: float) -> float:
    """Solve ``a cos(t) - b sin(t) = c`` and ``a sin(t) + b cos(t) = d`` together (Craig App. C): ``atan2(ad - bc, ac + bd)``."""
    return float(np.arctan2(a * d - b * c, a * c + b * d))


# ---------------------------------------------------------------------------------------------------------
# Planar arms
# ---------------------------------------------------------------------------------------------------------
def fk_2r(q, L1: float, L2: float) -> np.ndarray:
    """Tip position ``(x, y)`` of the planar 2R arm."""
    q1, q2 = q
    return np.array([L1 * np.cos(q1) + L2 * np.cos(q1 + q2), L1 * np.sin(q1) + L2 * np.sin(q1 + q2)])


def ik_2r(x: float, y: float, L1: float, L2: float, tol: float = _TOL) -> np.ndarray:
    """Algebraic IK of the planar 2R arm (Craig §4.4).

    ``c2 = (x^2 + y^2 - L1^2 - L2^2) / (2 L1 L2)``, ``theta2 = atan2(+-sqrt(1 - c2^2), c2)`` and
    ``theta1 = atan2(y, x) - atan2(L2 s2, L1 + L2 c2)``. Rows: ``theta2 > 0`` first ("elbow down" for a
    target on the right), then ``theta2 < 0`` ("elbow up"). One row on the workspace boundary, none outside.
    """
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    if abs(c2) > 1 + tol:
        return np.empty((0, 2))
    c2 = float(np.clip(c2, -1.0, 1.0))
    s2m = np.sqrt(max(0.0, 1.0 - c2 * c2))
    signs = [1.0] if s2m < _TANGENT else [1.0, -1.0]
    out = []
    for sg in signs:
        s2 = sg * s2m if s2m >= _TANGENT else 0.0
        th2 = np.arctan2(s2, c2)
        th1 = np.arctan2(y, x) - np.arctan2(L2 * s2, L1 + L2 * c2)
        out.append([th1, th2])
    return wrap_angle(np.array(out))


def ik_2r_geometric(x: float, y: float, L1: float, L2: float, tol: float = _TOL) -> np.ndarray:
    """Geometric IK of the 2R arm with the law of cosines (same rows and order as :func:`ik_2r`).

    ``r = |(x, y)|``; ``gamma = atan2(y, x)``; interior angles ``alpha`` (at the base) and ``beta`` (at the elbow)
    from the triangle ``(L1, L2, r)``. Elbow down: ``(gamma - alpha, pi - beta)``; elbow up: ``(gamma + alpha, beta - pi)``.
    """
    r = np.hypot(x, y)
    if r > L1 + L2 + tol or r < abs(L1 - L2) - tol or r < tol:
        return np.empty((0, 2))
    cos_beta = (L1 * L1 + L2 * L2 - r * r) / (2 * L1 * L2)
    cos_alpha = (L1 * L1 + r * r - L2 * L2) / (2 * L1 * r)
    beta = np.arccos(np.clip(cos_beta, -1, 1))
    alpha = np.arccos(np.clip(cos_alpha, -1, 1))
    gamma = np.arctan2(y, x)
    rows = [[gamma - alpha, np.pi - beta], [gamma + alpha, beta - np.pi]]
    if abs(np.sin(beta)) < _TANGENT:
        rows = rows[:1]
    return wrap_angle(np.array(rows))


def fk_3r_planar(q, L1: float, L2: float, L3: float) -> np.ndarray:
    """Tip pose ``(x, y, phi)`` of the planar 3R arm; ``phi = q1 + q2 + q3`` is the orientation of the last link."""
    q1, q2, q3 = q
    phi = q1 + q2 + q3
    return np.array([L1 * np.cos(q1) + L2 * np.cos(q1 + q2) + L3 * np.cos(phi),
                     L1 * np.sin(q1) + L2 * np.sin(q1 + q2) + L3 * np.sin(phi), phi])


def ik_3r_planar(x: float, y: float, phi: float, L1: float, L2: float, L3: float) -> np.ndarray:
    """Algebraic IK of the planar 3R arm for the pose ``(x, y, phi)`` of the tool tip (Craig §4.4).

    The wrist point ``(x - L3 cos(phi), y - L3 sin(phi))`` is reached by a 2R problem (rows as in
    :func:`ik_2r`), then ``theta3 = phi - theta1 - theta2``.
    """
    xw, yw = x - L3 * np.cos(phi), y - L3 * np.sin(phi)
    s = ik_2r(xw, yw, L1, L2)
    if len(s) == 0:
        return np.empty((0, 3))
    return wrap_angle(np.column_stack([s, phi - s[:, 0] - s[:, 1]]))


# ---------------------------------------------------------------------------------------------------------
# Simple spatial arms: articulated (elbow) arm and SCARA
# ---------------------------------------------------------------------------------------------------------
def fk_articulated(q, L2: float, L3: float, d1: float = 0.0) -> np.ndarray:
    """Wrist-point position of the articulated 3R arm: waist ``q1`` about Z, shoulder ``q2`` and elbow ``q3`` measured
    from the horizontal in the vertical plane; ``d1`` is the shoulder height."""
    q1, q2, q3 = q
    rho = L2 * np.cos(q2) + L3 * np.cos(q2 + q3)
    return np.array([np.cos(q1) * rho, np.sin(q1) * rho, d1 + L2 * np.sin(q2) + L3 * np.sin(q2 + q3)])


def ik_articulated(p, L2: float, L3: float, d1: float = 0.0, tol: float = _TOL) -> np.ndarray:
    """Up to **four** IK solutions of the articulated arm for the wrist position ``p`` (Craig §4.4; Example "front/over the shoulder").

    Rows ``[q1, q2, q3]`` in the order: front/elbow-down, front/elbow-up, over-the-shoulder/elbow-down,
    over-the-shoulder/elbow-up. ``p`` on the waist axis is singular (``q1`` arbitrary): ``q1 = 0`` is returned.
    """
    x, y, z = p
    h = z - d1
    r = np.hypot(x, y)
    rows = []
    for back in (False, True):
        q1 = np.arctan2(y, x) if r > tol else 0.0
        rho = r
        if back:
            q1, rho = q1 + np.pi, -r
        c3 = (rho * rho + h * h - L2 * L2 - L3 * L3) / (2 * L2 * L3)
        if abs(c3) > 1 + tol:
            return np.empty((0, 3))
        c3 = float(np.clip(c3, -1, 1))
        s3m = np.sqrt(max(0.0, 1.0 - c3 * c3))
        for sg in ((1.0, -1.0) if s3m > _TANGENT else (1.0,)):
            s3 = sg * s3m if s3m > _TANGENT else 0.0
            q3 = np.arctan2(s3, c3)
            q2 = np.arctan2(h, rho) - np.arctan2(L3 * s3, L2 + L3 * c3)
            rows.append([q1, q2, q3])
    return wrap_angle(np.array(rows))


def fk_scara(q, L1: float, L2: float, d4: float = 0.0) -> np.ndarray:
    """SCARA ``(theta1, theta2, d3, theta4)`` -> ``(x, y, z, phi)``; ``d3`` is the downward stroke, ``phi = q1 + q2 - q4``."""
    q1, q2, d3, q4 = q
    return np.array([L1 * np.cos(q1) + L2 * np.cos(q1 + q2), L1 * np.sin(q1) + L2 * np.sin(q1 + q2),
                     -d3 - d4, q1 + q2 - q4])


def ik_scara(x: float, y: float, z: float, phi: float, L1: float, L2: float, d4: float = 0.0) -> np.ndarray:
    """SCARA IK: ``(q1, q2)`` from the 2R formulas, ``d3 = -z - d4``, ``q4 = q1 + q2 - phi``. Rows ``[q1, q2, d3, q4]``
    (``q2 > 0`` first)."""
    s = ik_2r(x, y, L1, L2)
    if len(s) == 0:
        return np.empty((0, 4))
    out = np.column_stack([s[:, 0], s[:, 1], np.full(len(s), -z - d4), wrap_angle(s[:, 0] + s[:, 1] - phi)])
    return out


# ---------------------------------------------------------------------------------------------------------
# Arms with a spherical wrist: PUMA-type kinematic decoupling
# ---------------------------------------------------------------------------------------------------------
def puma_dh(a2: float, a3: float, d3: float, d4: float):
    """D-H table (Craig, modified) of a PUMA-type arm: shoulder offset ``d3``, links ``a2``, ``a3``, forearm offset ``d4``,
    spherical wrist (axes 4, 5, 6 meet at the origin of frames 4, 5, 6)."""
    hp = np.pi / 2
    return [(0.0, 0.0, 0.0, 0.0, "R"), (-hp, 0.0, 0.0, 0.0, "R"), (0.0, a2, d3, 0.0, "R"),
            (-hp, a3, d4, 0.0, "R"), (hp, 0.0, 0.0, 0.0, "R"), (-hp, 0.0, 0.0, 0.0, "R")]


def ik_puma(T, a2: float, a3: float, d3: float, d4: float, tol: float = _TOL, return_branches: bool = False):
    """All IK solutions (at most **eight**) of the PUMA-type arm of :func:`puma_dh` for the tool pose ``T = 0_6T``.

    Kinematic decoupling: the wrist centre ``p = T[:3, 3]`` fixes ``q1, q2, q3`` (up to 4 solutions), the rotation
    ``3_6R = 0_3R^T R`` fixes ``q4, q5, q6`` (2 solutions each). Branch signs ``(s1, s3, s5)``: ``s1 = +1`` "arm in front"
    (``-1`` the other root of the shoulder equation), ``s3 = +1`` / ``-1`` the sign of the root in the elbow equation,
    ``s5 = +1`` / ``-1`` the sign of ``sin(q5)`` (wrist "flip"). At a wrist singularity ``sin(q5) = 0`` only ``q4 + q6`` is
    determined and ``q6 = 0`` is returned for each arm solution.
    """
    T = np.asarray(T, float)
    R, p = T[:3, :3], T[:3, 3]
    px, py, pz = p
    rows, branches = [], []
    # shoulder: -s1 px + c1 py = d3
    q1s = solve_a_cos_b_sin(py, -px, d3, tol)
    for i1, q1 in enumerate(q1s):
        s1_sign = 1 if i1 == 0 else -1
        c1, s1 = np.cos(q1), np.sin(q1)
        X = c1 * px + s1 * py                     # radial coordinate in the arm plane
        # elbow: |p|^2 = X^2 + d3^2 + pz^2 = a2^2 + a3^2 + d3^2 + d4^2 + 2 a2 (a3 c3 - d4 s3)
        K = (px * px + py * py + pz * pz - a2 * a2 - a3 * a3 - d3 * d3 - d4 * d4) / (2.0 * a2)
        for i3, q3 in enumerate(solve_a_cos_b_sin(a3, -d4, K, tol)):
            c3, s3 = np.cos(q3), np.sin(q3)
            k1, k2 = a2 + a3 * c3 - d4 * s3, a3 * s3 + d4 * c3
            q2 = np.arctan2(-pz, X) - np.arctan2(k2, k1)
            # wrist: R36 = R03^T R, with R03 = Rz(q1) Rx(-90 deg) Rz(q2 + q3)
            R03 = rotz(q1) @ rotx(-np.pi / 2) @ rotz(q2 + q3)
            R36 = R03.T @ R
            c5 = float(np.clip(R36[1, 2], -1, 1))
            s5m = np.sqrt(R36[0, 2] ** 2 + R36[2, 2] ** 2)
            for s5_sign in ((1, -1) if s5m > tol else (1,)):
                if s5m > tol:
                    q5 = np.arctan2(s5_sign * s5m, c5)
                    q4 = np.arctan2(s5_sign * R36[2, 2], -s5_sign * R36[0, 2])
                    q6 = np.arctan2(-s5_sign * R36[1, 1], s5_sign * R36[1, 0])
                else:                              # singular wrist: q4 + q6 only; choose q6 = 0
                    q5 = np.arctan2(0.0, c5)
                    q6 = 0.0
                    q4 = np.arctan2(-R36[0, 1], R36[0, 0]) if c5 > 0 else np.arctan2(R36[0, 1], -R36[0, 0])
                rows.append([q1, q2, q3, q4, q5, q6])
                branches.append((s1_sign, 1 if i3 == 0 else -1, s5_sign))
    out = wrap_angle(np.array(rows)) if rows else np.empty((0, 6))
    return (out, branches) if return_branches else out


# ---------------------------------------------------------------------------------------------------------
# Universal-Robots-type arm (three parallel axes + offset, non-spherical wrist)
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class URGeometry:
    """Dimensions (metres) of a UR-type arm in the modified D-H table returned by :meth:`dh_table`.

    ``d1`` shoulder height, ``a2``/``a3`` upper-arm/forearm length, ``d4`` lateral offset of wrist axis 5 from the
    arm plane, ``d5`` and ``d6`` the wrist offsets along axes 5 and 6.
    """
    d1: float
    a2: float
    a3: float
    d4: float
    d5: float
    d6: float

    def dh_table(self):
        return ur_dh(self)


def ur_dh(g: URGeometry):
    """Modified D-H table of a UR-type arm whose zero configuration is the stretched arm along ``+X_0``.

    Choosing ``alpha_1 = -90 deg`` makes ``Z_2, Z_3, Z_4`` point along ``+Y_0`` at zero, the direction of the joint axes in
    the manufacturer's base frame, so the joint angles of this table coincide with the angles of the robot controller.
    """
    hp = np.pi / 2
    return [(0.0, 0.0, g.d1, 0.0, "R"), (-hp, 0.0, 0.0, 0.0, "R"), (0.0, g.a2, 0.0, 0.0, "R"),
            (0.0, g.a3, g.d4, 0.0, "R"), (-hp, 0.0, g.d5, 0.0, "R"), (hp, 0.0, g.d6, 0.0, "R")]


def fk_ur(q, g: URGeometry) -> np.ndarray:
    """Flange pose ``0_6T`` of the UR-type arm (4x4)."""
    return fk_dh(ur_dh(g), q)[-1]


def ik_ur(T, g: URGeometry, tol: float = _TOL, return_branches: bool = False):
    """All IK solutions (at most **eight**) of a UR-type arm for the flange pose ``T = 0_6T``.

    Branch signs ``(s1, s5, s3)`` ("shoulder", "wrist", "elbow"): ``s1 = +1`` for the first root of the shoulder equation,
    ``s5 = +1`` for ``sin(q5) > 0``, ``s3 = +1`` for ``q3 > 0``. Derivation: see notebook 4.4. At a wrist singularity
    (``sin q5 = 0``) ``q6 = 0`` is returned.
    """
    T = np.asarray(T, float)
    R, p = T[:3, :3], T[:3, 3]
    x6, y6, z6 = R[:, 0], R[:, 1], R[:, 2]
    w = p - g.d6 * z6                                    # wrist point O5 (axes 5 and 6 intersect)
    rows, branches = [], []
    for i1, q1 in enumerate(solve_a_cos_b_sin(w[1], -w[0], g.d4, tol)):   # -s1 wx + c1 wy = d4
        c1, s1 = np.cos(q1), np.sin(q1)
        n = np.array([-s1, c1, 0.0])                     # direction of the parallel axes 2, 3, 4
        u = np.array([c1, s1, 0.0])                      # horizontal direction in the arm plane
        c5 = (p @ n - g.d4) / g.d6
        if abs(c5) > 1 + tol:
            continue
        c5 = float(np.clip(c5, -1, 1))
        s5m = np.sqrt(max(0.0, 1 - c5 * c5))
        for s5_sign in ((1, -1) if s5m > _TANGENT else (1,)):
            s5 = s5_sign * s5m
            q5 = np.arctan2(s5, c5)
            if s5m > _TANGENT:
                q6 = np.arctan2((n @ y6) / s5, -(n @ x6) / s5)
                x4 = (z6 - c5 * n) / s5
            else:
                q6 = 0.0
                x4 = c5 * x6
            q234 = np.arctan2(-x4[2], x4 @ u)
            O4 = w + g.d5 * (np.sin(q234) * u + np.cos(q234) * np.array([0.0, 0.0, 1.0]))
            r, hh = O4 @ u, O4[2] - g.d1
            for i3, (q2, q3) in enumerate(ik_2r(r, -hh, g.a2, g.a3, tol)):
                rows.append([q1, q2, q3, q234 - q2 - q3, q5, q6])
                branches.append((1 if i1 == 0 else -1, s5_sign, 1 if q3 > 0 else -1))
    out = wrap_angle(np.array(rows)) if rows else np.empty((0, 6))
    return (out, branches) if return_branches else out


# ---------------------------------------------------------------------------------------------------------
# Choosing among solutions
# ---------------------------------------------------------------------------------------------------------
def within_limits(solutions, lower, upper) -> np.ndarray:
    """Boolean mask of the rows of ``solutions`` whose angles lie in ``[lower, upper]`` after adding a multiple of ``2 pi``
    (a revolute joint with a range wider than ``2 pi`` can realise several of them; the first fit is used)."""
    sol = np.atleast_2d(solutions)
    lower, upper = np.broadcast_to(lower, sol.shape[1]), np.broadcast_to(upper, sol.shape[1])
    ok = np.ones(len(sol), bool)
    for i, row in enumerate(sol):
        for j, v in enumerate(row):
            cands = v + 2 * np.pi * np.arange(-2, 3)
            ok[i] &= bool(np.any((cands >= lower[j] - 1e-12) & (cands <= upper[j] + 1e-12)))
    return ok


def nearest_solution(solutions, q_ref, weights=None):
    """Return ``(index, q)`` of the solution closest to ``q_ref`` in the weighted joint-space distance (angles compared
    modulo ``2 pi``). Typical weights are larger for the big, slow proximal joints than for the wrist."""
    sol = np.atleast_2d(solutions)
    if len(sol) == 0:
        return None, None
    w = np.ones(sol.shape[1]) if weights is None else np.asarray(weights, float)
    d = np.sqrt(((wrap_angle(sol - q_ref) ** 2) * w).sum(axis=1))
    i = int(np.argmin(d))
    return i, sol[i]


# ---------------------------------------------------------------------------------------------------------
# Numerical inverse kinematics
# ---------------------------------------------------------------------------------------------------------
def rotation_error(R_d, R) -> np.ndarray:
    """Rotation vector ``k * theta`` of ``R_d R^T`` (the world-frame rotation that carries ``R`` onto ``R_d``)."""
    axis, theta = axis_angle_from_rot(np.asarray(R_d) @ np.asarray(R).T)
    return axis * theta


def numeric_jacobian(F: Callable, q, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian ``dF/dq`` of a vector function ``F`` (``m x n``)."""
    q = np.asarray(q, float)
    cols = []
    for j in range(len(q)):
        dq = np.zeros_like(q)
        dq[j] = eps
        cols.append((np.asarray(F(q + dq)) - np.asarray(F(q - dq))) / (2 * eps))
    return np.column_stack(cols)


def jacobian_planar(L, q) -> np.ndarray:
    """Position Jacobian (2 x n) of a planar nR arm with link lengths ``L``: column ``j`` is ``z x (p_tip - p_j)``."""
    L, q = np.asarray(L, float), np.asarray(q, float)
    ang = np.cumsum(q)
    px = np.cumsum(L * np.cos(ang))
    py = np.cumsum(L * np.sin(ang))
    tip = np.array([px[-1], py[-1]])
    J = np.zeros((2, len(q)))
    for j in range(len(q)):
        o = np.zeros(2) if j == 0 else np.array([px[j - 1], py[j - 1]])
        d = tip - o
        J[:, j] = [-d[1], d[0]]
    return J


@dataclass
class IKResult:
    """Outcome of :func:`solve_ik`. ``history`` holds, per iteration, the iterate ``q``, the error norm ``err``, the step norm
    ``step`` and the smallest singular value ``sigma_min`` of the Jacobian."""
    q: np.ndarray
    converged: bool
    iterations: int
    error_norm: float
    history: dict = field(default_factory=dict)


def solve_ik(err: Callable, jac: Callable, q0, method: str = "newton", alpha: Optional[float] = 0.5, lam: float = 0.05,
             tol: float = 1e-9, max_iter: int = 100, step_limit: Optional[float] = None, nullspace: Optional[Callable] = None,
             eps_dls: float = 0.05, rcond: float = 1e-10) -> IKResult:
    """Iterative IK: repeat ``q <- q + dq`` until ``|e| < tol`` where ``e = err(q)`` is the task error ``x_d - F(q)``.

    ``method``
        ``"newton"``      ``dq = J^{-1} e`` (square, non-singular ``J``; falls back to the pseudoinverse if ``J`` is not square),
        ``"pinv"``        ``dq = J^+ e`` (minimum-norm / least-squares, via the SVD),
        ``"gd"``          ``dq = alpha J^T e`` (Jacobian transpose; ``alpha=None`` uses the optimal step along ``J^T e``),
        ``"dls"``         ``dq = J^T (J J^T + lam^2 I)^{-1} e`` (damped least squares, constant ``lam``),
        ``"dls_adaptive"`` damped least squares with ``lam`` increased smoothly when the smallest singular value is below ``eps_dls``.
    ``nullspace(q)`` may return a joint-space vector ``z`` that is projected onto the null space of ``J`` and added
    (redundancy resolution, "pinv"/"dls" only). ``step_limit`` clips ``|dq|``.
    """
    q = np.array(q0, dtype=float)
    hist = dict(q=[q.copy()], err=[], step=[], sigma_min=[])
    converged = False
    k = 0
    for k in range(max_iter + 1):
        e = np.asarray(err(q), float)
        en = float(np.linalg.norm(e))
        hist["err"].append(en)
        if en < tol:
            converged = True
            break
        if k == max_iter or not np.isfinite(en):
            break
        J = np.atleast_2d(np.asarray(jac(q), float))
        sv = np.linalg.svd(J, compute_uv=False)
        hist["sigma_min"].append(float(sv[-1]) if J.shape[0] <= J.shape[1] else float(sv[-1]))
        m, n = J.shape
        if method == "newton":
            if m == n:
                try:
                    dq = np.linalg.solve(J, e)
                except np.linalg.LinAlgError:
                    dq = np.full(n, np.nan)
            else:
                dq = np.linalg.pinv(J, rcond=rcond) @ e
        elif method == "pinv":
            dq = np.linalg.pinv(J, rcond=rcond) @ e
        elif method == "gd":
            g = J.T @ e
            a = alpha if alpha is not None else float(e @ (J @ g)) / max(float((J @ g) @ (J @ g)), 1e-300)
            dq = a * g
        elif method in ("dls", "dls_adaptive"):
            l2 = lam ** 2
            if method == "dls_adaptive":
                smin = sv[-1] if m <= n else sv[-1]
                l2 = 0.0 if smin >= eps_dls else (lam ** 2) * (1.0 - (smin / eps_dls) ** 2)
            dq = J.T @ np.linalg.solve(J @ J.T + l2 * np.eye(m), e)
        else:
            raise ValueError(f"unknown method {method!r}")
        if nullspace is not None and method in ("pinv", "dls", "dls_adaptive", "newton"):
            z = np.asarray(nullspace(q), float)
            dq = dq + (np.eye(n) - np.linalg.pinv(J, rcond=rcond) @ J) @ z
        if step_limit is not None:
            sn = np.linalg.norm(dq)
            if sn > step_limit:
                dq = dq * (step_limit / sn)
        if not np.all(np.isfinite(dq)):
            hist["step"].append(float("nan"))
            break
        hist["step"].append(float(np.linalg.norm(dq)))
        q = q + dq
        hist["q"].append(q.copy())
    return IKResult(q=q, converged=converged, iterations=len(hist["err"]) - 1, error_norm=hist["err"][-1], history=hist)


class MujocoArm:
    """Thin wrapper that gives a MuJoCo serial arm the ``fk`` / ``jac`` / ``error`` interface of :func:`solve_ik`.

    ``tool`` names a site (``kind="site"``) or a body (``kind="body"``; ``offset`` is a point in that body's frame, in
    metres). The first ``n`` generalised coordinates (default: all hinge/slide joints of the model that precede any gripper
    joint, i.e. ``model.nq`` unless ``n`` is given) are the arm joints. The wrapper **writes** ``data.qpos`` whenever it is called.
    """

    def __init__(self, model, data, tool: str, kind: str = "site", n: Optional[int] = None, offset=(0.0, 0.0, 0.0)):
        import mujoco
        self._mj = mujoco
        self.model, self.data, self.kind = model, data, kind
        self.n = model.nq if n is None else n
        self.offset = np.asarray(offset, float)
        if kind == "site":
            self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tool)
        else:
            self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tool)
        if self.id < 0:
            raise KeyError(f"no {kind} named {tool!r}")

    def set_q(self, q):
        self.data.qpos[: self.n] = q
        self._mj.mj_kinematics(self.model, self.data)
        self._mj.mj_comPos(self.model, self.data)

    def pose(self):
        d = self.data
        if self.kind == "site":
            R, p = d.site_xmat[self.id].reshape(3, 3).copy(), d.site_xpos[self.id].copy()
        else:
            R = d.xmat[self.id].reshape(3, 3).copy()
            p = d.xpos[self.id] + R @ self.offset
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return T

    def fk(self, q) -> np.ndarray:
        """4x4 tool pose in the world frame for the joint vector ``q``."""
        self.set_q(q)
        return self.pose()

    def jac(self, q) -> np.ndarray:
        """6 x n Jacobian ``[J_v; J_w]`` of the tool point (world frame): linear rows first, then angular rows."""
        self.set_q(q)
        mj, d = self._mj, self.data
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        if self.kind == "site":
            mj.mj_jacSite(self.model, d, jp, jr, self.id)
        else:
            point = d.xpos[self.id] + d.xmat[self.id].reshape(3, 3) @ self.offset
            mj.mj_jac(self.model, d, jp, jr, point, self.id)
        return np.vstack([jp[:, : self.n], jr[:, : self.n]])

    def position_error(self, p_d) -> Callable:
        """Return ``err(q) = p_d - p(q)`` (3-vector) for position-only IK."""
        p_d = np.asarray(p_d, float)
        return lambda q: p_d - self.fk(q)[:3, 3]

    def pose_error(self, T_d) -> Callable:
        """Return ``err(q) = [p_d - p(q); rotation_error(R_d, R(q))]`` (6-vector) for full-pose IK."""
        T_d = np.asarray(T_d, float)

        def err(q):
            T = self.fk(q)
            return np.concatenate([T_d[:3, 3] - T[:3, 3], rotation_error(T_d[:3, :3], T[:3, :3])])
        return err

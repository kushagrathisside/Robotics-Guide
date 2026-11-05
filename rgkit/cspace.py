"""Configuration-space helpers: angles on circles and tori, SO(3) and nonholonomic motion.

Supports MFoR Chapter 1 (configuration space). Contents:

* :func:`wrap_to_pi`, :func:`angle_diff`, :func:`circle_distance`, :func:`torus_distance`,
  :func:`torus_interpolate` — arithmetic on the circle ``S^1`` and the torus ``T^n`` (angles *wrap around*),
* :func:`rotvec_ball`, :func:`rotation_from_ball` and :func:`canonical_ball` — the ball of radius ``pi``
  with antipodal boundary points identified, a picture of ``SO(3)``,
* :func:`lie_bracket`, :func:`distribution_rank` — a numerical Frobenius test for integrability of velocity
  constraints,
* :func:`unicycle_step`, :func:`unicycle_path`, :func:`parking_manoeuvre` — the kinematic car (a nonholonomic
  system) and the four-step Lie-bracket manoeuvre that moves it sideways.
"""

from __future__ import annotations

import numpy as np

from .mechanisms import numerical_mobility

__all__ = [
    "wrap_to_pi", "angle_diff", "circle_distance", "torus_distance", "torus_interpolate",
    "rotvec_ball", "rotation_from_ball", "canonical_ball", "so3_constraint", "manifold_dimension",
    "lie_bracket", "distribution_rank", "unicycle_field_g1", "unicycle_field_g2", "unicycle_step",
    "unicycle_path", "parking_manoeuvre",
]


# --------------------------------------------------------------------------------------------------------------
# Circle and torus
# --------------------------------------------------------------------------------------------------------------

def wrap_to_pi(a):
    """Map angles to the interval ``(-pi, pi]``."""
    a = np.asarray(a, float)
    return np.pi - np.mod(np.pi - a, 2 * np.pi)


def angle_diff(a, b):
    """Signed shortest rotation from ``a`` to ``b`` on the circle, in ``(-pi, pi]``."""
    return wrap_to_pi(np.asarray(b, float) - np.asarray(a, float))


def circle_distance(a, b):
    """Geodesic (arc-length) distance between angles ``a`` and ``b`` on the unit circle."""
    return np.abs(angle_diff(a, b))


def torus_distance(q1, q2):
    """Geodesic distance between two points of the flat torus ``T^n`` (one angle per coordinate)."""
    return float(np.linalg.norm(angle_diff(q1, q2)))


def torus_interpolate(q1, q2, t):
    """Point at fraction ``t`` of the *shortest* path from ``q1`` to ``q2`` on the torus."""
    q1 = np.asarray(q1, float)
    return wrap_to_pi(q1 + t * angle_diff(q1, q2))


# --------------------------------------------------------------------------------------------------------------
# SO(3) as a ball with identified antipodes
# --------------------------------------------------------------------------------------------------------------

def rotvec_ball(R):
    """Rotation matrix -> rotation vector ``theta * axis`` with ``0 <= theta <= pi`` (a point of the ball)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(R, float)).as_rotvec()


def rotation_from_ball(w):
    """Rotation vector (any length) -> rotation matrix."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.asarray(w, float)).as_matrix()


def canonical_ball(w):
    """Representative of the rotation ``exp([w])`` inside the closed ball ``|w| <= pi``.

    Points ``w`` and ``-w`` with ``|w| = pi`` describe the same rotation (a half turn about ``n`` equals a half
    turn about ``-n``); this is the identification that makes the ball into ``SO(3)``.
    """
    return rotvec_ball(rotation_from_ball(w))


def so3_constraint(Rflat):
    """The six independent equations (upper triangle of ``R^T R - I``) that cut ``SO(3)`` out of ``R^{3x3}``."""
    R = np.asarray(Rflat, float).reshape(3, 3)
    E = R.T @ R - np.eye(3)
    return E[np.triu_indices(3)]


def manifold_dimension(fun, x, tol=1e-6):
    """Dimension of the solution set ``{x : fun(x) = 0}`` near a point ``x`` on it: ``n - rank(D fun)``."""
    return numerical_mobility(fun, x, tol=tol)[0]


# --------------------------------------------------------------------------------------------------------------
# Velocity constraints: Lie brackets and the Frobenius test
# --------------------------------------------------------------------------------------------------------------

def lie_bracket(f, g, x, h=1e-6):
    """Numerical Lie bracket ``[f, g](x) = Dg(x) f(x) - Df(x) g(x)`` of two vector fields ``R^n -> R^n``."""
    x = np.asarray(x, float)

    def jac(F):
        cols = []
        for k in range(len(x)):
            e = np.zeros_like(x); e[k] = h
            cols.append((np.asarray(F(x + e)) - np.asarray(F(x - e))) / (2 * h))
        return np.array(cols).T

    return jac(g) @ np.asarray(f(x)) - jac(f) @ np.asarray(g(x))


def distribution_rank(fields, x, depth=2, tol=1e-7):
    """Rank of the span of ``fields`` and their iterated brackets (up to ``depth`` levels) at ``x``.

    If the rank equals the dimension of the space, the (velocity) constraints that leave exactly these directions
    free are **not integrable**: they do not restrict the reachable configurations (Chow's theorem).
    """
    x = np.asarray(x, float)
    vecs = [np.asarray(f(x), float) for f in fields]
    level = list(fields)
    for _ in range(depth - 1):
        new = []
        for f in fields:
            for g in level:
                new.append(lambda y, f=f, g=g: lie_bracket(f, g, y))
        vecs += [np.asarray(b(x), float) for b in new]
        level = new
    M = np.array(vecs)
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol * max(1.0, s[0])))


# --------------------------------------------------------------------------------------------------------------
# The kinematic car (unicycle)
# --------------------------------------------------------------------------------------------------------------

def unicycle_field_g1(q):
    """Drive field of the unicycle ``q = (x, y, theta)``: ``(cos theta, sin theta, 0)``."""
    return np.array([np.cos(q[2]), np.sin(q[2]), 0.0])


def unicycle_field_g2(q):
    """Steering (turn-in-place) field of the unicycle: ``(0, 0, 1)``."""
    return np.array([0.0, 0.0, 1.0])


def unicycle_step(q, v, w, dt):
    """Exact solution of ``x' = v cos th, y' = v sin th, th' = w`` for constant ``v, w`` over ``dt``."""
    x, y, th = q
    if abs(w) < 1e-12:
        return np.array([x + v * dt * np.cos(th), y + v * dt * np.sin(th), th])
    th2 = th + w * dt
    return np.array([x + v / w * (np.sin(th2) - np.sin(th)), y - v / w * (np.cos(th2) - np.cos(th)), th2])


def unicycle_path(q0, controls, n_sub=20):
    """Follow a list of ``(v, w, duration)`` segments; return an ``(N, 3)`` array of states (sub-sampled)."""
    q = np.asarray(q0, float)
    out = [q]
    for v, w, T in controls:
        for _ in range(n_sub):
            q = unicycle_step(q, v, w, T / n_sub)
            out.append(q)
    return np.array(out)


def parking_manoeuvre(eps, q0=(0.0, 0.0, 0.0)):
    """The four-step Lie-bracket manoeuvre: forward ``eps``, turn ``eps``, back ``eps``, turn back ``eps``.

    Returns the path. To second order in ``eps`` the net displacement is ``eps^2 [g1, g2]``, a sideways step.
    """
    controls = [(1.0, 0.0, eps), (0.0, 1.0, eps), (-1.0, 0.0, eps), (0.0, -1.0, eps)]
    return unicycle_path(q0, controls)

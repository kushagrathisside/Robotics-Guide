"""Small numerical tools for Lyapunov analysis (RMC Chapter 3, notebook 3.1).

* :func:`rk4_trajectory` / :func:`rk4_batch` - fixed-step integration of ``xdot = f(x)`` for one or many initial states;
* :func:`numerical_jacobian` - finite-difference linearisation at an equilibrium;
* :func:`solve_lyapunov` - the Lyapunov matrix equation ``A'P + PA = -Q``;
* :func:`vdot_on_grid`, :func:`level_set_estimate` - evaluate ``Vdot`` on a grid and return the largest sublevel set of ``V``
  on which ``Vdot < 0`` (a *conservative estimate* of the region of attraction);
* :func:`pendulum_field`, :func:`pendulum_energy` - the (damped) pendulum used throughout the notebook.

State vectors are plain NumPy arrays; vector fields take ``x`` of shape ``(n,)`` or, for the batch routines, ``(N, n)``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

__all__ = ["rk4_trajectory", "rk4_batch", "numerical_jacobian", "solve_lyapunov", "is_positive_definite",
           "vdot_on_grid", "level_set_estimate", "pendulum_field", "pendulum_energy", "wrap_angle"]


def rk4_trajectory(f: Callable, x0, t) -> np.ndarray:
    """Integrate ``xdot = f(x)`` with RK4 on the time grid ``t``; return an array of shape ``(len(t), n)``."""
    t = np.asarray(t, float)
    x = np.asarray(x0, float)
    out = np.zeros((len(t), x.size))
    out[0] = x
    for k in range(len(t) - 1):
        h = t[k + 1] - t[k]
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k + 1] = x
    return out


def rk4_batch(f: Callable, X0, T: float, n_steps: int) -> np.ndarray:
    """Integrate many initial states at once; ``f`` must accept an ``(N, n)`` array. Returns the final states."""
    X = np.array(X0, float)
    h = T / n_steps
    for _ in range(n_steps):
        k1 = f(X)
        k2 = f(X + 0.5 * h * k1)
        k3 = f(X + 0.5 * h * k2)
        k4 = f(X + h * k3)
        X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X


def numerical_jacobian(f: Callable, x0, h: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian ``df/dx`` of a vector field at ``x0``."""
    x0 = np.asarray(x0, float)
    cols = []
    for i in range(x0.size):
        e = np.zeros_like(x0)
        e[i] = h
        cols.append((np.asarray(f(x0 + e)) - np.asarray(f(x0 - e))) / (2 * h))
    return np.array(cols).T


def solve_lyapunov(A, Q=None) -> np.ndarray:
    """Solve ``A' P + P A = -Q`` for ``P`` (``Q`` defaults to the identity). ``P > 0`` iff ``A`` is Hurwitz."""
    A = np.asarray(A, float)
    Q = np.eye(len(A)) if Q is None else np.asarray(Q, float)
    return solve_continuous_lyapunov(A.T, -Q)


def is_positive_definite(P, tol: float = 1e-12) -> bool:
    P = np.asarray(P, float)
    return bool(np.allclose(P, P.T) and np.all(np.linalg.eigvalsh((P + P.T) / 2) > tol))


def vdot_on_grid(f2: Callable, V: Callable, grad_V: Callable, X, Y):
    """For a planar system ``(xdot, ydot) = f2(x, y)`` return ``V`` and ``Vdot = grad V . f`` on the grid ``X, Y``."""
    fx, fy = f2(X, Y)
    gx, gy = grad_V(X, Y)
    return V(X, Y), gx * fx + gy * fy


def level_set_estimate(Vgrid, Vdotgrid, X, Y, center=(0.0, 0.0), exclude_radius: float = 0.05) -> float:
    """Largest ``c`` such that ``Vdot < 0`` on ``{V < c}`` (grid estimate, equilibrium neighbourhood excluded).

    ``c = min V`` over grid points with ``Vdot >= 0`` outside a small disc around ``center`` and over the grid
    boundary (so that the sublevel set is contained in the grid). The result is an *inner* estimate of the region of
    attraction of the equilibrium at ``center``.
    """
    far = (X - center[0]) ** 2 + (Y - center[1]) ** 2 > exclude_radius**2
    bad = (Vdotgrid >= 0) & far
    border = np.zeros_like(bad)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    return float(np.min(Vgrid[bad | border]))


def pendulum_field(wn2: float = 1.0, b: float = 0.0) -> Callable:
    """Vector field of ``theta'' = -wn2 sin(theta) - b theta'`` for ``x = (theta, omega)`` (single state or batch)."""
    def f(x):
        x = np.asarray(x, float)
        th, om = x[..., 0], x[..., 1]
        return np.stack([om, -wn2 * np.sin(th) - b * om], axis=-1)
    return f


def pendulum_energy(x, wn2: float = 1.0):
    """``E = omega^2 / 2 + wn2 (1 - cos theta)`` (zero at the hanging equilibrium)."""
    x = np.asarray(x, float)
    return 0.5 * x[..., 1] ** 2 + wn2 * (1 - np.cos(x[..., 0]))


def wrap_angle(a):
    """Wrap an angle (or array) to ``(-pi, pi]``."""
    return np.pi - np.mod(np.pi - np.asarray(a, float), 2 * np.pi)

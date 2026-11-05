"""Optimal and predictive control with numpy/scipy only (RMC Chapter 7).

The module collects the algorithms of the first three notebooks of Chapter 7 so that the notebooks can
concentrate on ideas rather than bookkeeping:

* **Linear systems and LQR** — :func:`linearize`, :func:`c2d`, :func:`lqr` (continuous time), :func:`dlqr`
  (discrete time), the finite-horizon Riccati recursion :func:`riccati_recursion`, and
  :func:`policy_cost_matrix` (cost of an arbitrary linear feedback).
* **Trajectory optimisation** — :func:`ilqr` (iterative LQR with Levenberg-Marquardt regularisation, a
  backtracking line search and box constraints on the input), with quadratic or user-defined costs.
* **Model predictive control** — two small quadratic-programming solvers (:func:`qp_box_pgd`, projected
  accelerated gradient for box constraints, and :func:`qp_admm`, an OSQP-style splitting method for
  ``l <= A x <= u``) and :class:`LinearMPC`, a condensed receding-horizon controller with input and state
  constraints.

Conventions: states ``x`` in R^n, inputs ``u`` in R^m, a discrete model ``x+ = f(x, u)`` or
``x+ = A x + B u``; costs are ``sum_k [ (x_k-x_g)'Q(x_k-x_g) + u_k'R u_k ] + (x_N-x_g)'Q_f(x_N-x_g)`` (with the
convention used in the notebooks that the factor 1/2 is included when a cost is *evaluated* by
:class:`QuadraticCost`). Everything is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import linalg as sla

__all__ = [
    "numerical_jacobian", "linearize", "c2d", "rk4_step", "rollout",
    "lqr", "dlqr", "riccati_recursion", "policy_cost_matrix", "closed_loop_poles", "controllability_matrix",
    "QuadraticCost", "FunctionCost", "ILQRResult", "ilqr", "finite_difference_jacobians",
    "qp_box_pgd", "qp_admm", "QPResult", "LinearMPC", "MPCSolution",
]


# ----------------------------------------------------------------------------------------------------------
# Linearisation and discretisation
# ----------------------------------------------------------------------------------------------------------
def numerical_jacobian(f: Callable[[np.ndarray], np.ndarray], x, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of a vector function ``f`` at ``x`` (shape ``(len(f(x)), len(x))``)."""
    x = np.asarray(x, float)
    cols = []
    for i in range(x.size):
        h = eps * max(1.0, abs(x[i]))
        e = np.zeros_like(x)
        e[i] = h
        cols.append((np.asarray(f(x + e), float) - np.asarray(f(x - e), float)) / (2.0 * h))
    return np.stack(cols, axis=-1)


def linearize(f: Callable[[np.ndarray, np.ndarray], np.ndarray], x0, u0, eps: float = 1e-6):
    """Jacobians ``A = df/dx`` and ``B = df/du`` of ``f(x, u)`` at ``(x0, u0)`` by central differences.

    ``f`` may be a continuous-time right-hand side ``xdot = f(x, u)`` or a discrete map ``x+ = f(x, u)``.
    """
    x0, u0 = np.asarray(x0, float), np.asarray(u0, float)
    A = numerical_jacobian(lambda x: f(x, u0), x0, eps)
    B = numerical_jacobian(lambda u: f(x0, u), u0, eps)
    return A, B


def c2d(A, B, dt: float):
    """Zero-order-hold discretisation ``x+ = Ad x + Bd u`` of ``xdot = A x + B u`` (matrix exponential)."""
    A, B = np.atleast_2d(A).astype(float), np.atleast_2d(B).astype(float)
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = sla.expm(M * dt)
    return E[:n, :n], E[:n, n:]


def rk4_step(f: Callable[[np.ndarray, np.ndarray], np.ndarray], x, u, dt: float) -> np.ndarray:
    """One classical Runge-Kutta step of ``xdot = f(x, u)`` with the input held constant over the step."""
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(f: Callable[[np.ndarray, np.ndarray], np.ndarray], x0, U) -> np.ndarray:
    """States ``X[0..T]`` produced by the input sequence ``U[0..T-1]`` from ``x0`` under ``x+ = f(x, u)``."""
    X = [np.asarray(x0, float)]
    for u in U:
        X.append(np.asarray(f(X[-1], u), float))
    return np.array(X)


def controllability_matrix(A, B) -> np.ndarray:
    """``[B, AB, ..., A^(n-1) B]``."""
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    n = A.shape[0]
    blocks = [B]
    for _ in range(n - 1):
        blocks.append(A @ blocks[-1])
    return np.hstack(blocks)


# ----------------------------------------------------------------------------------------------------------
# LQR
# ----------------------------------------------------------------------------------------------------------
def lqr(A, B, Q, R, N=None):
    """Continuous-time LQR: minimise ``int (x'Qx + u'Ru + 2x'Nu) dt`` for ``xdot = Ax + Bu``.

    Returns ``(K, P)`` with ``u = -K x`` and cost-to-go ``x'Px``; ``P`` solves the algebraic Riccati equation
    ``A'P + PA - (PB + N) R^-1 (B'P + N') + Q = 0``.
    """
    A, B, Q, R = (np.atleast_2d(np.asarray(M, float)) for M in (A, B, Q, R))
    Nn = None if N is None else np.atleast_2d(np.asarray(N, float))
    P = sla.solve_continuous_are(A, B, Q, R, s=Nn)
    K = np.linalg.solve(R, B.T @ P + (0 if Nn is None else Nn.T))
    return K, P


def dlqr(A, B, Q, R, N=None):
    """Discrete-time LQR: minimise ``sum x'Qx + u'Ru + 2x'Nu`` for ``x+ = Ax + Bu``; returns ``(K, P)``."""
    A, B, Q, R = (np.atleast_2d(np.asarray(M, float)) for M in (A, B, Q, R))
    Nn = None if N is None else np.atleast_2d(np.asarray(N, float))
    P = sla.solve_discrete_are(A, B, Q, R, s=Nn)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A + (0 if Nn is None else Nn.T))
    return K, P


def riccati_recursion(A, B, Q, R, Qf, horizon: int):
    """Finite-horizon discrete LQR by the backward Riccati recursion (dynamic programming).

    ``A, B, Q, R`` may be constant matrices or lists of length ``horizon`` (time-varying).
    Returns ``(K, P)``: ``K[k]`` is the gain of ``u_k = -K[k] x_k`` and ``P[k]`` the cost-to-go matrix, ``k = 0..N``
    (``P[N] = Qf``).
    """
    def at(M, k):
        return M[k] if isinstance(M, (list, tuple)) or (isinstance(M, np.ndarray) and M.ndim == 3) else M

    P = [None] * (horizon + 1)
    K = [None] * horizon
    P[horizon] = np.atleast_2d(np.asarray(Qf, float))
    for k in range(horizon - 1, -1, -1):
        Ak, Bk, Qk, Rk = (np.atleast_2d(np.asarray(at(M, k), float)) for M in (A, B, Q, R))
        S = Rk + Bk.T @ P[k + 1] @ Bk
        K[k] = np.linalg.solve(S, Bk.T @ P[k + 1] @ Ak)
        # P_k = Q + A'P A - A'P B (R + B'P B)^-1 B'P A, symmetrised against round-off
        P[k] = Qk + Ak.T @ P[k + 1] @ Ak - Ak.T @ P[k + 1] @ Bk @ K[k]
        P[k] = 0.5 * (P[k] + P[k].T)
    return K, P


def policy_cost_matrix(A, B, K, Q, R) -> np.ndarray:
    """Matrix ``P_K`` with ``sum_k x_k'Q x_k + u_k'R u_k = x_0' P_K x_0`` for ``u = -K x`` (discrete time).

    It solves the Lyapunov equation ``P = (Q + K'RK) + (A-BK)' P (A-BK)``; the series diverges (and an error is
    raised) if ``A - BK`` is not Schur stable.
    """
    A, B, K, Q, R = (np.atleast_2d(np.asarray(M, float)) for M in (A, B, K, Q, R))
    Acl = A - B @ K
    if np.max(np.abs(np.linalg.eigvals(Acl))) >= 1.0:
        raise ValueError("closed loop is not Schur stable; the infinite-horizon cost is unbounded")
    return sla.solve_discrete_lyapunov(Acl.T, Q + K.T @ R @ K)


def closed_loop_poles(A, B, K) -> np.ndarray:
    """Eigenvalues of ``A - B K``."""
    return np.linalg.eigvals(np.atleast_2d(A) - np.atleast_2d(B) @ np.atleast_2d(K))


# ----------------------------------------------------------------------------------------------------------
# Costs for trajectory optimisation
# ----------------------------------------------------------------------------------------------------------
class QuadraticCost:
    """``l(x,u) = 1/2 (x-xg)'Q(x-xg) + 1/2 (u-ug)'R(u-ug)`` and ``l_f(x) = 1/2 (x-xg)'Qf(x-xg)``.

    ``Q``, ``R`` and ``Qf`` are matrices (or 1-D arrays interpreted as diagonals).
    """

    def __init__(self, Q, R, Qf, x_goal, u_goal=None):
        self.Q, self.R, self.Qf = (np.diag(M) if np.ndim(M) == 1 else np.atleast_2d(M) for M in (Q, R, Qf))
        self.xg = np.asarray(x_goal, float)
        self.ug = np.zeros(self.R.shape[0]) if u_goal is None else np.asarray(u_goal, float)

    def running(self, x, u, t=0) -> float:
        dx, du = x - self.xg, u - self.ug
        return 0.5 * dx @ self.Q @ dx + 0.5 * du @ self.R @ du

    def final(self, x) -> float:
        dx = x - self.xg
        return 0.5 * dx @ self.Qf @ dx

    def derivs(self, x, u, t=0):
        dx, du = x - self.xg, u - self.ug
        return self.Q @ dx, self.R @ du, self.Q, self.R, np.zeros((self.R.shape[0], self.Q.shape[0]))

    def final_derivs(self, x):
        return self.Qf @ (x - self.xg), self.Qf


class FunctionCost:
    """Cost defined by Python callables ``l(x, u, t)`` and ``lf(x)``; derivatives by central differences.

    Use it for costs that are not quadratic in the state, for example a task-space error through forward
    kinematics. For speed give analytic derivatives with :class:`QuadraticCost` where possible.
    """

    def __init__(self, l, lf, eps: float = 1e-4):
        self.l, self.lf, self.eps = l, lf, eps

    def running(self, x, u, t=0):
        return float(self.l(x, u, t))

    def final(self, x):
        return float(self.lf(x))

    @staticmethod
    def _grad_hess(f, z, eps):
        n = z.size
        g = np.zeros(n)
        H = np.zeros((n, n))
        f0 = f(z)
        fp, fm = np.zeros(n), np.zeros(n)
        for i in range(n):
            e = np.zeros(n)
            e[i] = eps
            fp[i], fm[i] = f(z + e), f(z - e)
            g[i] = (fp[i] - fm[i]) / (2 * eps)
            H[i, i] = (fp[i] - 2 * f0 + fm[i]) / eps ** 2
        for i in range(n):
            for j in range(i + 1, n):
                ei, ej = np.zeros(n), np.zeros(n)
                ei[i], ej[j] = eps, eps
                H[i, j] = H[j, i] = (f(z + ei + ej) - fp[i] - fp[j] + 2 * f0 - fm[i] - fm[j]
                                     + f(z - ei - ej)) / (2 * eps ** 2)
        return g, H

    def derivs(self, x, u, t=0):
        n, m = x.size, u.size
        g, H = self._grad_hess(lambda z: self.l(z[:n], z[n:], t), np.concatenate([x, u]), self.eps)
        return g[:n], g[n:], H[:n, :n], H[n:, n:], H[n:, :n]

    def final_derivs(self, x):
        g, H = self._grad_hess(lambda z: self.lf(z), np.asarray(x, float), self.eps)
        return g, H


# ----------------------------------------------------------------------------------------------------------
# iLQR
# ----------------------------------------------------------------------------------------------------------
def finite_difference_jacobians(f, X, U, eps: float = 1e-6):
    """Stacks ``A_t = df/dx`` and ``B_t = df/du`` of ``x+ = f(x, u)`` along a trajectory (central differences)."""
    T = len(U)
    n, m = X.shape[1], U.shape[1]
    A, B = np.zeros((T, n, n)), np.zeros((T, n, m))
    for t in range(T):
        A[t], B[t] = linearize(f, X[t], U[t], eps)
    return A, B


@dataclass
class ILQRResult:
    X: np.ndarray                     # (T+1, n) optimised states
    U: np.ndarray                     # (T, m) optimised inputs
    K: np.ndarray                     # (T, m, n) feedback gains of the last backward pass: u = U + k + K (x - X)
    k: np.ndarray                     # (T, m) feed-forward corrections of the last backward pass
    costs: list = field(default_factory=list)   # cost after each accepted iteration (first entry = initial guess)
    converged: bool = False
    n_iter: int = 0
    mu: float = 0.0


def ilqr(f, cost, x0, U_init, *, jac=None, u_min=None, u_max=None, max_iter: int = 100, tol: float = 1e-7,
         mu_init: float = 1e-6, mu_max: float = 1e10, alphas=None, verbose: bool = False,
         fd_eps: float = 1e-6) -> ILQRResult:
    """Iterative LQR (Li & Todorov 2004; the regularised variant of Tassa, Mansard & Todorov 2012).

    Parameters
    ----------
    f : ``x+ = f(x, u)`` (discrete dynamics, e.g. an RK4 step or one MuJoCo control step).
    cost : an object with ``running``, ``final``, ``derivs`` and ``final_derivs`` (see :class:`QuadraticCost`).
    x0, U_init : initial state and initial input sequence of shape ``(T, m)``.
    jac : optional ``jac(x, u) -> (A, B)``; central differences are used if omitted.
    u_min, u_max : optional box limits on the input; the forward pass clips, and the backward pass freezes the
        components that are at a limit and would be pushed further outside (the "clamped" variant of box-DDP).
    """
    x0 = np.asarray(x0, float)
    U = np.array(U_init, float)
    T, m = U.shape
    n = x0.size
    lo = -np.inf * np.ones(m) if u_min is None else np.broadcast_to(np.asarray(u_min, float), (m,)).copy()
    hi = np.inf * np.ones(m) if u_max is None else np.broadcast_to(np.asarray(u_max, float), (m,)).copy()
    alphas = [1.0, 0.5, 0.25, 0.1, 0.03, 0.01, 0.003, 0.001] if alphas is None else list(alphas)
    U = np.clip(U, lo, hi)

    def total_cost(X, U_):
        return sum(cost.running(X[t], U_[t], t) for t in range(T)) + cost.final(X[T])

    X = rollout(f, x0, U)
    J = total_cost(X, U)
    costs = [J]
    mu, delta = mu_init, 2.0
    K_, k_ = np.zeros((T, m, n)), np.zeros((T, m))
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        # --- derivatives along the current trajectory
        if jac is None:
            A, B = finite_difference_jacobians(f, X, U, fd_eps)
        else:
            AB = [jac(X[t], U[t]) for t in range(T)]
            A, B = np.array([a for a, _ in AB]), np.array([b for _, b in AB])
        derivs = [cost.derivs(X[t], U[t], t) for t in range(T)]
        lxT, lxxT = cost.final_derivs(X[T])

        # --- backward pass (retry with larger regularisation if Quu is not positive definite)
        while True:
            Vx, Vxx = lxT.copy(), lxxT.copy()
            K = np.zeros((T, m, n))
            k = np.zeros((T, m))
            dV1 = dV2 = 0.0
            ok = True
            for t in range(T - 1, -1, -1):
                lx, lu, lxx, luu, lux = derivs[t]
                At, Bt = A[t], B[t]
                Vxx_r = Vxx + mu * np.eye(n)
                Qx = lx + At.T @ Vx
                Qu = lu + Bt.T @ Vx
                Qxx = lxx + At.T @ Vxx @ At
                Quu = luu + Bt.T @ Vxx_r @ Bt
                Qux = lux + Bt.T @ Vxx_r @ At
                Quu = 0.5 * (Quu + Quu.T)
                try:
                    np.linalg.cholesky(Quu)
                except np.linalg.LinAlgError:
                    ok = False
                    break
                free = np.ones(m, bool)
                if np.isfinite(lo).any() or np.isfinite(hi).any():
                    # components at a bound whose unconstrained step would leave the box are frozen
                    kfull = -np.linalg.solve(Quu, Qu)
                    free = ~(((U[t] <= lo + 1e-12) & (kfull < 0)) | ((U[t] >= hi - 1e-12) & (kfull > 0)))
                kt = np.zeros(m)
                Kt = np.zeros((m, n))
                if free.any():
                    Qff = Quu[np.ix_(free, free)]
                    kt[free] = -np.linalg.solve(Qff, Qu[free])
                    Kt[free] = -np.linalg.solve(Qff, Qux[free])
                K[t], k[t] = Kt, kt
                dV1 += kt @ Qu
                dV2 += 0.5 * kt @ Quu @ kt
                Vx = Qx + Kt.T @ Quu @ kt + Kt.T @ Qu + Qux.T @ kt
                Vxx = Qxx + Kt.T @ Quu @ Kt + Kt.T @ Qux + Qux.T @ Kt
                Vxx = 0.5 * (Vxx + Vxx.T)
            if ok:
                break
            delta = max(delta * 2.0, 2.0)
            mu = max(mu * delta, mu_init)
            if mu > mu_max:
                break
        if not ok:
            break

        # --- forward pass with backtracking line search
        accepted = False
        for alpha in alphas:
            Xn = np.zeros_like(X)
            Un = np.zeros_like(U)
            Xn[0] = x0
            for t in range(T):
                Un[t] = np.clip(U[t] + alpha * k[t] + K[t] @ (Xn[t] - X[t]), lo, hi)
                Xn[t + 1] = f(Xn[t], Un[t])
            Jn = total_cost(Xn, Un)
            expected = -(alpha * dV1 + alpha ** 2 * dV2)
            ratio = (J - Jn) / expected if expected > 1e-16 else (1.0 if Jn < J else 0.0)
            if np.isfinite(Jn) and Jn < J and ratio > 1e-4:
                accepted = True
                break
        if accepted:
            dJ = J - Jn
            X, U, J = Xn, Un, Jn
            K_, k_ = K, k
            costs.append(J)
            delta = min(delta / 2.0, 1.0 / 2.0)
            mu = mu * delta if mu * delta > 1e-9 else 0.0
            if verbose:
                print(f"iter {it:3d}  cost {J:.6g}  dJ {dJ:.2e}  mu {mu:.1e}  alpha {alpha}")
            if dJ < tol * max(1.0, abs(J)):
                converged = True
                break
        else:
            delta = max(delta * 2.0, 2.0)
            mu = max(mu * delta, mu_init)
            if verbose:
                print(f"iter {it:3d}  line search failed, mu -> {mu:.1e}")
            if mu > mu_max:
                break
    return ILQRResult(X=X, U=U, K=K_, k=k_, costs=costs, converged=converged, n_iter=it, mu=mu)


# ----------------------------------------------------------------------------------------------------------
# Quadratic programming
# ----------------------------------------------------------------------------------------------------------
@dataclass
class QPResult:
    x: np.ndarray
    iters: int
    converged: bool
    y: Optional[np.ndarray] = None       # dual variables (ADMM)
    z: Optional[np.ndarray] = None
    primal_res: float = 0.0
    dual_res: float = 0.0


def qp_box_pgd(H, g, lb, ub, x0=None, max_iter: int = 2000, tol: float = 1e-10) -> QPResult:
    """Minimise ``1/2 x'Hx + g'x`` subject to ``lb <= x <= ub`` by accelerated projected gradient (FISTA).

    ``H`` must be symmetric positive semidefinite. The step is ``1 / lambda_max(H)``; the iteration stops when
    the projected-gradient step is smaller than ``tol``.
    """
    H, g = np.atleast_2d(np.asarray(H, float)), np.asarray(g, float)
    lb = np.broadcast_to(np.asarray(lb, float), g.shape)
    ub = np.broadcast_to(np.asarray(ub, float), g.shape)
    L = float(np.linalg.eigvalsh(H)[-1])
    step = 1.0 / max(L, 1e-12)
    x = np.clip(np.zeros_like(g) if x0 is None else np.asarray(x0, float), lb, ub)
    yv, t = x.copy(), 1.0
    for i in range(1, max_iter + 1):
        x_new = np.clip(yv - step * (H @ yv + g), lb, ub)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        yv = x_new + (t - 1.0) / t_new * (x_new - x)
        done = np.linalg.norm(x_new - x, np.inf) < tol
        x, t = x_new, t_new
        if done:
            return QPResult(x=x, iters=i, converged=True)
    return QPResult(x=x, iters=max_iter, converged=False)


def qp_admm(P, q, A, l, u, *, rho: float = 1.0, sigma: float = 1e-6, alpha: float = 1.6, max_iter: int = 4000,
            eps_abs: float = 1e-8, eps_rel: float = 1e-8, warm=None) -> QPResult:
    """Minimise ``1/2 x'Px + q'x`` subject to ``l <= A x <= u`` by ADMM (the OSQP algorithm, Stellato et al. 2020).

    Rows with ``l == u`` (equalities) get a 1000-times larger penalty. ``warm = (x, z, y)`` restarts from the
    primal and dual variables of a previous solution (as returned in :class:`QPResult`).
    """
    P, A = np.atleast_2d(np.asarray(P, float)), np.atleast_2d(np.asarray(A, float))
    q, l, u = (np.asarray(v, float) for v in (q, l, u))
    n, m = q.size, l.size
    rho_vec = np.full(m, rho)
    rho_vec[np.isclose(l, u)] *= 1e3
    KKT = P + sigma * np.eye(n) + A.T @ (rho_vec[:, None] * A)
    cf = sla.cho_factor(KKT)
    if warm is None:
        x, z, y = np.zeros(n), np.clip(np.zeros(m), l, u), np.zeros(m)
    else:
        x, z, y = (np.array(v, float) for v in warm)
    pri = dua = np.inf
    for i in range(1, max_iter + 1):
        rhs = sigma * x - q + A.T @ (rho_vec * z - y)
        xt = sla.cho_solve(cf, rhs)
        zt = A @ xt
        x_new = alpha * xt + (1 - alpha) * x
        zr = alpha * zt + (1 - alpha) * z
        z_new = np.clip(zr + y / rho_vec, l, u)
        y = y + rho_vec * (zr - z_new)
        x, z = x_new, z_new
        if i % 5 == 0 or i == max_iter:
            Ax = A @ x
            pri = np.linalg.norm(Ax - z, np.inf)
            dua = np.linalg.norm(P @ x + q + A.T @ y, np.inf)
            tp = eps_abs + eps_rel * max(np.linalg.norm(Ax, np.inf), np.linalg.norm(z, np.inf))
            td = eps_abs + eps_rel * max(np.linalg.norm(P @ x, np.inf), np.linalg.norm(A.T @ y, np.inf),
                                         np.linalg.norm(q, np.inf))
            if pri <= tp and dua <= td:
                return QPResult(x=x, iters=i, converged=True, y=y, z=z, primal_res=pri, dual_res=dua)
    return QPResult(x=x, iters=max_iter, converged=False, y=y, z=z, primal_res=pri, dual_res=dua)


# ----------------------------------------------------------------------------------------------------------
# Linear MPC
# ----------------------------------------------------------------------------------------------------------
@dataclass
class MPCSolution:
    U: np.ndarray            # (N, m) optimal input sequence
    X: np.ndarray            # (N+1, n) predicted states, X[0] = x0
    cost: float
    qp: QPResult


class LinearMPC:
    """Condensed linear MPC for ``x+ = A x + B u`` with box constraints on inputs and states.

    The finite-horizon problem solved at every sampling instant is

        min  sum_{k=0}^{N-1} [ 1/2 (x_k - r)'Q(x_k - r) + 1/2 u_k'R u_k ] + 1/2 (x_N - r)'Qf(x_N - r)
        s.t. x_{k+1} = A x_k + B u_k,  umin <= u_k <= umax,  xmin <= x_k <= xmax  (k = 1..N).

    With the states eliminated, ``X = Phi x_0 + Gamma U`` and the problem is a QP in ``U`` whose Hessian
    ``H = Gamma' Qbar Gamma + Rbar`` is constant, so it is built and factorised once. ``Qf`` defaults to the
    solution ``P`` of the discrete Riccati equation, which makes the unconstrained controller identical to LQR.
    """

    def __init__(self, A, B, Q, R, N: int, Qf=None, umin=None, umax=None, xmin=None, xmax=None, solver: str = "admm"):
        self.A, self.B = np.atleast_2d(np.asarray(A, float)), np.atleast_2d(np.asarray(B, float))
        self.n, self.m = self.B.shape
        self.Q, self.R = np.atleast_2d(np.asarray(Q, float)), np.atleast_2d(np.asarray(R, float))
        self.N = N
        self.Qf = sla.solve_discrete_are(self.A, self.B, self.Q, self.R) if Qf is None else np.atleast_2d(Qf)
        n, m = self.n, self.m
        self.Phi = np.zeros((N * n, n))
        self.Gam = np.zeros((N * n, N * m))
        Ak = np.eye(n)
        for k in range(N):
            Ak = self.A @ Ak
            self.Phi[k * n:(k + 1) * n] = Ak
            for j in range(k + 1):
                self.Gam[k * n:(k + 1) * n, j * m:(j + 1) * m] = np.linalg.matrix_power(self.A, k - j) @ self.B
        self.Qbar = sla.block_diag(*([self.Q] * (N - 1) + [self.Qf]))
        self.Rbar = sla.block_diag(*([self.R] * N))
        self.H = self.Gam.T @ self.Qbar @ self.Gam + self.Rbar
        self.H = 0.5 * (self.H + self.H.T)
        self.umin = np.full(m, -np.inf) if umin is None else np.broadcast_to(np.asarray(umin, float), (m,)).copy()
        self.umax = np.full(m, np.inf) if umax is None else np.broadcast_to(np.asarray(umax, float), (m,)).copy()
        self.xmin = np.full(n, -np.inf) if xmin is None else np.broadcast_to(np.asarray(xmin, float), (n,)).copy()
        self.xmax = np.full(n, np.inf) if xmax is None else np.broadcast_to(np.asarray(xmax, float), (n,)).copy()
        self.solver = solver
        # constraint matrix [I; Gamma_c] for ADMM, rows with no finite bound removed
        xfin = np.tile(np.isfinite(self.xmin) | np.isfinite(self.xmax), N)
        self._xrows = np.where(xfin)[0]
        self._A_qp = np.vstack([np.eye(N * m), self.Gam[self._xrows]])
        self._warm = None

    # -- helpers ---------------------------------------------------------------------------------------
    def _bounds(self, x0):
        N = self.N
        lb = np.tile(self.umin, N)
        ub = np.tile(self.umax, N)
        xl = np.tile(self.xmin, N)[self._xrows] - (self.Phi @ x0)[self._xrows]
        xu = np.tile(self.xmax, N)[self._xrows] - (self.Phi @ x0)[self._xrows]
        return lb, ub, xl, xu

    def solve(self, x0, ref=None, warm: bool = True, **kw) -> MPCSolution:
        """Solve the QP for the current state ``x0`` (and constant state reference ``ref``)."""
        x0 = np.asarray(x0, float)
        r = np.zeros(self.n) if ref is None else np.asarray(ref, float)
        Xr = np.tile(r, self.N)
        g = self.Gam.T @ self.Qbar @ (self.Phi @ x0 - Xr)
        lb, ub, xl, xu = self._bounds(x0)
        if self.solver == "pgd":
            if self._xrows.size:
                raise ValueError("the projected-gradient solver handles input bounds only")
            res = qp_box_pgd(self.H, g, lb, ub, x0=None if self._warm is None else self._warm.x, **kw)
        else:
            warm_start = None
            if warm and self._warm is not None and self._warm.y is not None:
                warm_start = (self._warm.x, self._warm.z, self._warm.y)
            res = qp_admm(self.H, g, self._A_qp, np.concatenate([lb, xl]), np.concatenate([ub, xu]),
                          warm=warm_start, **kw)
        if warm:
            self._warm = res
        U = res.x.reshape(self.N, self.m)
        X = np.vstack([x0, (self.Phi @ x0 + self.Gam @ res.x).reshape(self.N, self.n)])
        e = X[1:].ravel() - Xr
        cost = 0.5 * e @ self.Qbar @ e + 0.5 * res.x @ self.Rbar @ res.x
        return MPCSolution(U=U, X=X, cost=float(cost), qp=res)

    def control(self, x0, ref=None) -> np.ndarray:
        """First input of the optimal sequence (the receding-horizon law ``u = kappa(x)``)."""
        return self.solve(x0, ref).U[0]

    def reset(self):
        """Forget the warm start."""
        self._warm = None

    def simulate(self, x0, steps: int, ref=None, plant=None, disturbance=None):
        """Closed-loop simulation. ``plant(x, u, k)`` defaults to the prediction model.

        Returns ``(X, U, info)`` with ``info['plans']`` the predicted state sequence at every step and
        ``info['iters']`` the QP iteration counts.
        """
        x = np.asarray(x0, float)
        X, U, plans, iters = [x.copy()], [], [], []
        for k in range(steps):
            sol = self.solve(x, ref)
            u = sol.U[0]
            plans.append(sol.X)
            iters.append(sol.qp.iters)
            x = self.A @ x + self.B @ u if plant is None else plant(x, u, k)
            if disturbance is not None:
                x = x + disturbance(k)
            X.append(x.copy())
            U.append(u.copy())
        return np.array(X), np.array(U), dict(plans=plans, iters=iters)

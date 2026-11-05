import numpy as np
import pytest
from scipy import linalg as sla
from scipy.optimize import minimize

from rgkit import optimal_control as oc


# ---------------------------------------------------------------- helpers
def pendulum_f(x, u, g=9.81, l=1.0, m=1.0, b=0.1):
    return np.array([x[1], (u[0] - m * g * l * np.sin(x[0]) - b * x[1]) / (m * l * l)])


def double_integrator(dt=0.1):
    return oc.c2d(np.array([[0, 1.0], [0, 0]]), np.array([[0.0], [1.0]]), dt)


# ---------------------------------------------------------------- linearisation, discretisation
def test_numerical_jacobian_matches_analytic():
    f = lambda x: np.array([np.sin(x[0]) * x[1], x[0] ** 2 + 3 * x[1]])
    x = np.array([0.7, -1.3])
    J = oc.numerical_jacobian(f, x)
    assert np.allclose(J, [[np.cos(0.7) * -1.3, np.sin(0.7)], [2 * 0.7, 3]], atol=1e-8)


def test_linearize_pendulum_upright():
    A, B = oc.linearize(pendulum_f, [np.pi, 0.0], [0.0])
    assert np.allclose(A, [[0, 1], [9.81, -0.1]], atol=1e-6)
    assert np.allclose(B, [[0], [1]], atol=1e-8)


def test_c2d_double_integrator():
    Ad, Bd = double_integrator(0.1)
    assert np.allclose(Ad, [[1, 0.1], [0, 1]])
    assert np.allclose(Bd, [[0.005], [0.1]])


def test_rk4_exponential():
    f = lambda x, u: -x
    x = np.array([1.0])
    for _ in range(100):
        x = oc.rk4_step(f, x, None, 0.01)
    assert abs(x[0] - np.exp(-1.0)) < 1e-9


# ---------------------------------------------------------------- LQR
def test_scalar_dlqr_closed_form():
    a, b, q, r = 1.2, 1.0, 1.0, 1.0
    K, P = oc.dlqr([[a]], [[b]], [[q]], [[r]])
    # scalar DARE:  p = q + a^2 p - a^2 b^2 p^2 / (r + b^2 p)
    p = P[0, 0]
    assert np.isclose(p, q + a * a * p - (a * b * p) ** 2 / (r + b * b * p))
    assert abs(a - b * K[0, 0]) < 1


def test_continuous_are_residual_and_stability():
    A = np.array([[0, 1.0], [9.81, -0.1]])
    B = np.array([[0.0], [1.0]])
    Q, R = np.diag([10.0, 1.0]), np.array([[0.1]])
    K, P = oc.lqr(A, B, Q, R)
    res = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
    assert np.allclose(res, 0, atol=1e-9)
    assert np.all(np.linalg.eigvals(A - B @ K).real < 0)
    assert np.allclose(P, P.T)


def test_riccati_recursion_converges_to_dare():
    Ad, Bd = double_integrator(0.1)
    Q, R = np.diag([1.0, 0.1]), np.array([[0.05]])
    Kinf, Pinf = oc.dlqr(Ad, Bd, Q, R)
    K, P = oc.riccati_recursion(Ad, Bd, Q, R, Q, 400)
    assert np.allclose(K[0], Kinf, atol=1e-9)
    assert np.allclose(P[0], Pinf, atol=1e-8)
    assert np.allclose(K[-1], np.linalg.solve(R + Bd.T @ Q @ Bd, Bd.T @ Q @ Ad))   # last step uses Qf


def test_finite_horizon_dp_equals_batch_least_squares():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3)) * 0.6
    B = rng.normal(size=(3, 2))
    Q, R, Qf = np.eye(3), 0.5 * np.eye(2), 3 * np.eye(3)
    N, x0 = 6, np.array([1.0, -2.0, 0.5])
    K, P = oc.riccati_recursion(A, B, Q, R, Qf, N)
    x, J = x0.copy(), 0.0
    for k in range(N):
        u = -K[k] @ x
        J += 0.5 * (x @ Q @ x + u @ R @ u)
        x = A @ x + B @ u
    J += 0.5 * x @ Qf @ x
    assert np.isclose(J, 0.5 * x0 @ P[0] @ x0)

    def cost(U):
        U = U.reshape(N, 2)
        x, c = x0.copy(), 0.0
        for k in range(N):
            c += 0.5 * (x @ Q @ x + U[k] @ R @ U[k])
            x = A @ x + B @ U[k]
        return c + 0.5 * x @ Qf @ x
    r = minimize(cost, np.zeros(2 * N), method="BFGS", options=dict(gtol=1e-10))
    assert abs(r.fun - J) < 1e-8


def test_time_varying_recursion_accepts_lists():
    Ad, Bd = double_integrator(0.1)
    K, P = oc.riccati_recursion([Ad] * 5, [Bd] * 5, [np.eye(2)] * 5, [np.eye(1)] * 5, np.eye(2), 5)
    K2, P2 = oc.riccati_recursion(Ad, Bd, np.eye(2), np.eye(1), np.eye(2), 5)
    assert all(np.allclose(a, b) for a, b in zip(K, K2))


def test_policy_cost_matrix_equals_riccati_for_optimal_gain():
    Ad, Bd = double_integrator(0.1)
    Q, R = np.diag([1.0, 0.1]), np.array([[0.05]])
    K, P = oc.dlqr(Ad, Bd, Q, R)
    assert np.allclose(oc.policy_cost_matrix(Ad, Bd, K, Q, R), P, atol=1e-8)
    worse = oc.policy_cost_matrix(Ad, Bd, 1.3 * K, Q, R)
    assert np.all(np.linalg.eigvalsh(worse - P) > -1e-9)          # any other stabilising gain costs more
    with pytest.raises(ValueError):
        oc.policy_cost_matrix(Ad, Bd, np.zeros((1, 2)), Q, R)


def test_lqr_return_difference_inequality():
    A = np.array([[0, 1.0], [9.81, -0.1]])
    B = np.array([[0.0], [1.0]])
    K, P = oc.lqr(A, B, np.diag([10.0, 1.0]), np.array([[1.0]]))
    for w in np.logspace(-2, 2, 60):
        L = K @ np.linalg.solve(1j * w * np.eye(2) - A, B)
        assert abs(1 + L[0, 0]) >= 1 - 1e-9


def test_controllability_matrix_rank():
    A = np.array([[0, 1.0], [0, 0]])
    assert np.linalg.matrix_rank(oc.controllability_matrix(A, np.array([[0.0], [1.0]]))) == 2
    assert np.linalg.matrix_rank(oc.controllability_matrix(A, np.array([[1.0], [0.0]]))) == 1


# ---------------------------------------------------------------- QP solvers
def random_qp(rng, n=8):
    M = rng.normal(size=(n, n))
    H = M @ M.T + 0.1 * np.eye(n)
    g = rng.normal(size=n) * 3
    return H, g


def test_box_pgd_matches_scipy():
    rng = np.random.default_rng(1)
    H, g = random_qp(rng)
    lb, ub = -0.5 * np.ones(8), 0.5 * np.ones(8)
    res = oc.qp_box_pgd(H, g, lb, ub, max_iter=20000)
    ref = minimize(lambda x: 0.5 * x @ H @ x + g @ x, np.zeros(8), jac=lambda x: H @ x + g, bounds=list(zip(lb, ub)),
                   method="L-BFGS-B", options=dict(ftol=1e-15, gtol=1e-12))
    assert res.converged
    assert np.allclose(res.x, ref.x, atol=1e-5)
    assert np.all(res.x >= lb - 1e-12) and np.all(res.x <= ub + 1e-12)


def test_admm_matches_scipy_general_constraints():
    rng = np.random.default_rng(2)
    H, g = random_qp(rng, 6)
    A = rng.normal(size=(4, 6))
    l, u = -np.ones(4), np.ones(4)
    res = oc.qp_admm(H, g, A, l, u)
    cons = [dict(type="ineq", fun=lambda x: A @ x - l), dict(type="ineq", fun=lambda x: u - A @ x)]
    ref = minimize(lambda x: 0.5 * x @ H @ x + g @ x, np.zeros(6), jac=lambda x: H @ x + g, constraints=cons,
                   method="SLSQP", options=dict(ftol=1e-14, maxiter=500))
    assert res.converged
    assert abs((0.5 * res.x @ H @ res.x + g @ res.x) - ref.fun) < 1e-6
    assert np.all(A @ res.x <= u + 1e-6) and np.all(A @ res.x >= l - 1e-6)


def test_admm_equality_rows():
    H, g = np.eye(2), np.array([-1.0, -1.0])
    A = np.array([[1.0, 1.0]])
    res = oc.qp_admm(H, g, A, [0.5], [0.5])
    assert np.allclose(res.x, [0.25, 0.25], atol=1e-6)


# ---------------------------------------------------------------- MPC
def test_unconstrained_mpc_equals_lqr():
    Ad, Bd = double_integrator(0.1)
    Q, R = np.diag([1.0, 0.1]), np.array([[0.05]])
    K, _ = oc.dlqr(Ad, Bd, Q, R)
    mpc = oc.LinearMPC(Ad, Bd, Q, R, N=10)            # terminal cost = DARE solution => exactly LQR
    for x0 in ([1.0, 0.0], [-2.0, 1.5]):
        u = mpc.control(np.array(x0))
        assert np.allclose(u, -K @ np.array(x0), atol=1e-6)


def test_mpc_respects_constraints_and_is_recursively_feasible():
    Ad, Bd = double_integrator(0.1)
    Q, R = np.diag([10.0, 1.0]), np.array([[0.1]])
    mpc = oc.LinearMPC(Ad, Bd, Q, R, N=25, umin=-1.0, umax=1.0, xmin=[-5.0, -0.8], xmax=[5.0, 0.8])
    X, U, info = mpc.simulate([-2.0, 0.0], 80)
    assert np.all(np.abs(U) <= 1 + 1e-6)
    assert np.all(np.abs(X[:, 1]) <= 0.8 + 1e-5)
    assert np.linalg.norm(X[-1]) < 1e-2                # reaches the origin
    assert max(info["iters"]) < 4000


def test_mpc_pgd_agrees_with_admm_on_box_problem():
    Ad, Bd = double_integrator(0.1)
    Q, R = np.diag([10.0, 1.0]), np.array([[0.1]])
    a = oc.LinearMPC(Ad, Bd, Q, R, N=15, umin=-0.5, umax=0.5, solver="admm")
    p = oc.LinearMPC(Ad, Bd, Q, R, N=15, umin=-0.5, umax=0.5, solver="pgd")
    x0 = np.array([-3.0, 0.0])
    sa, sp = a.solve(x0), p.solve(x0, max_iter=20000)
    assert np.allclose(sa.U, sp.U, atol=1e-4)


# ---------------------------------------------------------------- iLQR
def test_ilqr_on_lq_problem_reproduces_riccati():
    Ad, Bd = double_integrator(0.1)
    f = lambda x, u: Ad @ x + Bd @ u
    Q, R, Qf = np.diag([1.0, 0.1]), np.array([[0.05]]), np.diag([2.0, 0.2])
    T = 30
    cost = oc.QuadraticCost(Q, R, Qf, [0, 0])
    x0 = np.array([1.0, -0.5])
    res = oc.ilqr(f, cost, x0, np.zeros((T, 1)), max_iter=10)
    K, P = oc.riccati_recursion(Ad, Bd, Q, R, Qf, T)
    assert np.allclose(res.K[0], -K[0], atol=1e-6)          # iLQR gain convention: u = U + k + K (x - X)
    assert np.isclose(res.costs[-1], 0.5 * x0 @ P[0] @ x0, rtol=1e-8)


def test_ilqr_pendulum_swingup_with_box_constraint():
    dt, T = 0.05, 100
    f = lambda x, u: oc.rk4_step(pendulum_f, x, u, dt)
    cost = oc.QuadraticCost(np.diag([0.0, 0.0]) + 1e-3, 0.01 * np.eye(1), np.diag([500.0, 50.0]), [np.pi, 0.0])
    # warm start with a few energy-pumping swings so that the local method starts in the right basin
    U0 = 2.5 * np.sin(2.0 * np.pi * np.arange(T) * dt / 1.9).reshape(T, 1)
    res = oc.ilqr(f, cost, [0.0, 0.0], U0, u_min=-3.0, u_max=3.0, max_iter=150)
    assert np.all(np.abs(res.U) <= 3.0 + 1e-12)
    assert res.costs[-1] < res.costs[0]
    assert abs(res.X[-1, 0] - np.pi) < 0.1 and abs(res.X[-1, 1]) < 0.5
    assert all(b <= a + 1e-9 for a, b in zip(res.costs, res.costs[1:]))      # monotone decrease


def test_function_cost_matches_quadratic_cost_derivatives():
    Q, R, Qf = np.diag([1.0, 2.0]), np.array([[0.3]]), np.diag([4.0, 5.0])
    xg = np.array([0.5, -0.2])
    qc = oc.QuadraticCost(Q, R, Qf, xg)
    fc = oc.FunctionCost(lambda x, u, t: 0.5 * (x - xg) @ Q @ (x - xg) + 0.5 * u @ R @ u,
                         lambda x: 0.5 * (x - xg) @ Qf @ (x - xg))
    x, u = np.array([0.1, 0.4]), np.array([0.7])
    for a, b in zip(qc.derivs(x, u), fc.derivs(x, u)):
        assert np.allclose(a, b, atol=1e-5)
    for a, b in zip(qc.final_derivs(x), fc.final_derivs(x)):
        assert np.allclose(a, b, atol=1e-5)

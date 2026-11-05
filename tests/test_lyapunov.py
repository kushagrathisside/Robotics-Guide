import numpy as np

from rgkit import lyapunov as ly


def test_lyapunov_equation_solution_and_definiteness():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    P = ly.solve_lyapunov(A, np.eye(2))
    assert np.allclose(A.T @ P + P @ A, -np.eye(2))
    assert ly.is_positive_definite(P)
    assert np.allclose(P, [[1.25, 0.25], [0.25, 0.25]])          # worked out by hand
    # unstable A: P is not positive definite
    assert not ly.is_positive_definite(ly.solve_lyapunov(np.array([[1.0, 0.0], [0.0, -3.0]]), np.eye(2)))


def test_numerical_jacobian_of_pendulum():
    f = ly.pendulum_field(wn2=4.0, b=0.5)
    assert np.allclose(ly.numerical_jacobian(f, [0, 0]), [[0, 1], [-4, -0.5]], atol=1e-7)
    assert np.allclose(ly.numerical_jacobian(f, [np.pi, 0]), [[0, 1], [4, -0.5]], atol=1e-7)


def test_energy_is_conserved_without_damping_and_decays_with_damping():
    t = np.linspace(0, 20, 4001)
    x0 = [2.0, 0.0]
    E0 = ly.pendulum_energy(np.array(x0), 1.0)
    traj = ly.rk4_trajectory(ly.pendulum_field(1.0, 0.0), x0, t)
    assert np.ptp(ly.pendulum_energy(traj)) < 1e-9
    trajd = ly.rk4_trajectory(ly.pendulum_field(1.0, 0.4), x0, t)
    E = ly.pendulum_energy(trajd)
    assert np.all(np.diff(E) <= 1e-12) and E[-1] < 1e-3 * E0
    # Vdot = -b omega^2
    Edot = np.gradient(E, t)
    assert np.allclose(Edot[5:-5], -0.4 * trajd[5:-5, 1] ** 2, atol=1e-5)


def test_batch_integration_matches_single_trajectory():
    f = ly.pendulum_field(1.0, 0.3)
    X0 = np.array([[1.0, 0.0], [-2.0, 1.0]])
    t = np.linspace(0, 5, 501)
    ends = ly.rk4_batch(f, X0, 5.0, 500)
    for x0, xe in zip(X0, ends):
        assert np.allclose(ly.rk4_trajectory(f, x0, t)[-1], xe, atol=1e-10)


def test_region_of_attraction_estimate_is_conservative():
    # xdot = -x + x^3 has equilibria at +-1; V = x^2 gives c* = 1 (|x| < 1), the exact region of attraction.
    # Use a planar version: xdot = -x + x^3, ydot = -y.
    X, Y = np.meshgrid(np.linspace(-2, 2, 401), np.linspace(-2, 2, 401))
    f2 = lambda x, y: (-x + x**3, -y)
    V = lambda x, y: x**2 + y**2
    gV = lambda x, y: (2 * x, 2 * y)
    Vg, Vd = ly.vdot_on_grid(f2, V, gV, X, Y)
    c = ly.level_set_estimate(Vg, Vd, X, Y)
    assert 0.95 < c <= 1.01          # the disc of radius 1 touches the equilibrium (1, 0)


def test_wrap_angle():
    assert np.allclose(ly.wrap_angle([0.0, 3 * np.pi, -3 * np.pi + 0.1, 7.0]), [0.0, np.pi, 0.1 - np.pi, 7.0 - 2 * np.pi])

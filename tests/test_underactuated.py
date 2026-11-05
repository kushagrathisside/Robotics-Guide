"""Tests for rgkit.underactuated: dynamics against independent sympy derivations, structure, controllers."""
import numpy as np
import pytest
import sympy as sp

from rgkit import underactuated as ua

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- dynamics vs the sympy Euler-Lagrange derivation
def _subs(sym, values):
    return {s: float(v) for s, v in zip(sym, values)}


def test_cartpole_matches_sympy_derivation():
    p = ua.CartPoleParams(mc=1.3, mp=0.4, l=0.7, g=9.81)
    sy = ua.cartpole_symbolic()
    mc, mp, l, g = sy["params"]
    for _ in range(5):
        q, qd = RNG.normal(size=2), RNG.normal(size=2)
        sub = {mc: p.mc, mp: p.mp, l: p.l, g: p.g, **_subs(sy["q"], q), **_subs(sy["qd"], qd)}
        assert np.allclose(np.array(sy["M"].subs(sub), float), ua.cartpole_M(q, p))
        assert np.allclose(np.array(sy["h"].subs(sub), float).ravel(), ua.cartpole_h(q, qd, p))
        # closed-form accelerations solve M qdd + h = B u
        u = RNG.normal()
        s = np.r_[q, qd]
        qdd = ua.cartpole_deriv(s, u, p)[2:]
        assert np.allclose(ua.cartpole_M(q, p) @ qdd + ua.cartpole_h(q, qd, p), [u, 0.0])


def test_acrobot_matches_sympy_derivation():
    p = ua.AcrobotParams(m1=1.1, m2=0.7, l1=0.9, l2=1.2, lc1=0.4, lc2=0.6, I1=0.05, I2=0.08)
    sy = ua.acrobot_symbolic()
    m1, m2, l1, lc1, lc2, I1, I2, g = sy["params"]
    for _ in range(5):
        q, qd = RNG.normal(size=2), RNG.normal(size=2)
        sub = {m1: p.m1, m2: p.m2, l1: p.l1, lc1: p.lc1, lc2: p.lc2, I1: p.I1, I2: p.I2, g: p.g,
               **_subs(sy["q"], q), **_subs(sy["qd"], qd)}
        assert np.allclose(np.array(sy["M"].subs(sub), float), ua.acrobot_M(q, p))
        assert np.allclose(np.array(sy["h"].subs(sub), float).ravel(), ua.acrobot_h(q, qd, p))
        u = RNG.normal()
        qdd = ua.acrobot_deriv(np.r_[q, qd], u, p)[2:]
        assert np.allclose(ua.acrobot_M(q, p) @ qdd + ua.acrobot_h(q, qd, p), [0.0, u])


def test_pendulum_matches_sympy_and_christoffel_skew_symmetry():
    p = ua.PendulumParams(m=0.8, l=0.6, b=0.0, g=9.81)
    sy = ua.pendulum_symbolic()
    m, l, g = sy["params"]
    th = 0.7
    sub = {m: p.m, l: p.l, g: p.g, sy["q"][0]: th, sy["qd"][0]: 0.0}
    assert np.isclose(float(sy["h"][0].subs(sub)), p.m * p.g * p.l * np.sin(th))
    # Mdot - 2C is skew-symmetric for the cart-pole and the acrobot (property of the Christoffel choice)
    for sy2 in (ua.cartpole_symbolic(), ua.acrobot_symbolic()):
        q, qd = sy2["q"], sy2["qd"]
        Mdot = sum((sp.diff(sy2["M"], q[k]) * qd[k] for k in range(2)), sp.zeros(2, 2))
        S = (Mdot - 2 * sy2["C"]).applyfunc(sp.simplify)
        assert sp.simplify(S + S.T) == sp.zeros(2, 2)
        # and h = C qd - tau_g
        assert sp.simplify(sy2["h"] - (sy2["C"] * qd - sy2["tau_g"])) == sp.zeros(2, 1)


# ---------------------------------------------------------------- energy, equilibria, linearisation
def test_energy_is_conserved_without_input():
    pp, cp, ap = ua.PendulumParams(), ua.CartPoleParams(), ua.AcrobotParams()
    for deriv, energy, x0 in [
        (lambda x, u: ua.pendulum_deriv(x, u, pp), lambda x: ua.pendulum_energy(x, pp), [2.0, 1.0]),
        (lambda x, u: ua.cartpole_deriv(x, u, cp), lambda x: ua.cartpole_energy(x, cp), [0.1, 1.0, 0.3, -0.5]),
        (lambda x, u: ua.acrobot_deriv(x, u, ap), lambda x: ua.acrobot_energy(x, ap), [0.5, 0.3, 0.1, -0.2]),
    ]:
        t, X, _ = ua.simulate(deriv, lambda t, x: 0.0, np.array(x0), 0.002, 3.0)
        E = np.array([energy(x) for x in X])
        assert np.ptp(E) < 1e-8


def test_equilibria_and_linearisations():
    cp, ap, pp = ua.CartPoleParams(), ua.AcrobotParams(), ua.PendulumParams(b=0.1)
    assert np.allclose(ua.cartpole_deriv([0, np.pi, 0, 0], 0.0, cp), 0, atol=1e-12)
    assert np.allclose(ua.acrobot_deriv([np.pi, 0, 0, 0], 0.0, ap), 0, atol=1e-12)
    assert np.allclose(ua.pendulum_deriv([np.pi, 0], 0.0, pp), 0, atol=1e-12)
    A, B = ua.cartpole_linearized(cp)
    An, Bn = ua.linearize(lambda x, u: ua.cartpole_deriv(x, u, cp), [0, np.pi, 0, 0])
    assert np.allclose(A, An, atol=1e-5) and np.allclose(B, Bn, atol=1e-5)
    A, B = ua.pendulum_linearized(pp)
    An, Bn = ua.linearize(lambda x, u: ua.pendulum_deriv(x, u, pp), [np.pi, 0])
    assert np.allclose(A, An, atol=1e-5) and np.allclose(B, Bn, atol=1e-5)
    # cart-pole poles: 0, 0, +/- sqrt((mc+mp) g / (mc l))
    A, B = ua.cartpole_linearized(cp)
    lam = np.sqrt((cp.mc + cp.mp) * cp.g / (cp.mc * cp.l))
    assert np.allclose(sorted(np.linalg.eigvals(A).real), [-lam, 0, 0, lam], atol=1e-6)


def test_controllability_of_linearisations():
    cp, ap = ua.CartPoleParams(), ua.AcrobotParams()
    A, B = ua.cartpole_linearized(cp)
    assert np.linalg.matrix_rank(ua.controllability_matrix(A, B)) == 4
    # sympy: det [B AB A^2B A^3B] = -g^2 / (mc^4 l^4), independent of mp and never zero
    assert np.isclose(np.linalg.det(ua.controllability_matrix(A, B)), -cp.g ** 2 / (cp.mc ** 4 * cp.l ** 4))
    A, B = ua.acrobot_linearized(ap)
    assert np.linalg.matrix_rank(ua.controllability_matrix(A, B)) == 4


def test_lqr_stabilises_linear_systems():
    cp = ua.CartPoleParams()
    A, B = ua.cartpole_linearized(cp)
    K, P = ua.lqr(A, B, np.diag([1, 10, 1, 1]), [[0.1]])
    assert np.all(np.linalg.eigvals(A - B @ K).real < 0)
    # Riccati equation residual
    R = np.array([[0.1]])
    res = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + np.diag([1, 10, 1, 1])
    assert np.abs(res).max() < 1e-8


# ---------------------------------------------------------------- partial feedback linearisation
@pytest.mark.parametrize("system", ["cartpole", "acrobot"])
def test_pfl_gives_requested_accelerations(system):
    if system == "cartpole":
        p, act, Mf, hf, df = ua.CartPoleParams(), 0, ua.cartpole_M, ua.cartpole_h, ua.cartpole_deriv
    else:
        p, act, Mf, hf, df = ua.AcrobotParams(), 1, ua.acrobot_M, ua.acrobot_h, ua.acrobot_deriv
    for _ in range(5):
        q, qd = RNG.normal(size=2), RNG.normal(size=2)
        M, h = Mf(q, p), hf(q, qd, p)
        v = RNG.normal()
        u = ua.pfl_collocated(M, h, v, act)
        assert np.isclose(df(np.r_[q, qd], u, p)[2 + act], v)
        u = ua.pfl_noncollocated(M, h, v, act)
        assert np.isclose(df(np.r_[q, qd], u, p)[2 + (1 - act)], v)


# ---------------------------------------------------------------- controllers
def test_pendulum_swingup_with_torque_limit_then_lqr():
    p = ua.PendulumParams(m=1.0, l=0.5, b=0.0)
    A, B = ua.pendulum_linearized(p)
    K, P = ua.lqr(A, B, np.diag([10, 1]), [[1.0]])
    umax = 0.4 * p.m * p.g * p.l                    # cannot hold the pendulum horizontal
    ctrl = ua.hybrid_controller(lambda x: float(ua.pendulum_energy_shaping(x, p, 1.0, umax)), K, P,
                                np.array([np.pi, 0.0]), 1.0, umax, angle_idx=(0,))
    t, X, U = ua.simulate(lambda x, u: ua.pendulum_deriv(x, u, p), ctrl, np.array([0.05, 0.0]), 0.002, 12.0)
    assert abs(ua.wrap_pi(X[-1, 0] - np.pi)) < 1e-6 and abs(X[-1, 1]) < 1e-6
    assert np.abs(U).max() <= umax + 1e-12


def test_cartpole_swingup_and_balance():
    p = ua.CartPoleParams()
    A, B = ua.cartpole_linearized(p)
    K, P = ua.lqr(A, B, np.diag([1, 10, 1, 1]), [[0.1]])
    ctrl = ua.hybrid_controller(lambda s: ua.cartpole_swingup(s, p, 1.0, 1.0, 1.0, 40.0), K, P,
                                np.array([0, np.pi, 0, 0]), 5.0, 40.0, angle_idx=(1,))
    t, X, U = ua.simulate(lambda x, u: ua.cartpole_deriv(x, u, p), ctrl, np.array([0, 0.1, 0, 0]), 0.002, 10.0)
    assert abs(ua.wrap_pi(X[-1, 1] - np.pi)) < 1e-3 and np.abs(X[-1, [0, 2, 3]]).max() < 1e-2


def test_acrobot_swingup_and_balance():
    p = ua.AcrobotParams()
    A, B = ua.acrobot_linearized(p)
    K, P = ua.lqr(A, B, np.diag([10, 10, 1, 1]), [[1.0]])
    Ed = ua.acrobot_energy(np.array([np.pi, 0, 0, 0]), p)

    def swing(s):
        q, qd = s[:2], s[2:]
        gate = np.clip((Ed - ua.acrobot_energy(s, p)) / 3.0, -1, 1)
        q2d = 2 * 1.5 / np.pi * np.arctan(5.0 * qd[0]) * gate
        v = 50.0 * (q2d - q[1]) - 10.0 * qd[1]
        return float(np.clip(ua.pfl_collocated(ua.acrobot_M(q, p), ua.acrobot_h(q, qd, p), v, 1), -25, 25))

    ctrl = ua.hybrid_controller(swing, K, P, np.array([np.pi, 0, 0, 0]), 2.0, 25.0, angle_idx=(0, 1))
    t, X, U = ua.simulate(lambda x, u: ua.acrobot_deriv(x, u, p), ctrl, np.array([0.3, 0, 0, 0]), 0.004, 30.0)
    e = X[-1] - np.array([np.pi, 0, 0, 0])
    e[:2] = ua.wrap_pi(e[:2])
    assert np.abs(e).max() < 1e-3


def test_batch_simulation_matches_single():
    p = ua.CartPoleParams()
    x0 = RNG.normal(size=(4, 3))
    ctrl = lambda t, x: 0.5 * np.sin(3 * t) * np.ones(x.shape[1:])  # noqa: E731
    _, Xb, _ = ua.simulate(lambda x, u: ua.cartpole_deriv(x, u, p), ctrl, x0, 0.01, 1.0)
    for j in range(3):
        _, Xs, _ = ua.simulate(lambda x, u: ua.cartpole_deriv(x, u, p), ctrl, x0[:, j], 0.01, 1.0)
        assert np.allclose(Xb[:, :, j], Xs)


def test_lqr_basin_fraction_decreases_with_level():
    p = ua.PendulumParams()
    A, B = ua.pendulum_linearized(p)
    K, P = ua.lqr(A, B, np.diag([10, 1]), [[1.0]])
    f = ua.lqr_basin_fraction(lambda x, u: ua.pendulum_deriv(x, u, p), K, P, np.array([np.pi, 0.0]), 1.96,
                              [0.1, 1.0, 50.0], n=60, T=6.0, angle_idx=(0,))
    assert f[0] == 1.0 and f[2] < 1.0           # small level sets are inside the basin, large ones are not

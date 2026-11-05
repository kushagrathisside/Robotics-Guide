import numpy as np

from rgkit import rl_numpy as rl

LAYOUT = ["S..#G",
          ".#...",
          "...H."]


def test_transition_tensor_is_stochastic_and_terminals_absorb():
    env = rl.GridWorld(LAYOUT, slip=0.2)
    assert np.allclose(env.P.sum(2), 1.0)
    for s in np.flatnonzero(env.terminal):
        assert np.all(env.P[s, :, s] == 1.0) and np.all(env.R[s] == 0.0)


def test_value_iteration_satisfies_bellman_optimality_and_matches_policy_iteration():
    env = rl.GridWorld(LAYOUT, slip=0.2)
    gamma = 0.95
    V, Q, pol, deltas = rl.value_iteration(env.P, env.R, gamma)
    assert np.allclose(V, (env.R + gamma * np.einsum("sat,t->sa", env.P, V)).max(1), atol=1e-8)
    V2, pol2, n = rl.policy_iteration(env.P, env.R, gamma)
    assert np.allclose(V, V2, atol=1e-8)
    assert np.allclose(rl.policy_evaluation(env.P, env.R, pol, gamma), V, atol=1e-8)
    assert deltas[-1] < 1e-10 and all(a >= b - 1e-12 for a, b in zip(deltas, deltas[1:]))   # contraction


def test_deterministic_grid_shortest_path():
    env = rl.GridWorld(["S.G"], slip=0.0, step_reward=-1.0, goal_reward=10.0)
    V, Q, pol, _ = rl.value_iteration(env.P, env.R, 1.0)
    s0 = env.start[0]
    assert np.isclose(V[s0], -1 + 10.0)                       # one step costs -1, then the goal pays 10
    assert pol[s0] == 1                                       # move right


def test_q_learning_converges_to_value_iteration():
    env = rl.GridWorld(["S..", ".#.", "..G"], slip=0.1)
    gamma = 0.9
    V, Q, pol, _ = rl.value_iteration(env.P, env.R, gamma)
    res = rl.q_learning(env, 8000, alpha=lambda ep: max(0.01, 0.3 / (1 + ep / 200)), gamma=gamma, epsilon=0.3, seed=0,
                        Q_ref=Q)
    assert res.q_error[-1] < 0.5 and res.q_error[-1] < res.q_error[10]
    ns = ~env.terminal
    assert (rl.greedy_policy(res.Q)[ns] == pol[ns]).mean() > 0.85


def test_sarsa_runs_and_improves():
    env = rl.GridWorld(["S..", ".#.", "..G"], slip=0.1)
    res = rl.sarsa(env, 2000, alpha=0.1, gamma=0.9, epsilon=0.1, seed=1)
    assert res.returns[-200:].mean() > res.returns[:200].mean()


def test_mlp_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    net = rl.MLP([3, 5, 4, 2], rng)
    X, T = rng.normal(size=(7, 3)), rng.normal(size=(7, 2))
    loss = lambda th: (net.set_flat(th), 0.5 * ((net.forward(X, keep=False) - T) ** 2).sum())[1]
    th = net.get_flat()
    net.set_flat(th)
    Y = net.forward(X)
    g = net.backward(Y - T)
    num = np.array([(loss(th + 1e-6 * e) - loss(th - 1e-6 * e)) / 2e-6 for e in np.eye(th.size)])
    assert np.allclose(g, num, atol=1e-6)
    assert net.n_params == th.size


def test_adam_minimises_quadratic():
    opt, x = rl.Adam(2, lr=0.1), np.array([3.0, -2.0])
    for _ in range(500):
        x = opt.step(x, 2 * x)
    assert np.linalg.norm(x) < 1e-2


def test_cem_and_es_maximise_a_concave_function():
    target = np.array([1.0, -2.0, 0.5])
    f = lambda th, seed: -float(np.sum((th - target) ** 2))
    th, hist = rl.cem(f, np.zeros(3), 1.0, 60, pop=40, seed=0, extra_std=0.5)
    assert np.linalg.norm(th - target) < 0.1 and hist[-1, 0] > hist[0, 0]
    th, hist = rl.es(f, np.zeros(3), sigma=0.1, lr=0.05, iters=200, pairs=8, seed=0)
    assert np.linalg.norm(th - target) < 0.2

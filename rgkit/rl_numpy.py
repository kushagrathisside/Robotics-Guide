"""Reinforcement-learning building blocks in plain numpy (RMC Chapter 7).

PyTorch, Gymnasium and Stable-Baselines3 are deliberately not used: every algorithm of the notebooks is a few
lines of numpy, so that the reader can read, change and debug all of it.

* :class:`GridWorld` — a small finite MDP (walls, a goal, hazards, "slippery" moves) with its transition tensor.
* Dynamic programming — :func:`policy_evaluation`, :func:`value_iteration`, :func:`policy_iteration`.
* Tabular temporal-difference learning — :func:`q_learning`, :func:`sarsa`.
* :class:`MLP` and :class:`Adam` — a fully connected network with hand-written back-propagation.
* Gradient-free policy search — :func:`cem` (cross-entropy method) and :func:`es` (antithetic evolution
  strategy with rank shaping), which need only episode returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = ["GridWorld", "policy_evaluation", "value_iteration", "policy_iteration", "greedy_policy",
           "q_learning", "sarsa", "MLP", "Adam", "cem", "es", "ACTION_NAMES"]

ACTION_NAMES = ["up", "right", "down", "left"]
_MOVES = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])


class GridWorld:
    """A rectangular grid MDP.

    ``layout`` is a list of equal-length strings: ``#`` wall, ``.`` free cell, ``S`` start, ``G`` goal
    (terminal, reward ``goal_reward``) and ``H`` hazard (terminal, reward ``hazard_reward``). Every other move
    costs ``step_reward``. Actions are 0 up, 1 right, 2 down, 3 left; with probability ``slip`` the robot
    slips and moves in one of the two perpendicular directions (``slip / 2`` each). Moving into a wall or out
    of the grid leaves the state unchanged.
    """

    def __init__(self, layout: Sequence[str], slip: float = 0.0, step_reward: float = -1.0, goal_reward: float = 20.0,
                 hazard_reward: float = -50.0):
        self.layout = [str(r) for r in layout]
        self.nrow, self.ncol = len(self.layout), len(self.layout[0])
        assert all(len(r) == self.ncol for r in self.layout)
        self.cells = [(r, c) for r in range(self.nrow) for c in range(self.ncol) if self.layout[r][c] != "#"]
        self.index = {rc: i for i, rc in enumerate(self.cells)}
        self.nS, self.nA = len(self.cells), 4
        self.slip, self.step_reward = slip, step_reward
        self.goal_reward, self.hazard_reward = goal_reward, hazard_reward
        self.terminal = np.array([self.layout[r][c] in "GH" for r, c in self.cells])
        self.start = [self.index[(r, c)] for r, c in self.cells if self.layout[r][c] == "S"]
        self.P, self.R = self._build()

    def _target(self, s: int, a: int) -> int:
        r, c = self.cells[s]
        rr, cc = r + _MOVES[a][0], c + _MOVES[a][1]
        if 0 <= rr < self.nrow and 0 <= cc < self.ncol and self.layout[rr][cc] != "#":
            return self.index[(rr, cc)]
        return s

    def _reward_of(self, s2: int) -> float:
        ch = self.layout[self.cells[s2][0]][self.cells[s2][1]]
        return self.goal_reward if ch == "G" else self.hazard_reward if ch == "H" else self.step_reward

    def _build(self):
        P = np.zeros((self.nS, self.nA, self.nS))
        R = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            if self.terminal[s]:
                P[s, :, s] = 1.0
                continue
            for a in range(self.nA):
                outcomes = [(a, 1.0 - self.slip), ((a + 1) % 4, self.slip / 2), ((a - 1) % 4, self.slip / 2)]
                for act, p in outcomes:
                    if p <= 0:
                        continue
                    s2 = self._target(s, act)
                    P[s, a, s2] += p
                    R[s, a] += p * self._reward_of(s2)
        return P, R

    # sampling interface used by the model-free algorithms
    def reset(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.start))

    def step(self, s: int, a: int, rng: np.random.Generator):
        s2 = int(rng.choice(self.nS, p=self.P[s, a]))
        return s2, self._reward_of(s2), bool(self.terminal[s2])

    def to_grid(self, v: np.ndarray, fill=np.nan) -> np.ndarray:
        """Scatter a per-state vector onto the ``nrow x ncol`` grid (walls get ``fill``)."""
        g = np.full((self.nrow, self.ncol), fill, float)
        for i, (r, c) in enumerate(self.cells):
            g[r, c] = v[i]
        return g


# --------------------------------------------------------------------------------------------- dynamic programming
def policy_evaluation(P, R, policy, gamma: float) -> np.ndarray:
    """Exact ``V^pi`` from the Bellman equation ``V = R_pi + gamma P_pi V`` (a linear system).

    ``policy`` is a deterministic action per state (1-D integer array) or a stochastic ``(S, A)`` array.
    """
    nS, nA = R.shape
    pol = np.asarray(policy)
    if pol.ndim == 1:
        pol = np.eye(nA)[pol]
    P_pi = np.einsum("sa,sat->st", pol, P)
    R_pi = (pol * R).sum(1)
    return np.linalg.solve(np.eye(nS) - gamma * P_pi, R_pi)


def greedy_policy(Q: np.ndarray) -> np.ndarray:
    return np.argmax(Q, axis=1)


def value_iteration(P, R, gamma: float, tol: float = 1e-10, max_iter: int = 10000):
    """Value iteration. Returns ``(V, Q, policy, deltas)`` with ``deltas[k] = max |V_{k+1} - V_k|``."""
    V = np.zeros(R.shape[0])
    deltas = []
    for _ in range(max_iter):
        Q = R + gamma * np.einsum("sat,t->sa", P, V)
        V_new = Q.max(1)
        deltas.append(float(np.abs(V_new - V).max()))
        V = V_new
        if deltas[-1] < tol:
            break
    Q = R + gamma * np.einsum("sat,t->sa", P, V)
    return V, Q, greedy_policy(Q), deltas


def policy_iteration(P, R, gamma: float, max_iter: int = 200):
    """Policy iteration. Returns ``(V, policy, n_iterations)``."""
    nS, nA = R.shape
    pol = np.zeros(nS, int)
    for it in range(1, max_iter + 1):
        V = policy_evaluation(P, R, pol, gamma)
        Q = R + gamma * np.einsum("sat,t->sa", P, V)
        new = greedy_policy(Q)
        # keep the old action on ties so that the iteration terminates
        keep = np.isclose(Q[np.arange(nS), pol], Q.max(1))
        new = np.where(keep, pol, new)
        if np.array_equal(new, pol):
            return V, pol, it
        pol = new
    return V, pol, max_iter


# ----------------------------------------------------------------------------------------- temporal-difference learning
@dataclass
class TDResult:
    Q: np.ndarray
    returns: np.ndarray                  # undiscounted return of each training episode
    q_error: np.ndarray = field(default_factory=lambda: np.zeros(0))   # max |Q - Q_ref| after each episode (if Q_ref)


def _eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))        # random tie-break


def q_learning(env, episodes: int, alpha: float = 0.2, gamma: float = 0.95, epsilon=0.1, seed: int = 0,
               max_steps: int = 200, Q_ref=None, Q0=None) -> TDResult:
    """Off-policy tabular Q-learning, ``Q(s,a) += alpha [r + gamma max_a' Q(s',a') - Q(s,a)]``.

    ``alpha`` and ``epsilon`` are numbers or functions of the episode index. ``Q_ref`` (e.g. from value iteration) is used only
    to record the error curve.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA)) if Q0 is None else Q0.copy()
    rets, errs = np.zeros(episodes), np.zeros(episodes if Q_ref is not None else 0)
    for ep in range(episodes):
        eps = epsilon(ep) if callable(epsilon) else epsilon
        lr = alpha(ep) if callable(alpha) else alpha
        s, G = env.reset(rng), 0.0
        for _ in range(max_steps):
            a = _eps_greedy(Q, s, eps, rng)
            s2, r, done = env.step(s, a, rng)
            target = r + (0.0 if done else gamma * Q[s2].max())
            Q[s, a] += lr * (target - Q[s, a])
            G += r
            s = s2
            if done:
                break
        rets[ep] = G
        if Q_ref is not None:
            errs[ep] = np.abs(Q - Q_ref)[~env.terminal].max()
    return TDResult(Q, rets, errs)


def sarsa(env, episodes: int, alpha: float = 0.2, gamma: float = 0.95, epsilon=0.1, seed: int = 0,
          max_steps: int = 200, Q_ref=None) -> TDResult:
    """On-policy SARSA, ``Q(s,a) += alpha [r + gamma Q(s',a') - Q(s,a)]`` with ``a'`` drawn epsilon-greedily."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))
    rets, errs = np.zeros(episodes), np.zeros(episodes if Q_ref is not None else 0)
    for ep in range(episodes):
        eps = epsilon(ep) if callable(epsilon) else epsilon
        lr = alpha(ep) if callable(alpha) else alpha
        s = env.reset(rng)
        a = _eps_greedy(Q, s, eps, rng)
        G = 0.0
        for _ in range(max_steps):
            s2, r, done = env.step(s, a, rng)
            a2 = _eps_greedy(Q, s2, eps, rng)
            target = r + (0.0 if done else gamma * Q[s2, a2])
            Q[s, a] += lr * (target - Q[s, a])
            G += r
            s, a = s2, a2
            if done:
                break
        rets[ep] = G
        if Q_ref is not None:
            errs[ep] = np.abs(Q - Q_ref)[~env.terminal].max()
    return TDResult(Q, rets, errs)


# ------------------------------------------------------------------------------------------------- neural network
class MLP:
    """Fully connected network ``x -> tanh -> ... -> linear`` with manual back-propagation.

    ``sizes = [n_in, h1, ..., n_out]``. ``forward(X)`` takes a batch ``(B, n_in)``; ``backward(dY)`` takes the
    gradient of a scalar loss with respect to the output and returns the gradient with respect to the flat
    parameter vector (same layout as :meth:`get_flat`).
    """

    def __init__(self, sizes: Sequence[int], rng: np.random.Generator | None = None, out_scale: float = 1.0):
        rng = np.random.default_rng(0) if rng is None else rng
        self.sizes = list(sizes)
        self.W, self.b = [], []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(1.0 / a) * (out_scale if i == len(sizes) - 2 else 1.0)
            self.W.append(rng.normal(size=(a, b)) * scale)
            self.b.append(np.zeros(b))
        self._cache = None

    @property
    def n_params(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def get_flat(self) -> np.ndarray:
        return np.concatenate([np.concatenate([w.ravel(), b]) for w, b in zip(self.W, self.b)])

    def set_flat(self, v: np.ndarray):
        i = 0
        for k in range(len(self.W)):
            n = self.W[k].size
            self.W[k] = v[i:i + n].reshape(self.W[k].shape).copy()
            i += n
            m = self.b[k].size
            self.b[k] = v[i:i + m].copy()
            i += m

    def forward(self, X: np.ndarray, keep: bool = True) -> np.ndarray:
        h = np.atleast_2d(X)
        acts = [h]
        for k, (w, b) in enumerate(zip(self.W, self.b)):
            h = h @ w + b
            if k < len(self.W) - 1:
                h = np.tanh(h)
            acts.append(h)
        if keep:
            self._cache = acts
        return acts[-1]

    __call__ = forward

    def backward(self, dY: np.ndarray) -> np.ndarray:
        acts = self._cache
        grads = []
        d = np.atleast_2d(dY)
        for k in range(len(self.W) - 1, -1, -1):
            gW = acts[k].T @ d
            gb = d.sum(0)
            grads.append((gW, gb))
            if k > 0:
                d = (d @ self.W[k].T) * (1.0 - acts[k] ** 2)
        grads.reverse()
        return np.concatenate([np.concatenate([gw.ravel(), gb]) for gw, gb in grads])


class Adam:
    """Adam optimiser acting on a flat parameter vector (minimisation)."""

    def __init__(self, n: int, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        mh, vh = self.m / (1 - self.b1 ** self.t), self.v / (1 - self.b2 ** self.t)
        return theta - self.lr * mh / (np.sqrt(vh) + self.eps)


# ------------------------------------------------------------------------------------------ gradient-free search
def cem(fitness: Callable[[np.ndarray, int], float], theta0, sigma0: float, iters: int, pop: int = 40,
        elite_frac: float = 0.2, seed: int = 0, sigma_floor: float = 1e-3, extra_std: float = 0.0, callback=None):
    """Cross-entropy method: sample from a diagonal Gaussian, refit it to the best ``elite_frac`` of the samples.

    ``fitness(theta, episode_seed)`` returns a scalar to **maximise**. ``extra_std`` adds a Gaussian "noise floor"
    to the refitted standard deviation that decays linearly to zero over the iterations, which prevents the
    distribution from collapsing before the optimum is reached. Returns ``(theta_mean, history)`` where history
    rows are ``(mean fitness, best fitness)`` per iteration.
    """
    rng = np.random.default_rng(seed)
    mu, sd = np.array(theta0, float), np.full(len(theta0), sigma0)
    n_elite = max(2, int(round(elite_frac * pop)))
    hist = []
    for it in range(iters):
        seeds = rng.integers(0, 2 ** 31 - 1, size=1)[0]               # common random numbers inside an iteration
        thetas = mu + sd * rng.normal(size=(pop, mu.size))
        f = np.array([fitness(th, int(seeds)) for th in thetas])
        elite = thetas[np.argsort(f)[-n_elite:]]
        mu = elite.mean(0)
        sd = np.maximum(np.sqrt(elite.var(0) + (extra_std * (1.0 - it / iters)) ** 2), sigma_floor)
        hist.append((float(f.mean()), float(f.max())))
        if callback:
            callback(it, mu, hist[-1])
    return mu, np.array(hist)


def es(fitness: Callable[[np.ndarray, int], float], theta0, sigma: float, lr: float, iters: int, pairs: int = 16,
       seed: int = 0, callback=None):
    """Evolution strategy with antithetic sampling, centred-rank fitness shaping and Adam.

    ``fitness(theta, episode_seed)`` is maximised; both members of an antithetic pair share the same episode
    seed (common random numbers) which reduces the variance of the gradient estimate. Returns
    ``(theta, history)`` with ``history[k] = fitness(theta_k)`` evaluated with that iteration's seed.
    """
    rng = np.random.default_rng(seed)
    theta = np.array(theta0, float)
    opt = Adam(theta.size, lr=lr)
    hist = []
    for it in range(iters):
        eps = rng.normal(size=(pairs, theta.size))
        ep_seed = int(rng.integers(0, 2 ** 31 - 1))
        fp = np.array([fitness(theta + sigma * e, ep_seed) for e in eps])
        fm = np.array([fitness(theta - sigma * e, ep_seed) for e in eps])
        allf = np.concatenate([fp, fm])
        ranks = np.empty_like(allf)
        ranks[np.argsort(allf)] = np.arange(allf.size)
        shaped = ranks / (allf.size - 1) - 0.5
        sp, sm = shaped[:pairs], shaped[pairs:]
        grad_ascent = ((sp - sm)[:, None] * eps).sum(0) / (2 * pairs * sigma)
        theta = opt.step(theta, -grad_ascent)
        hist.append(float(allf.mean()))
        if callback:
            callback(it, theta, hist[-1])
    return theta, np.array(hist)

"""LinUCB contextual-bandit tests."""

import numpy as np
import pytest

from bandit.contextual import LinearContextualBandit, LinUCB, run_contextual


def _env(seed=0):
    # 3 arms, 4-dim contexts, well-separated linear weights
    theta = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    return LinearContextualBandit(theta, sigma=0.1, seed=seed)


def test_env_optimal_mean_is_best_arm():
    env = _env()
    x = np.array([0.3, 0.9, -0.2, 0.5])
    opt = env.optimal_mean(x)
    assert opt == pytest.approx(max(env.mean(a, x) for a in range(env.n_arms)))


def test_linucb_select_returns_valid_arm():
    algo = LinUCB(n_arms=3, dim=4, alpha=1.0)
    a = algo.select(np.ones(4))
    assert a in (0, 1, 2)


def test_linucb_validates():
    with pytest.raises(ValueError):
        LinUCB(n_arms=0, dim=3)
    with pytest.raises(ValueError):
        LinearContextualBandit(np.zeros(3))  # not 2-D


def test_linucb_regret_is_sublinear():
    """Average regret per step in the 2nd half should be well below the 1st
    half — the signature of a learning (sublinear-regret) policy."""
    env = _env(seed=1)
    algo = LinUCB(n_arms=3, dim=4, alpha=1.0)
    regret = run_contextual(env, algo, horizon=3000)
    half = len(regret) // 2
    first_half_rate = regret[half] / half
    second_half_rate = (regret[-1] - regret[half]) / (len(regret) - half)
    assert second_half_rate < 0.5 * first_half_rate


def test_linucb_beats_random():
    """LinUCB should accumulate far less regret than uniform-random selection."""
    _env(seed=2)
    lin = run_contextual(_env(seed=2), LinUCB(n_arms=3, dim=4, alpha=1.0),
                         horizon=2000)

    # Random policy regret on the same environment
    rng = np.random.default_rng(2)
    env_r = _env(seed=2)
    cum, rand = 0.0, []
    for _ in range(2000):
        x = env_r.context()
        a = int(rng.integers(0, env_r.n_arms))
        env_r.pull(a, x)
        cum += env_r.optimal_mean(x) - env_r.mean(a, x)
        rand.append(cum)

    assert lin[-1] < 0.5 * rand[-1]


class _DirectInverseLinUCB(LinUCB):
    """Reference implementation: refactorise A_a on every index evaluation."""

    def select(self, x):
        x = np.asarray(x, dtype=float)
        best_arm, best_p = 0, -np.inf
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            p = (A_inv @ self.b[a]) @ x + self.alpha * np.sqrt(max(x @ A_inv @ x, 0.0))
            if p > best_p:
                best_arm, best_p = a, p
        return best_arm


def test_sherman_morrison_tracks_the_true_inverse():
    """A_inv is maintained by rank-1 updates; it must not drift from the
    inverse it stands in for."""
    algo = LinUCB(n_arms=3, dim=5, alpha=1.0)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        x = rng.standard_normal(5)
        algo.update(algo.select(x), float(rng.normal()), x)
    for a in range(algo.n_arms):
        assert np.allclose(algo.A_inv[a], np.linalg.inv(algo.A[a]), atol=1e-10)


def test_sherman_morrison_picks_the_same_arms_as_the_direct_inverse():
    """The optimisation is meant to be invisible: identical decisions, 1k
    rounds, same environment."""
    fast, slow = LinUCB(3, 4, alpha=1.0), _DirectInverseLinUCB(3, 4, alpha=1.0)
    env = _env(seed=5)
    for _ in range(1000):
        x = env.context()
        arm = fast.select(x)
        assert slow.select(x) == arm
        r = env.pull(arm, x)
        fast.update(arm, r, x)
        slow.update(arm, r, x)

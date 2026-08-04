"""Algorithm unit tests."""

import math

import pytest

from bandit.algos import (
    EXP3,
    UCB1,
    EpsilonGreedy,
    ThompsonBernoulli,
    ThompsonGaussian,
    annealed,
    anytime_gamma,
)
from bandit.envs import BernoulliBandit


def test_epsilon_greedy_records_value():
    a = EpsilonGreedy(n_arms=2, eps=0.0, seed=0)
    a.update(arm=0, reward=1.0)
    a.update(arm=0, reward=0.0)
    assert a.values[0] == pytest.approx(0.5)
    assert a.counts[0] == 2


def test_epsilon_greedy_explore_when_eps_one():
    a = EpsilonGreedy(n_arms=3, eps=1.0, seed=0)
    picks = []
    for t in range(1, 100):
        i = a.select(t)
        a.update(i, 10.0 if i == 0 else 0.0)  # bias toward arm 0
        picks.append(i)
    # With eps=1 we keep seeing every arm, not just the empirical best
    assert set(picks) == {0, 1, 2}


def test_epsilon_greedy_exploit_when_eps_zero():
    a = EpsilonGreedy(n_arms=3, eps=0.0, seed=0)
    a.update(0, 0.1)
    a.update(1, 0.9)
    a.update(2, 0.5)
    assert a.select(1) == 1


def test_epsilon_greedy_tries_every_arm_before_exploiting():
    """Regression: with eps=0 the old select() argmaxed over all-zero value
    estimates, returned arm 0 forever, and never observed the other arms."""
    a = EpsilonGreedy(n_arms=3, eps=0.0, seed=0)
    picks = []
    for t in range(1, 4):
        i = a.select(t)
        a.update(i, 1.0 if i == 2 else 0.0)
        picks.append(i)
    assert sorted(picks) == [0, 1, 2]
    # ... and having seen everything, it exploits the best arm
    assert a.select(4) == 2


def test_epsilon_greedy_breaks_value_ties_at_random():
    """Regression: max() used to resolve ties deterministically toward the
    lowest arm index, biasing exploitation onto arm 0."""
    a = EpsilonGreedy(n_arms=2, eps=0.0, seed=0)
    a.update(0, 0.5)
    a.update(1, 0.5)
    picks = {a.select(t) for t in range(1, 201)}
    assert picks == {0, 1}


def test_annealed_schedule_decreases():
    s = annealed(c=1.0)
    assert s(1) >= s(10) >= s(100)


def test_ucb1_pulls_each_arm_first():
    a = UCB1(n_arms=4)
    picks = []
    for t in range(1, 5):
        i = a.select(t)
        a.update(i, 1.0)
        picks.append(i)
    assert sorted(picks) == [0, 1, 2, 3]


def test_ucb1_index_widens_for_untried():
    a = UCB1(n_arms=2)
    a.update(0, 1.0)  # arm 0 looks good
    # arm 1 hasn't been pulled — UCB should pick it next
    assert a.select(2) == 1


def test_ucb1_rejects_invalid():
    with pytest.raises(ValueError):
        UCB1(n_arms=0)


def test_thompson_bernoulli_updates_posterior():
    a = ThompsonBernoulli(n_arms=2, seed=0)
    a.update(0, 1.0)
    a.update(0, 1.0)
    a.update(0, 0.0)
    assert a.alphas[0] == 3.0  # prior 1 + 2 successes
    assert a.betas[0] == 2.0   # prior 1 + 1 failure


def test_thompson_bernoulli_validates_reward():
    a = ThompsonBernoulli(n_arms=2)
    with pytest.raises(ValueError):
        a.update(0, 1.5)


def test_thompson_gaussian_posterior_means():
    a = ThompsonGaussian(n_arms=1, sigma_obs=1.0, mu_0=0.0, sigma_0=100.0, seed=0)
    for _ in range(100):
        a.update(0, 5.0)
    mu_n, _ = a._posterior(0)
    assert mu_n == pytest.approx(5.0, abs=0.05)


def test_exp3_probs_sum_to_one():
    a = EXP3(n_arms=4, gamma=0.1, seed=0)
    probs = a._probs()
    assert abs(sum(probs) - 1.0) < 1e-12
    assert all(0 < p < 1 for p in probs)


def test_exp3_weight_grows_on_reward():
    a = EXP3(n_arms=3, gamma=0.5, seed=0)
    w_before = a.weights[0]
    a.update(0, 1.0)
    assert a.weights[0] > w_before


def test_exp3_validates():
    with pytest.raises(ValueError):
        EXP3(n_arms=3, gamma=0.0)
    with pytest.raises(ValueError):
        EXP3(n_arms=3, gamma=1.5)
    a = EXP3(n_arms=3, gamma=0.1)
    with pytest.raises(ValueError):
        a.update(0, -0.1)


def test_exp3_long_horizon_weights_stay_finite():
    """Regression: linear-space weights overflowed to inf around t~9,000
    (gamma=0.2, 2 arms), probabilities went NaN, and the NaN-safe fallthrough
    in select() silently returned the last arm forever."""
    env = BernoulliBandit([0.2, 0.8], seed=1)
    a = EXP3(n_arms=2, gamma=0.2, seed=0)
    for t in range(1, 20001):
        arm = a.select(t)
        a.update(arm, env.pull(arm))
    assert all(math.isfinite(w) for w in a.weights)
    probs = a._probs()
    assert all(math.isfinite(p) and 0.0 < p < 1.0 for p in probs)
    assert sum(probs) == pytest.approx(1.0)
    # The gamma/K exploration floor must still reach every arm.
    picks = {a.select(t) for t in range(20001, 20301)}
    assert picks == {0, 1}


def test_exp3_update_uses_select_time_distribution():
    """The importance weight must come from the distribution the arm was
    actually drawn from, not one recomputed after the fact."""
    a = EXP3(n_arms=2, gamma=0.5, seed=0)
    a.select(1)  # caches the uniform distribution [0.5, 0.5]
    a._log_w[0] += 5.0  # out-of-band skew: recomputing would now give p != 0.5
    a.update(0, 1.0)
    # increment = gamma * (reward / p_cached) / K = 0.5 * (1 / 0.5) / 2 = 0.5
    assert a._log_w[0] == pytest.approx(5.5)


def test_anytime_gamma_schedule():
    assert anytime_gamma(3, 1) == 1.0  # capped at 1 early
    g1, g2 = anytime_gamma(3, 1000), anytime_gamma(3, 100000)
    assert 0.0 < g2 < g1 < 1.0
    expected = math.sqrt(3 * math.log(3) / ((math.e - 1) * 1000))
    assert g1 == pytest.approx(expected)
    with pytest.raises(ValueError):
        anytime_gamma(3, 0)


def _second_half_regret_slower(algo_factory, horizon, probs=(0.25, 0.5, 0.75)):
    """Second-half per-step pseudo-regret rate must be well below the first
    half's — the signature of a sublinear-regret (still-learning) policy."""
    from bandit.metrics import cumulative_pseudo_regret

    env = BernoulliBandit(list(probs), seed=3)
    algo = algo_factory(len(probs))
    arms = []
    for t in range(1, horizon + 1):
        a = algo.select(t)
        algo.update(a, env.pull(a))
        arms.append(a)
    regret = cumulative_pseudo_regret(env, arms)
    half = horizon // 2
    first_rate = regret[half] / half
    second_rate = (regret[-1] - regret[half]) / (horizon - half)
    return first_rate, second_rate


def test_exp3_anytime_regret_sublinear_100k():
    """100k-step run: weights stay finite and regret keeps decelerating.
    (With the old linear-space weights this horizon was unreachable.)"""
    first, second = _second_half_regret_slower(
        lambda k: EXP3(n_arms=k, gamma=None, seed=0), horizon=100_000)
    assert second < 0.7 * first


def test_ucb1_regret_sublinear_long_horizon():
    first, second = _second_half_regret_slower(
        lambda k: UCB1(n_arms=k), horizon=50_000)
    assert second < 0.5 * first


def test_thompson_regret_sublinear_long_horizon():
    first, second = _second_half_regret_slower(
        lambda k: ThompsonBernoulli(n_arms=k, seed=0), horizon=50_000)
    assert second < 0.5 * first

"""Benchmarks that tie measured regret back to the Lai-Robbins floor.

The rest of the suite checks that each policy does what its pseudocode says.
These tests check the claim the library is actually making: that the
asymptotically-optimal policies track ``C(env) * ln T`` and the others pay a
multiple of it. They are the executable version of `docs/figures.py optimality`,
run at a horizon short enough for CI.
"""

import math

import pytest

from bandit import (
    KLUCB,
    UCB1,
    BernoulliBandit,
    ThompsonBernoulli,
    cumulative_pseudo_regret,
    lai_robbins_lower_bound,
)

HORIZON = 20_000
SEEDS = (0, 1, 2, 3)

# Two regimes. Hoeffding (UCB1) is tight in the middle of [0, 1] and hopeless
# near the edges, so the second instance is where the KL index earns its keep.
MODERATE = [0.50, 0.45, 0.40, 0.35, 0.30]
RARE = [0.10, 0.02, 0.02, 0.01, 0.01]


def regret_per_log_t(factory, probs, horizon=HORIZON, seeds=SEEDS) -> float:
    """Mean R(T) / ln T — the quantity the Lai-Robbins constant bounds."""
    totals = []
    for seed in seeds:
        env = BernoulliBandit(probs, seed=seed)
        algo = factory(env.n_arms, seed + 7919)
        arms = []
        for t in range(1, horizon + 1):
            arm = algo.select(t)
            algo.update(arm, env.pull(arm))
            arms.append(arm)
        totals.append(cumulative_pseudo_regret(env, arms)[-1])
    return sum(totals) / len(totals) / math.log(horizon)


@pytest.mark.parametrize("factory", [
    lambda k, s: KLUCB(k),
    lambda k, s: ThompsonBernoulli(k, seed=s),
], ids=["klucb", "thompson"])
def test_optimal_policies_stay_at_the_floor(factory):
    """KL-UCB and Thompson should sit at or under C * ln T at this horizon:
    the bound is asymptotic, and both approach it from below."""
    floor = lai_robbins_lower_bound(BernoulliBandit(RARE))
    assert regret_per_log_t(factory, RARE) < 1.2 * floor


def test_ucb1_pays_a_large_multiple_of_the_floor_on_rare_rewards():
    """UCB1's Hoeffding bonus ignores that a mean near 0.01 concentrates far
    faster than 1/sqrt(n) suggests, so it over-explores by roughly an order of
    magnitude exactly where KL-UCB is tight."""
    floor = lai_robbins_lower_bound(BernoulliBandit(RARE))
    ucb1 = regret_per_log_t(lambda k, s: UCB1(k), RARE)
    klucb = regret_per_log_t(lambda k, s: KLUCB(k), RARE)
    assert ucb1 > 4.0 * floor
    assert klucb < 0.2 * ucb1


def test_kl_index_also_wins_with_moderate_gaps():
    """Less dramatic than the rare-reward regime, but the ordering holds."""
    ucb1 = regret_per_log_t(lambda k, s: UCB1(k), MODERATE)
    klucb = regret_per_log_t(lambda k, s: KLUCB(k), MODERATE)
    assert klucb < 0.6 * ucb1


def test_floor_is_reported_per_family_not_per_value_range():
    """Guard against the regression that made the bound value-range-driven:
    the same means with a different reward family must give a different C."""
    from bandit import GaussianBandit

    means = [0.5, 0.45, 0.40]
    bern = lai_robbins_lower_bound(BernoulliBandit(means))
    gauss = lai_robbins_lower_bound(GaussianBandit(means, sigmas=[1.0] * 3))
    assert gauss == pytest.approx(2 / 0.05 + 2 / 0.10)  # sum 2 / gap_i
    assert bern < 0.3 * gauss  # 0/1 rewards are much easier to tell apart

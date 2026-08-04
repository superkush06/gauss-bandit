"""KL-UCB: the index solver and the policy built on it."""

import math

import pytest

from bandit import KLUCB, UCB1, BernoulliBandit, bernoulli_kl, klucb_index


def test_index_solves_the_kl_equation():
    """u = klucb_index(p, level) must satisfy d(p, u) = level exactly."""
    for p_hat in (0.05, 0.3, 0.5, 0.87):
        for level in (0.01, 0.1, 0.5):
            u = klucb_index(p_hat, level)
            assert p_hat <= u <= 1.0
            assert bernoulli_kl(p_hat, u) == pytest.approx(level, abs=1e-4)


def test_index_is_monotone_in_the_budget():
    prev = 0.0
    for level in (0.001, 0.01, 0.1, 1.0, 10.0):
        u = klucb_index(0.4, level)
        assert u >= prev
        prev = u
    assert klucb_index(0.4, 100.0) == pytest.approx(1.0)


def test_index_edges():
    assert klucb_index(0.4, 0.0) == 0.4      # no budget, no optimism
    assert klucb_index(1.0, 5.0) == 1.0      # already at the ceiling
    assert klucb_index(0.0, 0.1) == pytest.approx(1 - math.exp(-0.1), abs=1e-5)
    with pytest.raises(ValueError):
        klucb_index(1.5, 0.1)


def test_index_reaches_the_ceiling_only_from_p_hat_itself():
    """d(p, 1) is infinite for every p < 1, so no finite exploration budget
    ever carries the index all the way to 1 from below; only an arm whose
    empirical mean is already 1 returns exactly 1. The solver used to carry a
    `d(p_hat, 1) <= level` shortcut for that case, which could not fire."""
    assert klucb_index(1.0, 1e-9) == 1.0
    for p_hat in (0.0, 0.5, 0.999):
        assert klucb_index(p_hat, 1e6) < 1.0


def test_index_never_exceeds_the_hoeffding_bonus():
    """Pinsker's inequality d(p, q) >= 2(p - q)^2 bounds the KL index by
    p_hat + sqrt(level / 2) — i.e. KL-UCB is never more optimistic than the
    quadratic bound UCB1 uses, and is usually a lot less."""
    for p_hat in (0.02, 0.2, 0.6, 0.95):
        for level in (0.02, 0.2, 2.0):
            assert klucb_index(p_hat, level) <= p_hat + math.sqrt(level / 2) + 1e-9


def test_pulls_every_arm_before_indexing():
    a = KLUCB(n_arms=4)
    picks = []
    for t in range(1, 5):
        arm = a.select(t)
        a.update(arm, 1.0)
        picks.append(arm)
    assert sorted(picks) == [0, 1, 2, 3]


def test_index_of_untried_arm_is_infinite():
    a = KLUCB(n_arms=2)
    assert a.index(0, t=10) == math.inf


def test_validates():
    with pytest.raises(ValueError):
        KLUCB(n_arms=0)
    with pytest.raises(ValueError):
        KLUCB(n_arms=2, c=-1.0)
    with pytest.raises(ValueError):
        KLUCB(n_arms=2).update(0, 1.5)


def test_concentrates_on_the_best_arm():
    env = BernoulliBandit([0.1, 0.3, 0.7], seed=4)
    a = KLUCB(n_arms=3)
    for t in range(1, 5001):
        arm = a.select(t)
        a.update(arm, env.pull(arm))
    assert a.counts[2] > 0.9 * sum(a.counts)


def test_beats_ucb1_when_rewards_are_rare():
    """Hoeffding is loosest near p = 0, which is exactly where KL-UCB wins.
    On arms with means 0.10/0.02/0.01 it should need far fewer suboptimal
    pulls than UCB1 over the same horizon."""
    probs = [0.10, 0.02, 0.02, 0.01, 0.01]
    horizon = 20_000

    def suboptimal_pulls(algo):
        env = BernoulliBandit(probs, seed=7)
        for t in range(1, horizon + 1):
            arm = algo.select(t)
            algo.update(arm, env.pull(arm))
        return horizon - algo.counts[0]

    assert suboptimal_pulls(KLUCB(5)) < 0.35 * suboptimal_pulls(UCB1(5))

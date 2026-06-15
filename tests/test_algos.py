"""Algorithm unit tests."""


import pytest

from bandit.algos import EpsilonGreedy, annealed


def test_epsilon_greedy_records_value():
    a = EpsilonGreedy(n_arms=2, eps=0.0, seed=0)
    a.update(arm=0, reward=1.0)
    a.update(arm=0, reward=0.0)
    assert a.values[0] == pytest.approx(0.5)
    assert a.counts[0] == 2


def test_epsilon_greedy_explore_when_eps_one():
    a = EpsilonGreedy(n_arms=3, eps=1.0, seed=0)
    a.update(0, 10.0)
    picks = [a.select(t) for t in range(1, 100)]
    assert len(set(picks)) >= 2


def test_epsilon_greedy_exploit_when_eps_zero():
    a = EpsilonGreedy(n_arms=3, eps=0.0, seed=0)
    a.update(0, 0.1)
    a.update(1, 0.9)
    a.update(2, 0.5)
    assert a.select(1) == 1


def test_annealed_schedule_decreases():
    s = annealed(c=1.0)
    assert s(1) >= s(10) >= s(100)

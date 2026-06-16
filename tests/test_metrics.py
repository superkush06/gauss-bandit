"""Regret-metric tests."""

import math

import pytest

from bandit.envs import BernoulliBandit
from bandit.metrics import (
    cumulative_pseudo_regret,
    cumulative_regret,
    lai_robbins_lower_bound,
)


def test_cumulative_pseudo_regret_zero_if_optimal_always():
    env = BernoulliBandit([0.2, 0.8], seed=0)
    # always pulling optimal arm (1) -> zero pseudo regret
    arms = [1] * 10
    pr = cumulative_pseudo_regret(env, arms)
    assert pr == [0.0] * 10


def test_cumulative_pseudo_regret_grows_on_suboptimal():
    env = BernoulliBandit([0.2, 0.8], seed=0)
    arms = [0] * 5
    pr = cumulative_pseudo_regret(env, arms)
    assert pr[-1] == pytest.approx(5 * 0.6)


def test_cumulative_regret_uses_observed_rewards():
    env = BernoulliBandit([0.5], seed=0)
    rewards = [1.0, 1.0, 1.0, 0.0, 0.0]
    cr = cumulative_regret(env, [0]*5, rewards)
    # optimal_mean=0.5, t=5 -> 2.5; cum reward = 3.0; regret = -0.5
    assert cr[-1] == pytest.approx(2.5 - 3.0)


def test_lai_robbins_is_finite_and_positive():
    env = BernoulliBandit([0.3, 0.5, 0.9], seed=0)
    c = lai_robbins_lower_bound(env)
    assert math.isfinite(c)
    assert c > 0

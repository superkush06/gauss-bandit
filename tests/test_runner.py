"""Runner tests."""

import numpy as np
import pytest

from bandit.algos import UCB1, ThompsonBernoulli
from bandit.envs import BernoulliBandit
from bandit.runner import run_experiment, run_one


def test_run_one_horizon_matches():
    res = run_one(
        env_factory=lambda seed: BernoulliBandit([0.1, 0.9], seed=seed),
        algo_factory=lambda n_arms, seed: UCB1(n_arms=n_arms),
        horizon=100, seed=0,
    )
    assert len(res.arms_pulled) == 100
    assert len(res.rewards) == 100
    assert len(res.pseudo_regret) == 100


def test_run_one_deterministic_under_seed():
    def f_env(seed):
        return BernoulliBandit([0.1, 0.9], seed=seed)
    def f_alg(n_arms, seed):
        return ThompsonBernoulli(n_arms=n_arms, seed=seed)
    a = run_one(f_env, f_alg, horizon=50, seed=42)
    b = run_one(f_env, f_alg, horizon=50, seed=42)
    assert a.arms_pulled == b.arms_pulled
    assert a.rewards == b.rewards


def test_run_experiment_aggregates():
    res = run_experiment(
        env_factory=lambda seed: BernoulliBandit([0.1, 0.9], seed=seed),
        algo_factory=lambda n_arms, seed: UCB1(n_arms=n_arms),
        horizon=200, n_runs=5, base_seed=0,
    )
    assert len(res.runs) == 5
    assert len(res.pseudo_regret_mean) == 200
    # Mean regret is non-decreasing
    pr = res.pseudo_regret_mean
    assert all(pr[i+1] >= pr[i] - 1e-9 for i in range(len(pr) - 1))


def test_algo_factory_receives_replication_seed():
    """The documented contract: algo_factory(n_arms, seed) gets a distinct
    seed per replication, offset from the env's."""
    seeds = []

    def f_alg(n_arms, seed):
        seeds.append(seed)
        return ThompsonBernoulli(n_arms=n_arms, seed=seed)

    run_experiment(
        env_factory=lambda seed: BernoulliBandit([0.4, 0.6], seed=seed),
        algo_factory=f_alg, horizon=10, n_runs=3, base_seed=100,
    )
    assert len(seeds) == len(set(seeds)) == 3


def test_algo_seed_is_not_clobbered():
    """Regression: run_one used to monkeypatch algo._rng with its own
    random.Random(seed + 7919), so two factories seeded 123 vs 999 produced
    byte-identical trajectories (the README's own seed was dead code)."""
    def f_env(seed):
        return BernoulliBandit([0.4, 0.6], seed=0)  # env stream held fixed

    def make(s):
        return lambda n_arms, seed: ThompsonBernoulli(n_arms=n_arms, seed=s)

    r1 = run_one(f_env, make(123), horizon=200, seed=0)
    r2 = run_one(f_env, make(999), horizon=200, seed=0)
    assert r1.arms_pulled != r2.arms_pulled


def test_numpy_generator_algorithm_runs_unmodified():
    """Regression: the _rng monkeypatch crashed any algorithm holding a numpy
    Generator (AttributeError: 'Random' object has no attribute 'integers')."""

    class NumpyRandomAlgo:
        def __init__(self, n_arms, seed):
            self.n_arms = n_arms
            self._rng = np.random.default_rng(seed)

        def select(self, t):
            return int(self._rng.integers(0, self.n_arms))

        def update(self, arm, reward):
            pass

        def reset(self):
            pass

    res = run_one(
        env_factory=lambda seed: BernoulliBandit([0.4, 0.6], seed=seed),
        algo_factory=lambda n_arms, seed: NumpyRandomAlgo(n_arms, seed),
        horizon=50, seed=0,
    )
    assert len(res.arms_pulled) == 50


def test_old_style_factory_warns_but_still_runs():
    """Factories that only take n_arms keep working (their own seeding,
    e.g. via closure, is now respected) but raise a DeprecationWarning."""
    with pytest.warns(DeprecationWarning):
        res = run_one(
            env_factory=lambda seed: BernoulliBandit([0.1, 0.9], seed=seed),
            algo_factory=lambda n_arms: ThompsonBernoulli(n_arms=n_arms, seed=7),
            horizon=20, seed=0,
        )
    assert len(res.arms_pulled) == 20

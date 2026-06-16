"""Runner tests."""

from bandit.algos import UCB1
from bandit.envs import BernoulliBandit
from bandit.runner import run_experiment, run_one


def test_run_one_horizon_matches():
    res = run_one(
        env_factory=lambda seed: BernoulliBandit([0.1, 0.9], seed=seed),
        algo_factory=lambda n_arms: UCB1(n_arms=n_arms),
        horizon=100, seed=0,
    )
    assert len(res.arms_pulled) == 100
    assert len(res.rewards) == 100
    assert len(res.pseudo_regret) == 100


def test_run_one_deterministic_under_seed():
    def f_env(seed):
        return BernoulliBandit([0.1, 0.9], seed=seed)
    def f_alg(n_arms):
        return UCB1(n_arms=n_arms)
    a = run_one(f_env, f_alg, horizon=50, seed=42)
    b = run_one(f_env, f_alg, horizon=50, seed=42)
    assert a.arms_pulled == b.arms_pulled
    assert a.rewards == b.rewards


def test_run_experiment_aggregates():
    res = run_experiment(
        env_factory=lambda seed: BernoulliBandit([0.1, 0.9], seed=seed),
        algo_factory=lambda n_arms: UCB1(n_arms=n_arms),
        horizon=200, n_runs=5, base_seed=0,
    )
    assert len(res.runs) == 5
    assert len(res.pseudo_regret_mean) == 200
    # Mean regret is non-decreasing
    pr = res.pseudo_regret_mean
    assert all(pr[i+1] >= pr[i] - 1e-9 for i in range(len(pr) - 1))

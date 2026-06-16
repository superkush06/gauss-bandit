"""Experiment runner: replicate (env, algo) pairs and collect regret."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .metrics import cumulative_pseudo_regret


@dataclass
class RunResult:
    """One replication's trajectory."""
    arms_pulled: list[int]
    rewards: list[float]
    pseudo_regret: list[float]


@dataclass
class ExperimentResult:
    """Aggregated over replications."""
    pseudo_regret_mean: list[float]
    pseudo_regret_std: list[float]
    runs: list[RunResult]


def run_one(env_factory, algo_factory, horizon: int, seed: int = 0) -> RunResult:
    """One replication: fresh env + fresh algo, run for `horizon` steps."""
    env = env_factory(seed=seed)
    algo = algo_factory(n_arms=env.n_arms)
    if hasattr(algo, "_rng"):
        import random
        algo._rng = random.Random(seed + 7919)
    arms, rewards = [], []
    for t in range(1, horizon + 1):
        a = algo.select(t)
        r = env.pull(a)
        algo.update(a, r)
        arms.append(a)
        rewards.append(r)
    return RunResult(
        arms_pulled=arms, rewards=rewards,
        pseudo_regret=cumulative_pseudo_regret(env, arms),
    )


def run_experiment(env_factory, algo_factory, horizon: int,
                   n_runs: int = 50, base_seed: int = 0) -> ExperimentResult:
    runs = [run_one(env_factory, algo_factory, horizon, seed=base_seed + i)
            for i in range(n_runs)]
    # Stack regrets columnwise
    mean = [statistics.fmean(run.pseudo_regret[t] for run in runs)
            for t in range(horizon)]
    std = [statistics.stdev([run.pseudo_regret[t] for run in runs])
           if n_runs > 1 else 0.0 for t in range(horizon)]
    return ExperimentResult(pseudo_regret_mean=mean,
                            pseudo_regret_std=std, runs=runs)

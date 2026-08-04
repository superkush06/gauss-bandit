"""Experiment runner: replicate (env, algo) pairs and collect regret."""

from __future__ import annotations

import inspect
import statistics
import warnings
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


def _make_algo(algo_factory, n_arms: int, seed: int):
    """Instantiate the algorithm, threading the replication seed through.

    The contract is ``algo_factory(n_arms, seed)``. Factories that only take
    ``n_arms`` still work (with a DeprecationWarning): the runner never
    touches the algorithm's internals, so whatever RNG the factory built —
    including a seed baked in via closure, or a numpy Generator — is used
    as-is.
    """
    try:
        params = inspect.signature(algo_factory).parameters
    except (TypeError, ValueError):  # builtins / C callables: assume new-style
        return algo_factory(n_arms=n_arms, seed=seed)
    if "seed" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD
                               for p in params.values()):
        return algo_factory(n_arms=n_arms, seed=seed)
    warnings.warn(
        "algo_factory should accept (n_arms, seed) so each replication gets "
        "an independently seeded algorithm; factories taking only n_arms are "
        "deprecated",
        DeprecationWarning, stacklevel=3)
    return algo_factory(n_arms=n_arms)


def run_one(env_factory, algo_factory, horizon: int, seed: int = 0) -> RunResult:
    """One replication: fresh env + fresh algo, run for `horizon` steps.

    `env_factory(seed)` builds the environment; `algo_factory(n_arms, seed)`
    builds the algorithm (its seed is offset from the env's so the two RNG
    streams never coincide).
    """
    env = env_factory(seed=seed)
    algo = _make_algo(algo_factory, env.n_arms, seed + 7919)
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

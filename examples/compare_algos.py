"""Six policies, one 10-arm Bernoulli instance, a short horizon.

Run:  python3 examples/compare_algos.py

Short horizons are where intuition goes wrong: the asymptotically-optimal
policies have barely started paying their ln T bill, so the ranking here is
not the ranking at T = 100,000. `examples/optimality.py` shows what happens
when you keep running.
"""

import argparse
import math

from bandit.algos import (
    EXP3,
    KLUCB,
    UCB1,
    EpsilonGreedy,
    ThompsonBernoulli,
    annealed,
)
from bandit.envs import BernoulliBandit
from bandit.metrics import lai_robbins_lower_bound
from bandit.runner import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=2000)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    probs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]

    algos = {
        "eps=0.1":            lambda n_arms, seed: EpsilonGreedy(n_arms, eps=0.1, seed=seed),
        "eps annealed":       lambda n_arms, seed: EpsilonGreedy(n_arms, eps=annealed(c=2.0), seed=seed),
        "UCB1":               lambda n_arms, seed: UCB1(n_arms=n_arms),
        "KL-UCB":             lambda n_arms, seed: KLUCB(n_arms=n_arms),
        "Thompson Bernoulli": lambda n_arms, seed: ThompsonBernoulli(n_arms=n_arms, seed=seed),
        "EXP3 (anytime)":     lambda n_arms, seed: EXP3(n_arms=n_arms, seed=seed),
    }

    lr = lai_robbins_lower_bound(BernoulliBandit(probs))
    print(f"Horizon: {args.horizon}, Runs: {args.runs}, K={len(probs)}, "
          f"best arm prob={max(probs):.2f}")
    print(f"Lai-Robbins: any consistent policy has regret >= "
          f"{lr:.2f} ln T ~ {lr * math.log(args.horizon):.1f} at this horizon")
    print(f"{'algorithm':<22} {'final pseudo-regret':>20} {'std':>10}")
    print("-" * 56)
    for name, factory in algos.items():
        res = run_experiment(
            env_factory=lambda seed: BernoulliBandit(probs, seed=seed),
            algo_factory=factory,
            horizon=args.horizon,
            n_runs=args.runs,
        )
        print(f"{name:<22} {res.pseudo_regret_mean[-1]:>20.2f} "
              f"{res.pseudo_regret_std[-1]:>10.2f}")


if __name__ == "__main__":
    main()

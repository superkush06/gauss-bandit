"""Compare algorithms on a 10-arm Bernoulli bandit.

Run:  PYTHONPATH=. python3 examples/compare_algos.py
"""

import argparse

from bandit.algos import EXP3, UCB1, EpsilonGreedy, ThompsonBernoulli, annealed
from bandit.envs import BernoulliBandit
from bandit.runner import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=2000)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    probs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]

    algos = {
        "eps=0.1":            lambda n_arms: EpsilonGreedy(n_arms, eps=0.1, seed=0),
        "eps annealed":       lambda n_arms: EpsilonGreedy(n_arms, eps=annealed(c=2.0), seed=0),
        "UCB1":               lambda n_arms: UCB1(n_arms=n_arms),
        "Thompson Bernoulli": lambda n_arms: ThompsonBernoulli(n_arms=n_arms, seed=0),
        "EXP3 (gamma=0.07)":  lambda n_arms: EXP3(n_arms=n_arms, gamma=0.07, seed=0),
    }

    print(f"Horizon: {args.horizon}, Runs: {args.runs}, K={len(probs)}, "
          f"best arm prob={max(probs):.2f}")
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

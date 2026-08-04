"""How close does each policy get to the Lai-Robbins floor?

    python3 examples/optimality.py

Lai & Robbins (1985) proved that any policy which is consistent across
problem instances must pull each suboptimal arm at least ln T / KL_i times,
so its regret is at least C(env) ln T with

    C(env) = sum over suboptimal arms of  (mu* - mu_i) / KL(mu_i || mu*).

That constant is computable — `lai_robbins_lower_bound` returns it — which
means "is this policy any good?" has a number, not an opinion. This script
prints R(T) / (C ln T) for four policies: 1.0 is the asymptotic floor, and
anything that keeps climbing is paying a constant multiple of optimal forever.

The instance is deliberately the awkward one: five arms with means near zero,
where the Hoeffding bound behind UCB1 is loosest and the KL index is exact.
Runs in a few seconds. `docs/figures.py optimality` carries the same study
out to 200,000 rounds and plots it.
"""

from __future__ import annotations

import argparse
import math

from bandit import (
    KLUCB,
    UCB1,
    BernoulliBandit,
    EpsilonGreedy,
    ThompsonBernoulli,
    cumulative_pseudo_regret,
    lai_robbins_lower_bound,
)

PROBS = [0.10, 0.02, 0.02, 0.01, 0.01]
POLICIES = {
    "KL-UCB": lambda k, s: KLUCB(k),
    "Thompson": lambda k, s: ThompsonBernoulli(k, seed=s),
    "UCB1": lambda k, s: UCB1(k),
    "eps-greedy (0.1)": lambda k, s: EpsilonGreedy(k, eps=0.1, seed=s),
}


def mean_regret_curve(factory, horizon: int, runs: int) -> list[float]:
    acc = [0.0] * horizon
    for seed in range(runs):
        env = BernoulliBandit(PROBS, seed=seed)
        algo = factory(env.n_arms, seed + 7919)
        arms = []
        for t in range(1, horizon + 1):
            arm = algo.select(t)
            algo.update(arm, env.pull(arm))
            arms.append(arm)
        for i, r in enumerate(cumulative_pseudo_regret(env, arms)):
            acc[i] += r
    return [v / runs for v in acc]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=20_000)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    floor = lai_robbins_lower_bound(BernoulliBandit(PROBS))
    checkpoints = [t for t in (100, 1_000, 5_000, 20_000, 100_000)
                   if t <= args.horizon]

    print(f"K={len(PROBS)} Bernoulli {PROBS}")
    print(f"Lai-Robbins constant C = {floor:.2f}  "
          f"(regret >= {floor:.2f} ln T for any consistent policy)")
    print(f"{args.runs} runs x {args.horizon:,} rounds\n")
    print("R(T) / (C ln T)   — 1.00 is the asymptotic floor")
    print(f"{'policy':<20}" + "".join(f"{f'T={t:,}':>12}" for t in checkpoints))
    print("-" * (20 + 12 * len(checkpoints)))
    for name, factory in POLICIES.items():
        curve = mean_regret_curve(factory, args.horizon, args.runs)
        cells = "".join(f"{curve[t - 1] / (floor * math.log(t)):>12.2f}"
                        for t in checkpoints)
        print(f"{name:<20}{cells}")


if __name__ == "__main__":
    main()

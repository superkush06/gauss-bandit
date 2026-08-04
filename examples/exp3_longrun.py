"""EXP3 on a 100,000-step horizon: fixed gamma vs the anytime schedule.

Run:  python3 examples/exp3_longrun.py

This is exactly the regime that used to be unreachable: with weights kept in
linear space they overflowed to inf around t~9,000 and EXP3 silently pulled
one arm forever. Log-space weights make the 100k-step run routine, and the
anytime schedule gamma_t = sqrt(K ln K / ((e-1) t)) keeps regret sublinear
where any fixed gamma eventually pays a linear gamma*t exploration tax.
"""

from __future__ import annotations

import argparse

from bandit.algos import EXP3, anytime_gamma
from bandit.envs import BernoulliBandit
from bandit.metrics import cumulative_pseudo_regret


def run(gamma: float | None, horizon: int, seed: int) -> list[float]:
    env = BernoulliBandit([0.25, 0.5, 0.75], seed=seed)
    algo = EXP3(n_arms=env.n_arms, gamma=gamma, seed=seed + 7919)
    arms = []
    for t in range(1, horizon + 1):
        a = algo.select(t)
        algo.update(a, env.pull(a))
        arms.append(a)
    return cumulative_pseudo_regret(env, arms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=100_000)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    variants: dict[str, float | None] = {
        "gamma=0.2 (fixed)": 0.2,
        "gamma=0.07 (fixed)": 0.07,
        "anytime schedule": None,
    }
    curves = {}
    for name, gamma in variants.items():
        per_run = [run(gamma, args.horizon, seed=s) for s in range(args.runs)]
        curves[name] = [sum(c[t] for c in per_run) / args.runs
                        for t in range(args.horizon)]

    print(f"K=3 Bernoulli [0.25, 0.50, 0.75], horizon={args.horizon:,}, "
          f"{args.runs} runs")
    print(f"final anytime gamma_t = "
          f"{anytime_gamma(3, args.horizon):.4f}")
    print(f"{'variant':<20} {'regret @10k':>12} {'@50k':>10} {'@100k':>10}")
    print("-" * 56)
    for name, c in curves.items():
        cols = [c[min(t, args.horizon) - 1] for t in (10_000, 50_000, 100_000)]
        print(f"{name:<20} " + " ".join(f"{v:>10.0f}" for v in cols))

    print("\nChart: python3 docs/figures.py exp3")


if __name__ == "__main__":
    main()

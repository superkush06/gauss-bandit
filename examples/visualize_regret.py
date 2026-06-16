"""Plot mean regret curves with shaded std bands.

Run:  PYTHONPATH=. python3 examples/visualize_regret.py
"""

import argparse

from bandit.algos import EXP3, UCB1, EpsilonGreedy, ThompsonBernoulli
from bandit.envs import BernoulliBandit
from bandit.runner import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1500)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", default="regret.png")
    args = ap.parse_args()

    probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    algos = {
        "EpsilonGreedy (0.1)": lambda n_arms: EpsilonGreedy(n_arms, eps=0.1, seed=0),
        "UCB1":                lambda n_arms: UCB1(n_arms=n_arms),
        "Thompson":            lambda n_arms: ThompsonBernoulli(n_arms=n_arms, seed=0),
        "EXP3 (0.07)":         lambda n_arms: EXP3(n_arms=n_arms, gamma=0.07, seed=0),
    }

    results = {}
    for name, factory in algos.items():
        results[name] = run_experiment(
            env_factory=lambda seed: BernoulliBandit(probs, seed=seed),
            algo_factory=factory,
            horizon=args.horizon, n_runs=args.runs,
        )
        print(f"{name:<22} final regret = {results[name].pseudo_regret_mean[-1]:.2f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nInstall matplotlib to render plots.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in results.items():
        ts = list(range(1, args.horizon + 1))
        m = res.pseudo_regret_mean
        s = res.pseudo_regret_std
        ax.plot(ts, m, label=name)
        ax.fill_between(ts, [mi - si for mi, si in zip(m, s, strict=False)],
                        [mi + si for mi, si in zip(m, s, strict=False)], alpha=0.15)
    ax.set_xlabel("time t")
    ax.set_ylabel("cumulative pseudo-regret")
    ax.set_title(f"Bandit algorithms on Bernoulli-{len(probs)} ({args.runs} runs)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

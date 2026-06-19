"""Render the README hero image: regret curves.

Run:  python examples/render_hero.py   ->  writes docs/demo.png

Top: cumulative pseudo-regret of the context-free algorithms on a Bernoulli
bandit (Thompson/UCB are sublinear; fixed-eps greedy is ~linear).
Bottom: LinUCB cumulative regret on a linear contextual bandit (sublinear).
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bandit.algos import EXP3, UCB1, EpsilonGreedy, ThompsonBernoulli  # noqa: E402
from bandit.contextual import (  # noqa: E402
    LinearContextualBandit,
    LinUCB,
    run_contextual,
)
from bandit.envs import BernoulliBandit  # noqa: E402
from bandit.runner import run_experiment  # noqa: E402


def main() -> None:
    horizon, runs = 2000, 40
    probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    algos = {
        "ε-greedy (0.1)": lambda n_arms: EpsilonGreedy(n_arms, eps=0.1, seed=0),
        "UCB1": lambda n_arms: UCB1(n_arms=n_arms),
        "Thompson": lambda n_arms: ThompsonBernoulli(n_arms=n_arms, seed=0),
        "EXP3 (0.07)": lambda n_arms: EXP3(n_arms=n_arms, gamma=0.07, seed=0),
    }

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    ts = range(1, horizon + 1)
    for name, factory in algos.items():
        res = run_experiment(
            env_factory=lambda seed: BernoulliBandit(probs, seed=seed),
            algo_factory=factory, horizon=horizon, n_runs=runs,
        )
        ax1.plot(ts, res.pseudo_regret_mean, lw=1.4, label=name)
    ax1.set_title("gauss-bandit — cumulative regret on a 5-arm Bernoulli bandit "
                  f"({runs} runs)")
    ax1.set_ylabel("pseudo-regret")
    ax1.legend(loc="upper left", ncol=4, fontsize=9)

    # Contextual: LinUCB regret, averaged over runs
    curves = []
    for s in range(runs):
        env = LinearContextualBandit(
            np.eye(3, 4) * 1.0, sigma=0.1, seed=s)  # 3 arms, dim 4
        curves.append(run_contextual(env, LinUCB(3, 4, alpha=1.0), horizon))
    ax2.plot(ts, np.mean(curves, axis=0), color="#7b3294", lw=1.6, label="LinUCB")
    ax2.set_title("LinUCB — cumulative regret on a linear contextual bandit")
    ax2.set_xlabel("round t")
    ax2.set_ylabel("regret")
    ax2.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parents[1] / "docs" / "demo.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

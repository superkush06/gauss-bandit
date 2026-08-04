"""Render every figure in the README.

    python3 docs/figures.py              # all figures
    python3 docs/figures.py optimality   # just one

Each figure prints the numbers it plots before saving, so the README's tables
and its charts can never drift apart: they come out of the same run.

Nothing here is decorative. `optimality` is the claim the library exists to
support (regret per ln T against the Lai-Robbins constant), `exp3` is the
long-horizon regime that used to be unreachable, and `contextual` is the one
problem where none of the context-free policies can win at all.
"""

from __future__ import annotations

import math
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bandit import (  # noqa: E402
    EXP3,
    KLUCB,
    UCB1,
    BernoulliBandit,
    EpsilonGreedy,
    LinearContextualBandit,
    LinUCB,
    ThompsonBernoulli,
    anytime_gamma,
    cumulative_pseudo_regret,
    lai_robbins_lower_bound,
)

DOCS = pathlib.Path(__file__).resolve().parent

# One palette for the whole documentation set.
INK = "#1d1f21"
GRID = "#d8dade"
COLORS = {
    "KL-UCB": "#1b3a6b",
    "Thompson": "#2a9d8f",
    "UCB1": "#e07a3f",
    "eps-greedy (0.1)": "#9aa0a6",
    "LinUCB": "#1b3a6b",
    "UCB1 (context-blind)": "#e07a3f",
}
FLOOR = "#c1121f"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "font.size": 9.5,
    })


def _sweep(probs, factories, horizon, runs):
    """Mean cumulative pseudo-regret per algorithm, averaged over `runs` seeds."""
    out = {}
    for name, factory in factories.items():
        acc = [0.0] * horizon
        for s in range(runs):
            env = BernoulliBandit(probs, seed=s)
            algo = factory(env.n_arms, s + 7919)
            arms = []
            for t in range(1, horizon + 1):
                a = algo.select(t)
                algo.update(a, env.pull(a))
                arms.append(a)
            for i, r in enumerate(cumulative_pseudo_regret(env, arms)):
                acc[i] += r
        out[name] = [v / runs for v in acc]
    return out


def _log_grid(horizon: int, start: int = 50, n: int = 260) -> list[int]:
    """Log-spaced round indices — a linear grid wastes 90% of its ink past 10k."""
    return sorted({int(round(x)) for x in np.geomspace(start, horizon, n)})


def fig_optimality(horizon: int = 200_000, runs: int = 20) -> None:
    """Regret per ln T against the Lai-Robbins constant, in two reward regimes."""
    instances = {
        "moderate gaps  K=5": [0.50, 0.45, 0.40, 0.35, 0.30],
        "rare rewards  K=10": [0.10, 0.05, 0.05, 0.05, 0.02,
                               0.02, 0.02, 0.01, 0.01, 0.01],
    }
    factories = {
        "KL-UCB": lambda k, s: KLUCB(k),
        "Thompson": lambda k, s: ThompsonBernoulli(k, seed=s),
        "UCB1": lambda k, s: UCB1(k),
        "eps-greedy (0.1)": lambda k, s: EpsilonGreedy(k, eps=0.1, seed=s),
    }
    checkpoints = [1_000, 10_000, 50_000, horizon]

    style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    grid = _log_grid(horizon)

    for ax, (label, probs) in zip(axes, instances.items(), strict=True):
        t0 = time.time()
        curves = _sweep(probs, factories, horizon, runs)
        floor = lai_robbins_lower_bound(BernoulliBandit(probs))
        print(f"\n{label}   Lai-Robbins constant C = {floor:.2f}"
              f"   ({runs} runs x {horizon:,} rounds, {time.time() - t0:.0f}s)")
        head = "".join(f"{f'R/lnT @{t // 1000}k' if t >= 1000 else t:>14}"
                       for t in checkpoints)
        print(f"{'policy':<18}{head}")
        for name, curve in curves.items():
            cells = "".join(f"{curve[t - 1] / math.log(t):>14.1f}"
                            for t in checkpoints)
            print(f"{name:<18}{cells}")
            ax.plot([t for t in grid],
                    [curve[t - 1] / math.log(t) for t in grid],
                    color=COLORS[name], lw=1.8, label=name)
        ax.axhline(floor, color=FLOOR, ls=(0, (5, 3)), lw=1.4)
        ax.annotate(f"Lai-Robbins floor  C = {floor:.1f}",
                    xy=(horizon, floor), xytext=(-6, 6),
                    textcoords="offset points", ha="right",
                    color=FLOOR, fontsize=8.5, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("rounds T")
        ax.set_title(label)

    axes[0].set_ylabel("cumulative regret / ln T")
    axes[0].legend(loc="upper left", ncol=2)
    fig.suptitle("Regret divided by ln T — a flat curve is logarithmic regret, "
                 "and the dashed line is the floor no policy can beat",
                 x=0.008, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(DOCS / "optimality.png", dpi=125, bbox_inches="tight")
    print(f"\nwrote {DOCS / 'optimality.png'}")


def fig_exp3(horizon: int = 100_000, runs: int = 10) -> None:
    """Fixed gamma vs the anytime schedule, well past the old overflow point."""
    probs = [0.25, 0.50, 0.75]
    variants = {
        "gamma = 0.2 (fixed)": ("#c1121f", 0.2),
        "gamma = 0.07 (fixed)": ("#e07a3f", 0.07),
        "anytime schedule": ("#1b3a6b", None),
    }
    curves = _sweep(probs, {n: (lambda k, s, g=g: EXP3(k, gamma=g, seed=s))
                            for n, (_, g) in variants.items()}, horizon, runs)

    print(f"\nK=3 Bernoulli {probs}, {runs} runs x {horizon:,} rounds")
    print(f"final anytime gamma_t = {anytime_gamma(3, horizon):.4f}")
    print(f"{'variant':<22}{'@10k':>10}{'@50k':>10}{'@100k':>10}")
    for name, curve in curves.items():
        print(f"{name:<22}" + "".join(f"{curve[t - 1]:>10.0f}"
                                      for t in (10_000, 50_000, 100_000)))

    style()
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    grid = list(range(1, horizon + 1, 50))
    for name, (color, _) in variants.items():
        ax.plot(grid, [curves[name][t - 1] for t in grid], color=color,
                lw=1.8, label=name)
    top = max(curves[n][-1] for n in variants)
    # Stop the marker below the legend: at full height the dashed rule runs
    # straight through all three legend labels.
    ax.axvline(9_000, ymax=0.70, color="#9aa0a6", ls=(0, (2, 3)), lw=1.2)
    ax.annotate("linear-space weights used to overflow here",
                xy=(9_000, top * 0.30), xytext=(-4, 0), rotation=90,
                textcoords="offset points", va="center", ha="right",
                color="#6b7075", fontsize=8.5)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.set_xlabel("rounds t")
    ax.set_ylabel("cumulative pseudo-regret")
    ax.set_title("EXP3 over 100,000 rounds — a fixed exploration rate never "
                 "stops paying for itself")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(DOCS / "exp3_longrun.png", dpi=125, bbox_inches="tight")
    print(f"\nwrote {DOCS / 'exp3_longrun.png'}")


def fig_contextual(horizon: int = 4_000, runs: int = 30, dim: int = 6,
                   n_arms: int = 5) -> None:
    """The problem no context-free policy can solve, and what LinUCB does to it."""
    lin = np.zeros((runs, horizon))
    blind = np.zeros((runs, horizon))
    for s in range(runs):
        theta = np.random.default_rng(1000 + s).standard_normal((n_arms, dim))
        env = LinearContextualBandit(theta, sigma=0.1, seed=s)
        algo = LinUCB(n_arms, dim, alpha=1.0)
        cum = 0.0
        for t in range(horizon):
            x = env.context()
            a = algo.select(x)
            algo.update(a, env.pull(a, x), x)
            cum += env.optimal_mean(x) - env.mean(a, x)
            lin[s, t] = cum
        # Same environment, a policy that never looks at the context.
        env = LinearContextualBandit(theta, sigma=0.1, seed=s)
        ucb = UCB1(n_arms)  # sees rewards, never sees x
        cum = 0.0
        for t in range(1, horizon + 1):
            x = env.context()
            a = ucb.select(t)
            r = env.pull(a, x)
            ucb.update(a, r)
            cum += env.optimal_mean(x) - env.mean(a, x)
            blind[s, t - 1] = cum

    lin_m, blind_m = lin.mean(0), blind.mean(0)
    print(f"\nLinear contextual bandit: {n_arms} arms, dim {dim}, "
          f"{runs} runs x {horizon:,} rounds")
    print(f"{'policy':<24}{'@500':>10}{'@2000':>10}{'@4000':>10}")
    for name, m in (("LinUCB", lin_m), ("UCB1 (context-blind)", blind_m)):
        print(f"{name:<24}" + "".join(f"{m[t - 1]:>10.1f}"
                                      for t in (500, 2_000, 4_000)))

    style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ts = np.arange(1, horizon + 1)
    for name, m, sd in (("LinUCB", lin_m, lin.std(0)),
                        ("UCB1 (context-blind)", blind_m, blind.std(0))):
        ax1.plot(ts, m, color=COLORS[name], lw=1.8, label=name)
        ax1.fill_between(ts, m - sd, m + sd, color=COLORS[name], alpha=0.12,
                         lw=0)
    ax1.set_xlabel("rounds t")
    ax1.set_ylabel("cumulative regret")
    ax1.set_title("Ignoring the context costs linear regret")
    ax1.legend(loc="upper left")

    sd = lin.std(0)
    ax2.plot(ts, lin_m, color=COLORS["LinUCB"], lw=1.8, label="LinUCB")
    ax2.fill_between(ts, lin_m - sd, lin_m + sd, color=COLORS["LinUCB"],
                     alpha=0.12, lw=0)
    ax2.set_xscale("log")
    ax2.set_xlabel("rounds t (log)")
    ax2.set_ylabel("cumulative regret")
    ax2.set_title(f"LinUCB stops paying: {lin_m[-1] - lin_m[499]:.1f} regret "
                  f"over the last 3,500 rounds")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(DOCS / "contextual.png", dpi=125, bbox_inches="tight")
    print(f"\nwrote {DOCS / 'contextual.png'}")


FIGURES = {"optimality": fig_optimality, "exp3": fig_exp3,
           "contextual": fig_contextual}


def main(argv: list[str]) -> int:
    names = argv[1:] or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        print(f"unknown figure(s): {', '.join(unknown)}; "
              f"choose from {', '.join(FIGURES)}")
        return 2
    for name in names:
        FIGURES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

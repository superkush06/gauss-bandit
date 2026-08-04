"""Where a bandit sits in a real pipeline: routing child orders to venues.

    python3 examples/execution_router.py
    python3 examples/execution_router.py --slices 40000 --runs 40

Nothing in this library knows what a basis point is, and that is the point:
the bandit is a middle layer. Something upstream decides *how much* to trade
and slices it; something downstream measures what the fills cost. In between
sits one question asked twenty thousand times — which venue for this slice? —
under feedback that only ever reports the venue you chose.

This script inlines a stand-in for both ends so it runs on its own:

    upstream   a VWAP-style schedule of equal child slices, plus a venue
               scorecard (queue depth, adverse selection) that determines each
               venue's true passive-fill rate. The router never sees it.
    here       KL-UCB / Thompson / eps-greedy allocating slices to venues.
    downstream shortfall in basis points, and a posterior over venues to hand
               the next parent order so it does not start from nothing.

The number worth stealing is the last one. `lai_robbins_lower_bound` converts
the scorecard into the cheapest possible price of finding out which venue is
best -- here 0.008 bp. The incumbent equal-split router burns 0.176 bp. That
splits a routing post-mortem cleanly in two: 0.008 bp was never recoverable,
and the other 0.169 bp is a defect in the allocation, not a fact about the
market. Very few execution reports can draw that line.
"""

from __future__ import annotations

import argparse
import math

from bandit import (
    KLUCB,
    BernoulliBandit,
    EpsilonGreedy,
    ThompsonBernoulli,
    lai_robbins_lower_bound,
)

# Half the quoted spread, in basis points. A slice that rests and gets filled
# passively saves this; one that crosses pays it.
HALF_SPREAD_BP = 1.4

# --- upstream: the venue scorecard the router does not get to see -----------
# depth  = resting size at the touch, relative to a child slice
# tox    = fraction of resting volume that is adversely selected
VENUES = {
    "primary":   {"depth": 3.2, "tox": 0.24},
    "inverted":  {"depth": 1.8, "tox": 0.11},
    "midpoint":  {"depth": 0.9, "tox": 0.06},
    "retail-mm": {"depth": 2.4, "tox": 0.35},
}


def fill_rate(depth: float, tox: float) -> float:
    """Probability a resting child slice fills before the quote moves.

    A queue you are near the front of fills often; a queue full of informed
    flow fills you only when you are about to be wrong. The functional form
    is a stand-in -- what matters downstream is that it produces four rates
    the router has to discover, with gaps small enough that discovering them
    costs real money.
    """
    return round((1.0 - math.exp(-depth / 2.2)) * (1.0 - tox), 3)


def scorecard() -> tuple[list[str], list[float]]:
    names = list(VENUES)
    return names, [fill_rate(**VENUES[v]) for v in names]


# --- the routers ------------------------------------------------------------

def route(policy, env, slices: int) -> list[int]:
    """Send `slices` child orders through `policy`, one venue per slice."""
    picks = []
    for t in range(1, slices + 1):
        v = policy.select(t)
        policy.update(v, env.pull(v))
        picks.append(v)
    return picks


class EqualSplit:
    """The status quo: rotate through every venue forever."""

    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        self.n_arms = n_arms

    def select(self, t: int) -> int:
        return (t - 1) % self.n_arms

    def update(self, arm: int, reward: float) -> None:
        pass

    def reset(self) -> None:
        pass


class PilotThenCommit:
    """Run a fixed A/B pilot across all venues, then send everything to the
    winner. This is what a routing decision looks like when it is made by a
    two-week experiment instead of by a policy, and it is the baseline the
    adaptive methods have to beat."""

    def __init__(self, n_arms: int, pilot: int = 2_000, seed: int | None = None) -> None:
        self.n_arms = n_arms
        self.pilot = pilot
        self.reset()

    def reset(self) -> None:
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms
        self._winner: int | None = None

    def select(self, t: int) -> int:
        if t <= self.pilot:
            return (t - 1) % self.n_arms
        if self._winner is None:
            self._winner = max(range(self.n_arms), key=lambda i: self.values[i])
        return self._winner

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, default=20_000,
                    help="child orders in the parent")
    ap.add_argument("--runs", type=int, default=20, help="replications")
    ap.add_argument("--notional", type=float, default=250e6,
                    help="parent order notional, for the dollar column")
    args = ap.parse_args()

    names, rates = scorecard()
    best = max(range(len(rates)), key=lambda i: rates[i])
    floor_c = lai_robbins_lower_bound(BernoulliBandit(rates))

    print(f"parent order: {args.slices:,} child slices, "
          f"${args.notional / 1e6:,.0f}M notional, half-spread "
          f"{HALF_SPREAD_BP:.1f} bp")
    print(f"{'venue':<12}{'depth':>8}{'toxicity':>10}{'fill rate':>12}")
    for i, v in enumerate(names):
        star = "  <- best" if i == best else ""
        print(f"{v:<12}{VENUES[v]['depth']:>8.1f}{VENUES[v]['tox']:>10.2f}"
              f"{rates[i]:>12.3f}{star}")

    routers = {
        "equal split":        lambda k, s: EqualSplit(k),
        "pilot 2k -> commit": lambda k, s: PilotThenCommit(k, pilot=2_000),
        "eps-greedy (0.05)":  lambda k, s: EpsilonGreedy(k, eps=0.05, seed=s),
        "Thompson":           lambda k, s: ThompsonBernoulli(k, seed=s),
        "KL-UCB":             lambda k, s: KLUCB(k),
    }

    def bp_of(fills_missed: float) -> float:
        """Missed passive fills -> basis points. Each one crosses the spread."""
        return fills_missed * HALF_SPREAD_BP / args.slices

    print(f"\nshortfall against an oracle that always routes to "
          f"{names[best]}  ({args.runs} runs)")
    print(f"{'router':<20}{'fill rate':>11}{'best venue':>12}"
          f"{'shortfall':>14}{'$ on notional':>16}")
    posterior, shortfall = None, {}
    for label, factory in routers.items():
        missed, on_best = 0.0, 0
        for seed in range(args.runs):
            env = BernoulliBandit(rates, seed=seed)
            policy = factory(len(rates), seed + 7919)
            picks = route(policy, env, args.slices)
            missed += sum(rates[best] - rates[v] for v in picks)
            on_best += sum(1 for v in picks if v == best)
            if label == "Thompson" and seed == 0:
                posterior = (list(policy.alphas), list(policy.betas))
        missed /= args.runs
        bp = shortfall[label] = bp_of(missed)
        print(f"{label:<20}{rates[best] - missed / args.slices:>11.3f}"
              f"{100 * on_best / (args.runs * args.slices):>11.1f}%"
              f"{bp:>12.3f} bp{bp * args.notional / 1e4:>16,.0f}")

    floor_fills = floor_c * math.log(args.slices)
    floor_bp = bp_of(floor_fills)
    print(f"\nLai-Robbins floor   C = {floor_c:.2f}, C ln T = {floor_fills:.0f} "
          f"missed fills = {floor_bp:.3f} bp "
          f"(${floor_bp * args.notional / 1e4:,.0f})")
    waste = shortfall["equal split"] - floor_bp
    under = sum(1 for bp in shortfall.values() if bp < floor_bp)
    print(f"  Of equal split's {shortfall['equal split']:.3f} bp, at most "
          f"{floor_bp:.3f} bp was ever unavoidable;")
    print(f"  the remaining {waste:.3f} bp "
          f"(${waste * args.notional / 1e4:,.0f}) is allocation waste. The")
    print("  adaptive routers sit at the same order as the floor, so there is")
    print("  no meaningful money left in the venue choice -- go and look at")
    print("  the schedule instead.")
    print(f"  {under} of them come in slightly *under* the floor. That is")
    print("  finite-T, not a violated theorem: at this horizon the o(ln T)")
    print("  term has not washed out, and docs/validation.md measures how far")
    print("  short of the asymptote each policy still is.")

    if posterior is not None:
        alphas, betas = posterior
        print("\nhanded downstream (Thompson posterior after one parent order)")
        print(f"{'venue':<12}{'E[fill]':>10}{'sd':>9}{'truth':>9}")
        for i, v in enumerate(names):
            a, b = alphas[i], betas[i]
            mean = a / (a + b)
            sd = math.sqrt(mean * (1 - mean) / (a + b + 1))
            print(f"{v:<12}{mean:>10.3f}{sd:>9.3f}{rates[i]:>9.3f}")
        print("  The next order starts from this, not from a uniform prior --")
        print("  which is the whole reason to keep the allocator stateful.")


if __name__ == "__main__":
    main()

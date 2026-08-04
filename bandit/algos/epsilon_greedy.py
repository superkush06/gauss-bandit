"""Epsilon-greedy: explore uniformly with prob eps, otherwise exploit."""

from __future__ import annotations

import random
from collections.abc import Callable

from .base import Algorithm


class EpsilonGreedy(Algorithm):
    """With probability `eps`, pick a uniformly random arm; otherwise the
    arm with the highest empirical mean (ties broken uniformly at random).

    Every arm is pulled once before the eps/greedy rule kicks in — without
    that round-robin, eps=0 (or a small eps early on) can lock onto arm 0
    before ever observing the others.

    Simple, fundamental, but achieves only linear regret unless `eps_t` is
    annealed (see `annealed`).
    """

    def __init__(self, n_arms: int,
                 eps: float | Callable[[int], float] = 0.1,
                 seed: int | None = None) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        self.n_arms = n_arms
        self.eps_fn = eps if callable(eps) else (lambda t, e=eps: e)
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms

    def select(self, t: int) -> int:
        for arm, n in enumerate(self.counts):
            if n == 0:
                return arm
        if self._rng.random() < self.eps_fn(t):
            return self._rng.randrange(self.n_arms)
        best = max(self.values)
        ties = [i for i, v in enumerate(self.values) if v == best]
        return ties[0] if len(ties) == 1 else self._rng.choice(ties)

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        # incremental mean update
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


def annealed(c: float = 1.0):
    """Classic 1/t-style schedule: eps_t = min(1, c/t)."""
    def fn(t: int) -> float:
        return min(1.0, c / max(t, 1))
    return fn

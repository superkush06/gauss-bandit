"""Epsilon-greedy: explore uniformly with prob eps, otherwise exploit."""

from __future__ import annotations

import random

from .base import Algorithm


class EpsilonGreedy(Algorithm):
    """With probability `eps`, pick a uniformly random arm; otherwise the
    arm with the highest empirical mean.

    Simple, fundamental, but achieves only linear regret unless `eps_t` is
    annealed (see `eps_schedule`).
    """

    def __init__(self, n_arms: int, eps: float | callable = 0.1,
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
        eps = self.eps_fn(t)
        if self._rng.random() < eps:
            return self._rng.randrange(self.n_arms)
        return max(range(self.n_arms), key=lambda i: self.values[i])

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

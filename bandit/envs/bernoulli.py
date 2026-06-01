"""Bernoulli bandit: each arm yields 0/1 with arm-specific probability."""

from __future__ import annotations

import random

from .base import BanditEnv


class BernoulliBandit(BanditEnv):
    """K Bernoulli arms with success probabilities `probs`."""

    def __init__(self, probs: list[float], seed: int | None = None) -> None:
        if not probs:
            raise ValueError("probs must be non-empty")
        if any(not 0.0 <= p <= 1.0 for p in probs):
            raise ValueError("each prob must be in [0, 1]")
        self.probs = list(probs)
        self.n_arms = len(probs)
        self.optimal_arm = max(range(self.n_arms), key=lambda i: probs[i])
        self.optimal_mean = probs[self.optimal_arm]
        self._rng = random.Random(seed)

    def pull(self, arm: int) -> float:
        return 1.0 if self._rng.random() < self.probs[arm] else 0.0

    def mean(self, arm: int) -> float:
        return self.probs[arm]

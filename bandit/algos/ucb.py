"""UCB1 (Auer, Cesa-Bianchi, Fischer 2002).

Picks arm maximizing  mean_i + sqrt(2 * ln(t) / n_i).
Achieves O(sqrt(KT log T)) regret on bounded rewards.
"""

from __future__ import annotations

import math

from .base import Algorithm


class UCB1(Algorithm):
    """Optimism-in-the-face-of-uncertainty index policy.

    Each arm has an upper confidence bound on its mean; we always pick the
    arm whose UCB is highest. Untried arms are pulled first.
    """

    def __init__(self, n_arms: int, c: float = math.sqrt(2.0)) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        self.n_arms = n_arms
        self.c = c
        self.reset()

    def reset(self) -> None:
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms
        self._total_pulls = 0

    def select(self, t: int) -> int:
        # Pull each arm once first
        for arm, n in enumerate(self.counts):
            if n == 0:
                return arm
        log_total = math.log(self._total_pulls)
        best_arm, best_ucb = 0, -math.inf
        for arm in range(self.n_arms):
            ucb = self.values[arm] + self.c * math.sqrt(log_total / self.counts[arm])
            if ucb > best_ucb:
                best_arm, best_ucb = arm, ucb
        return best_arm

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self._total_pulls += 1

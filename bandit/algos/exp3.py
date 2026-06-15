"""EXP3 — adversarial bandits (Auer, Cesa-Bianchi, Freund, Schapire 2002).

Maintains a weight per arm; samples proportionally to weights. Updates use
importance-weighted reward estimates.
"""

from __future__ import annotations

import math
import random

from .base import Algorithm


class EXP3(Algorithm):
    """EXP3 for adversarial bandits, rewards in [0, 1].

    With learning rate gamma in (0, 1], the algorithm picks arm i with
    probability  p_i = (1-gamma) * w_i/sum(w) + gamma/K. Empirically robust
    even on stationary stochastic bandits.
    """

    def __init__(self, n_arms: int, gamma: float = 0.1,
                 seed: int | None = None) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.n_arms = n_arms
        self.gamma = gamma
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.weights = [1.0] * self.n_arms

    def _probs(self) -> list[float]:
        total = sum(self.weights)
        return [(1.0 - self.gamma) * (w / total) + self.gamma / self.n_arms
                for w in self.weights]

    def select(self, t: int) -> int:
        probs = self._probs()
        r = self._rng.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return i
        return self.n_arms - 1

    def update(self, arm: int, reward: float) -> None:
        if not 0.0 <= reward <= 1.0:
            raise ValueError("EXP3 expects reward in [0, 1]")
        p = self._probs()[arm]
        x_hat = reward / p
        self.weights[arm] *= math.exp(self.gamma * x_hat / self.n_arms)

"""Gaussian bandit: each arm yields N(mu_i, sigma_i^2)."""

from __future__ import annotations

import random

from .base import BanditEnv


class GaussianBandit(BanditEnv):
    """K Gaussian arms with means `mus` and standard deviations `sigmas`."""

    def __init__(self, mus: list[float], sigmas: list[float] | None = None,
                 seed: int | None = None) -> None:
        if not mus:
            raise ValueError("mus must be non-empty")
        if sigmas is None:
            sigmas = [1.0] * len(mus)
        if len(sigmas) != len(mus):
            raise ValueError("len(sigmas) must equal len(mus)")
        if any(s <= 0 for s in sigmas):
            raise ValueError("all sigmas must be positive")
        self.mus = list(mus)
        self.sigmas = list(sigmas)
        self.n_arms = len(mus)
        self.optimal_arm = max(range(self.n_arms), key=lambda i: mus[i])
        self.optimal_mean = mus[self.optimal_arm]
        self._rng = random.Random(seed)

    def pull(self, arm: int) -> float:
        return self._rng.gauss(self.mus[arm], self.sigmas[arm])

    def mean(self, arm: int) -> float:
        return self.mus[arm]

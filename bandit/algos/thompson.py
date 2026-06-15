"""Thompson sampling — Bayesian posterior sampling.

Two flavours:
  - `ThompsonBernoulli`: Beta(alpha, beta) conjugate posterior for 0/1 rewards.
  - `ThompsonGaussian`: Normal-known-variance posterior for real rewards.
"""

from __future__ import annotations

import random

from .base import Algorithm


class ThompsonBernoulli(Algorithm):
    """Beta-Bernoulli Thompson sampling.

    Posterior for arm i after s_i successes and f_i failures is Beta(1+s_i, 1+f_i).
    Each step, sample one mu_i from each posterior and pick argmax.
    """

    def __init__(self, n_arms: int, seed: int | None = None,
                 prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        self.n_arms = n_arms
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.alphas = [self.prior_alpha] * self.n_arms
        self.betas = [self.prior_beta] * self.n_arms

    def select(self, t: int) -> int:
        samples = [self._rng.betavariate(self.alphas[i], self.betas[i])
                   for i in range(self.n_arms)]
        return max(range(self.n_arms), key=lambda i: samples[i])

    def update(self, arm: int, reward: float) -> None:
        if not 0.0 <= reward <= 1.0:
            raise ValueError("Beta-Bernoulli expects reward in [0, 1]")
        self.alphas[arm] += reward
        self.betas[arm] += 1.0 - reward


class ThompsonGaussian(Algorithm):
    """Gaussian Thompson with known observation variance.

    Posterior mean for arm i: weighted by precision = 1/sigma_obs^2.
    Given prior N(mu_0, sigma_0^2), after n observations with mean x_bar,
    posterior is N(mu_n, sigma_n^2) with
      precision_n = 1/sigma_0^2 + n/sigma_obs^2
      mu_n = (mu_0/sigma_0^2 + n*x_bar/sigma_obs^2) / precision_n
    """

    def __init__(self, n_arms: int, sigma_obs: float = 1.0,
                 mu_0: float = 0.0, sigma_0: float = 100.0,
                 seed: int | None = None) -> None:
        if sigma_obs <= 0 or sigma_0 <= 0:
            raise ValueError("sigmas must be positive")
        self.n_arms = n_arms
        self.sigma_obs = sigma_obs
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.counts = [0] * self.n_arms
        self.sums = [0.0] * self.n_arms

    def _posterior(self, arm: int) -> tuple[float, float]:
        n = self.counts[arm]
        x_bar = self.sums[arm] / n if n > 0 else 0.0
        prec_n = 1.0 / (self.sigma_0 ** 2) + n / (self.sigma_obs ** 2)
        mu_n = ((self.mu_0 / self.sigma_0 ** 2) +
                (n * x_bar) / (self.sigma_obs ** 2)) / prec_n
        sigma_n = (1.0 / prec_n) ** 0.5
        return mu_n, sigma_n

    def select(self, t: int) -> int:
        samples = []
        for arm in range(self.n_arms):
            mu_n, sigma_n = self._posterior(arm)
            samples.append(self._rng.gauss(mu_n, sigma_n))
        return max(range(self.n_arms), key=lambda i: samples[i])

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.sums[arm] += reward

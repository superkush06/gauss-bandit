"""Contextual bandits: LinUCB (Li et al. 2010) on a linear-reward environment.

The bandits in `bandit.algos` are *context-free*: every pull of arm i draws
from the same fixed distribution. Real applications (news/ad recommendation,
clinical dosing) get a **context** x each round and the best arm depends on x.

LinUCB models each arm's reward as linear in the context, E[r | a, x] = θ_aᵀx,
and plays optimism-in-the-face-of-uncertainty in that linear model:

    p_a = θ̂_aᵀx + α · sqrt(xᵀ A_a⁻¹ x),     pick argmax_a p_a

where A_a = I + Σ xxᵀ over arm a's history and θ̂_a = A_a⁻¹ b_a. The second
term is the confidence width — large when x points in a poorly-explored
direction for arm a. This is the "disjoint" LinUCB (one model per arm).
"""

from __future__ import annotations

import numpy as np


class LinearContextualBandit:
    """Linear contextual environment: reward = θ_aᵀx + Gaussian noise.

    `theta` is (n_arms, dim). Each round a fresh standard-normal context is
    drawn; `optimal_mean` gives the best achievable expected reward for it.
    """

    def __init__(self, theta, sigma: float = 0.1, seed: int | None = None) -> None:
        self.theta = np.asarray(theta, dtype=float)
        if self.theta.ndim != 2:
            raise ValueError("theta must be (n_arms, dim)")
        self.n_arms, self.dim = self.theta.shape
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    def context(self) -> np.ndarray:
        return self._rng.standard_normal(self.dim)

    def pull(self, arm: int, x: np.ndarray) -> float:
        return float(x @ self.theta[arm] + self._rng.normal(0.0, self.sigma))

    def mean(self, arm: int, x: np.ndarray) -> float:
        return float(x @ self.theta[arm])

    def optimal_mean(self, x: np.ndarray) -> float:
        return float(np.max(self.theta @ x))


class LinUCB:
    """Disjoint LinUCB. `alpha` controls exploration (confidence width)."""

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0) -> None:
        if n_arms < 1 or dim < 1:
            raise ValueError("n_arms and dim must be >= 1")
        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self.A = [np.eye(self.dim) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.dim) for _ in range(self.n_arms)]

    def select(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float)
        best_arm, best_p = 0, -np.inf
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p = theta @ x + self.alpha * np.sqrt(max(x @ A_inv @ x, 0.0))
            if p > best_p:
                best_arm, best_p = a, p
        return best_arm

    def update(self, arm: int, reward: float, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


def run_contextual(env: LinearContextualBandit, algo: LinUCB,
                   horizon: int) -> list[float]:
    """Run `algo` on `env` for `horizon` steps; return cumulative regret."""
    cum = 0.0
    regret = []
    for _ in range(horizon):
        x = env.context()
        a = algo.select(x)
        r = env.pull(a, x)
        algo.update(a, r, x)
        cum += env.optimal_mean(x) - env.mean(a, x)
        regret.append(cum)
    return regret

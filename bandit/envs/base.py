"""Bandit environments."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BanditEnv(ABC):
    """Stationary K-armed bandit environment.

    `pull(arm)` returns a stochastic reward. `n_arms` is the number of arms.
    `optimal_arm` and `optimal_mean` are used to compute regret.
    """

    n_arms: int
    optimal_arm: int
    optimal_mean: float

    @abstractmethod
    def pull(self, arm: int) -> float:
        ...

    @abstractmethod
    def mean(self, arm: int) -> float:
        """Return the true mean reward of `arm` (oracle access — for regret)."""
        ...

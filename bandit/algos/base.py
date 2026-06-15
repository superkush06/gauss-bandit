"""Bandit algorithms — base interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Algorithm(ABC):
    """Sequential decision-maker for the K-armed bandit problem."""

    n_arms: int

    @abstractmethod
    def select(self, t: int) -> int:
        """Return the arm to pull at step `t` (zero-indexed)."""

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """Record observed reward for `arm`."""

    @abstractmethod
    def reset(self) -> None:
        """Clear state so the algorithm can be replayed."""

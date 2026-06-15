"""Bandit algorithms."""

from .base import Algorithm
from .epsilon_greedy import EpsilonGreedy, annealed
from .thompson import ThompsonBernoulli, ThompsonGaussian
from .ucb import UCB1

__all__ = ["Algorithm", "EpsilonGreedy", "annealed", "UCB1",
           "ThompsonBernoulli", "ThompsonGaussian"]

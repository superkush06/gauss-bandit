"""gauss-bandit: multi-armed bandits library."""

from .algos import (
    EXP3,
    UCB1,
    EpsilonGreedy,
    ThompsonBernoulli,
    ThompsonGaussian,
    annealed,
)
from .envs import BanditEnv, BernoulliBandit, GaussianBandit
from .metrics import cumulative_pseudo_regret, cumulative_regret, lai_robbins_lower_bound
from .runner import ExperimentResult, RunResult, run_experiment, run_one

__version__ = "0.1.0"
__all__ = [
    "BanditEnv", "BernoulliBandit", "GaussianBandit",
    "EpsilonGreedy", "UCB1", "ThompsonBernoulli", "ThompsonGaussian", "EXP3",
    "annealed",
    "cumulative_regret", "cumulative_pseudo_regret", "lai_robbins_lower_bound",
    "run_one", "run_experiment", "RunResult", "ExperimentResult",
    "__version__",
]

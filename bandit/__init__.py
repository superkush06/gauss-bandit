"""gauss-bandit: multi-armed bandits library."""

from .algos import (
    EXP3,
    UCB1,
    EpsilonGreedy,
    ThompsonBernoulli,
    ThompsonGaussian,
    annealed,
)
from .contextual import LinearContextualBandit, LinUCB, run_contextual
from .envs import BanditEnv, BernoulliBandit, GaussianBandit
from .metrics import cumulative_pseudo_regret, cumulative_regret, lai_robbins_lower_bound
from .runner import ExperimentResult, RunResult, run_experiment, run_one

__version__ = "0.2.0"
__all__ = [
    "BanditEnv", "BernoulliBandit", "GaussianBandit",
    "EpsilonGreedy", "UCB1", "ThompsonBernoulli", "ThompsonGaussian", "EXP3",
    "annealed",
    "LinUCB", "LinearContextualBandit", "run_contextual",
    "cumulative_regret", "cumulative_pseudo_regret", "lai_robbins_lower_bound",
    "run_one", "run_experiment", "RunResult", "ExperimentResult",
    "__version__",
]

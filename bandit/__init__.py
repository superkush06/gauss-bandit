"""gauss-bandit: multi-armed bandits library."""

from .algos import (
    EXP3,
    KLUCB,
    UCB1,
    EpsilonGreedy,
    ThompsonBernoulli,
    ThompsonGaussian,
    annealed,
    anytime_gamma,
    klucb_index,
)
from .contextual import LinearContextualBandit, LinUCB, run_contextual
from .envs import BanditEnv, BernoulliBandit, GaussianBandit
from .metrics import (
    bernoulli_kl,
    cumulative_pseudo_regret,
    cumulative_regret,
    gaussian_kl,
    lai_robbins_lower_bound,
)
from .runner import ExperimentResult, RunResult, run_experiment, run_one

__version__ = "0.5.3"
__all__ = [
    "BanditEnv", "BernoulliBandit", "GaussianBandit",
    "EpsilonGreedy", "UCB1", "KLUCB", "ThompsonBernoulli", "ThompsonGaussian", "EXP3",
    "klucb_index",
    "annealed", "anytime_gamma",
    "LinUCB", "LinearContextualBandit", "run_contextual",
    "cumulative_regret", "cumulative_pseudo_regret", "lai_robbins_lower_bound",
    "bernoulli_kl", "gaussian_kl",
    "run_one", "run_experiment", "RunResult", "ExperimentResult",
    "__version__",
]

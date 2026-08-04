"""Regret-metric tests."""

import math

import pytest

from bandit.envs import BernoulliBandit, GaussianBandit
from bandit.metrics import (
    bernoulli_kl,
    cumulative_pseudo_regret,
    cumulative_regret,
    gaussian_kl,
    lai_robbins_lower_bound,
)


def test_cumulative_pseudo_regret_zero_if_optimal_always():
    env = BernoulliBandit([0.2, 0.8], seed=0)
    # always pulling optimal arm (1) -> zero pseudo regret
    arms = [1] * 10
    pr = cumulative_pseudo_regret(env, arms)
    assert pr == [0.0] * 10


def test_cumulative_pseudo_regret_grows_on_suboptimal():
    env = BernoulliBandit([0.2, 0.8], seed=0)
    arms = [0] * 5
    pr = cumulative_pseudo_regret(env, arms)
    assert pr[-1] == pytest.approx(5 * 0.6)


def test_cumulative_regret_uses_observed_rewards():
    env = BernoulliBandit([0.5], seed=0)
    rewards = [1.0, 1.0, 1.0, 0.0, 0.0]
    cr = cumulative_regret(env, [0]*5, rewards)
    # optimal_mean=0.5, t=5 -> 2.5; cum reward = 3.0; regret = -0.5
    assert cr[-1] == pytest.approx(2.5 - 3.0)


def test_lai_robbins_is_finite_and_positive():
    env = BernoulliBandit([0.3, 0.5, 0.9], seed=0)
    c = lai_robbins_lower_bound(env)
    assert math.isfinite(c)
    assert c > 0


def test_bernoulli_kl_edges():
    assert bernoulli_kl(0.5, 0.5) == 0.0
    assert bernoulli_kl(0.5, 1.0) == math.inf
    assert bernoulli_kl(0.5, 0.0) == math.inf
    assert bernoulli_kl(0.0, 0.9) == pytest.approx(math.log(1 / 0.1))
    assert bernoulli_kl(1.0, 0.9) == pytest.approx(math.log(1 / 0.9))
    with pytest.raises(ValueError):
        bernoulli_kl(1.2, 0.5)


def test_lai_robbins_gaussian_uses_gaussian_kl():
    """Regression: Gaussian envs with means inside (0,1) used to be scored
    with Bernoulli KL, understating the bound ~6x for unit-variance arms."""
    env = GaussianBandit([0.3, 0.5, 0.9], sigmas=[1.0, 1.0, 1.0], seed=0)
    # C = sum 2 sigma^2 / gap = 2/0.6 + 2/0.4
    assert lai_robbins_lower_bound(env) == pytest.approx(2 / 0.6 + 2 / 0.4)


def test_lai_robbins_gaussian_respects_arm_sigmas():
    """Regression: arm sigmas were ignored entirely — sigma=5 arms are 25x
    harder to distinguish and the constant must scale accordingly."""
    lo = lai_robbins_lower_bound(GaussianBandit([0.3, 0.9], sigmas=[1.0, 1.0]))
    hi = lai_robbins_lower_bound(GaussianBandit([0.3, 0.9], sigmas=[5.0, 1.0]))
    assert hi == pytest.approx(25 * lo)
    assert lo == pytest.approx(2 / 0.6)


def test_lai_robbins_bernoulli_certain_optimum_is_zero():
    """Regression: p* = 1.0 fell into a quadratic fallback and returned 4.0;
    the true KL is infinite (one failure identifies a suboptimal arm), so the
    log-regret constant is 0."""
    assert lai_robbins_lower_bound(BernoulliBandit([0.5, 1.0])) == 0.0


def test_lai_robbins_bernoulli_zero_prob_arm():
    """Regression: p_i = 0.0 also fell into the fallback (returned 2.22);
    the exact constant is gap / KL(0 || 0.9) = 0.9 / ln 10."""
    c = lai_robbins_lower_bound(BernoulliBandit([0.0, 0.9]))
    assert c == pytest.approx(0.9 / math.log(10))


def test_lai_robbins_custom_kl_and_unknown_env():
    env = BernoulliBandit([0.4, 0.8])

    class Opaque:
        n_arms = env.n_arms
        optimal_mean = env.optimal_mean

        def mean(self, arm):
            return env.mean(arm)

    with pytest.raises(TypeError):
        lai_robbins_lower_bound(Opaque())
    c = lai_robbins_lower_bound(
        Opaque(), kl=lambda arm: bernoulli_kl(env.mean(arm), env.optimal_mean))
    assert c == pytest.approx(lai_robbins_lower_bound(env))


def test_gaussian_kl_formula():
    assert gaussian_kl(0.0, 1.0, 1.0) == pytest.approx(0.5)
    assert gaussian_kl(0.0, 1.0, 2.0) == pytest.approx(0.125)
    with pytest.raises(ValueError):
        gaussian_kl(0.0, 1.0, 0.0)

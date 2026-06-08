"""Environment unit tests."""

import pytest

from bandit.envs import BernoulliBandit, GaussianBandit


def test_bernoulli_optimal_arm():
    env = BernoulliBandit([0.1, 0.5, 0.9], seed=0)
    assert env.n_arms == 3
    assert env.optimal_arm == 2
    assert env.optimal_mean == 0.9


def test_bernoulli_deterministic_under_seed():
    a = BernoulliBandit([0.1, 0.9], seed=42)
    b = BernoulliBandit([0.1, 0.9], seed=42)
    sa = [a.pull(0) for _ in range(50)]
    sb = [b.pull(0) for _ in range(50)]
    assert sa == sb


def test_bernoulli_rejects_invalid_probs():
    with pytest.raises(ValueError):
        BernoulliBandit([])
    with pytest.raises(ValueError):
        BernoulliBandit([0.5, 1.5])
    with pytest.raises(ValueError):
        BernoulliBandit([0.5, -0.1])


def test_gaussian_optimal_arm():
    env = GaussianBandit(mus=[0.0, 1.0, 0.5], sigmas=[1, 1, 1], seed=0)
    assert env.optimal_arm == 1
    assert env.optimal_mean == 1.0


def test_gaussian_deterministic_under_seed():
    a = GaussianBandit([0.0, 1.0], seed=7)
    b = GaussianBandit([0.0, 1.0], seed=7)
    assert [a.pull(0) for _ in range(20)] == [b.pull(0) for _ in range(20)]


def test_gaussian_default_sigmas():
    env = GaussianBandit(mus=[0.0, 1.0])
    assert env.sigmas == [1.0, 1.0]


def test_gaussian_rejects_bad_sigmas():
    with pytest.raises(ValueError):
        GaussianBandit(mus=[0.0, 1.0], sigmas=[0, 1])
    with pytest.raises(ValueError):
        GaussianBandit(mus=[0.0, 1.0], sigmas=[1.0])

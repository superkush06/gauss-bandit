"""Randomized property tests: laws that must hold for every valid input.

The rest of the suite checks fixtures — this instance, that horizon, this
expected number. These tests instead draw hundreds of random instances,
horizons, dimensions and reward streams from a seeded NumPy generator and
assert the *algebra*: probabilities that sum to one, divergences that are
non-negative, design matrices that stay positive definite, regret that
telescopes into the gaps that produced it.

Everything here is deterministic: `rng(name)` derives a stream from a fixed
root seed and the test's own name, so a failure is reproducible from the
report alone and adding a test never perturbs its neighbours.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import pytest

from bandit import (
    EXP3,
    KLUCB,
    UCB1,
    BernoulliBandit,
    EpsilonGreedy,
    GaussianBandit,
    LinearContextualBandit,
    LinUCB,
    ThompsonBernoulli,
    anytime_gamma,
    bernoulli_kl,
    cumulative_pseudo_regret,
    gaussian_kl,
    klucb_index,
    lai_robbins_lower_bound,
    run_experiment,
)

ROOT_SEED = 0x6AB1


def rng(name: str) -> np.random.Generator:
    """A per-test generator: same root seed, independent stream per test.

    `crc32`, not `hash`: string hashing is salted per interpreter process, so
    a seed derived from it would change between runs and quietly turn every
    test below into a different test each time CI starts.
    """
    return np.random.default_rng([ROOT_SEED, zlib.crc32(name.encode())])


DRAWS = 300


# --------------------------------------------------------------------------
# Divergences
# --------------------------------------------------------------------------

def test_bernoulli_kl_is_a_divergence():
    """d(p, q) >= 0 with equality iff p == q — the defining property of a KL.

    A negative or spuriously-zero divergence would make the Lai-Robbins
    constant infinite or negative, so this is the load-bearing invariant of
    the whole metrics module.
    """
    r = rng("bernoulli_kl_divergence")
    for _ in range(DRAWS):
        p, q = r.uniform(0.001, 0.999, size=2)
        d = bernoulli_kl(float(p), float(q))
        assert d >= 0.0
        assert (d == 0.0) == (p == q)
        assert bernoulli_kl(float(p), float(p)) == 0.0


def test_bernoulli_kl_dominates_pinsker():
    """d(p, q) >= 2 (p - q)^2 for all p, q — Pinsker's inequality.

    This is the exact sense in which KL-UCB's index is never looser than
    UCB1's Hoeffding bonus; if it ever failed, the claim that the KL index
    is the tighter of the two would be false.
    """
    r = rng("pinsker")
    for _ in range(DRAWS):
        p, q = r.uniform(0.0, 1.0, size=2)
        d = bernoulli_kl(float(p), float(q))
        if math.isfinite(d):
            assert d >= 2.0 * (p - q) ** 2 - 1e-12


def test_bernoulli_kl_is_label_symmetric():
    """d(p, q) == d(1-p, 1-q): relabelling success as failure cannot change
    how distinguishable two coins are."""
    r = rng("kl_label_symmetry")
    for _ in range(DRAWS):
        p, q = r.uniform(0.01, 0.99, size=2)
        assert bernoulli_kl(float(p), float(q)) == pytest.approx(
            bernoulli_kl(1.0 - float(p), 1.0 - float(q)))


def test_gaussian_kl_scales_quadratically_with_sigma():
    """KL(N(a, s^2) || N(b, s^2)) is symmetric in (a, b) and scales as
    1/s^2, so widening every arm by c multiplies the divergence by 1/c^2.

    The Lai-Robbins constant inherits that scaling; it is what makes a
    five-times-noisier arm twenty-five times more expensive to rule out.
    """
    r = rng("gaussian_kl_scaling")
    for _ in range(DRAWS):
        a, b = r.normal(size=2)
        s, c = r.uniform(0.1, 5.0), r.uniform(0.5, 4.0)
        base = gaussian_kl(float(a), float(b), float(s))
        assert base == pytest.approx(gaussian_kl(float(b), float(a), float(s)))
        assert gaussian_kl(float(a), float(b), float(s * c)) == pytest.approx(
            base / c ** 2)


# --------------------------------------------------------------------------
# The KL-UCB index
# --------------------------------------------------------------------------

def test_klucb_index_brackets_the_root_to_its_stated_tolerance():
    """klucb_index(p, L) lies in [p, 1] and brackets the root of
    d(p, u) = L to within the solver's 1e-6 tolerance on u.

    The contract is on the *u* axis, not on the divergence: near u = 1 the KL
    is near-vertical, so a root located to 1e-6 in u can still be off by a
    lot in d. Asserting the wrong one of those two makes the test either
    vacuous or permanently red.
    """
    r = rng("klucb_inversion")
    tol = 2e-6
    for _ in range(DRAWS):
        p = float(r.uniform(0.0, 0.999))
        level = float(r.uniform(1e-4, 2.0))
        u = klucb_index(p, level)
        assert p <= u <= 1.0
        below = max(p, u - tol)
        assert bernoulli_kl(p, below) <= level + 1e-12
        if u < 1.0 - tol:
            assert bernoulli_kl(p, u + tol) >= level - 1e-12


def test_klucb_index_matches_the_closed_form_at_p_zero():
    """d(0, u) = -ln(1 - u), so the index at an arm that has never paid out is
    exactly 1 - exp(-L). That is an analytic root, not another bisection, and
    it pins the solver against ground truth rather than against itself."""
    r = rng("klucb_closed_form")
    for _ in range(DRAWS):
        level = float(r.uniform(1e-4, 8.0))
        assert klucb_index(0.0, level) == pytest.approx(
            1.0 - math.exp(-level), abs=1e-6)


def test_klucb_index_is_monotone_in_both_arguments():
    """More exploration budget never lowers the index, and a higher empirical
    mean never lowers it either. Non-monotonicity here would let an arm look
    worse after a success."""
    r = rng("klucb_monotone")
    for _ in range(DRAWS):
        p = float(r.uniform(0.0, 0.9))
        lo, hi = sorted(r.uniform(1e-3, 1.0, size=2))
        assert klucb_index(p, float(lo)) <= klucb_index(p, float(hi)) + 1e-9
        bump = float(min(p + r.uniform(0.0, 0.09), 0.99))
        level = float(r.uniform(1e-3, 1.0))
        assert klucb_index(p, level) <= klucb_index(bump, level) + 1e-9


def test_klucb_index_never_exceeds_pinsker():
    """Because d >= 2(p-u)^2, the KL index is at most p + sqrt(level/2) — the
    Hoeffding bonus UCB1 uses. KL-UCB is optimistic, but never more optimistic
    than the policy it is meant to improve on."""
    r = rng("klucb_pinsker")
    for _ in range(DRAWS):
        p = float(r.uniform(0.0, 1.0))
        level = float(r.uniform(1e-4, 3.0))
        assert klucb_index(p, level) <= min(1.0, p + math.sqrt(level / 2)) + 1e-9


# --------------------------------------------------------------------------
# EXP3's sampling distribution
# --------------------------------------------------------------------------

def test_exp3_probabilities_are_a_valid_distribution_with_a_floor():
    """After any history, p sums to 1 and every p_i >= gamma/K.

    The exploration floor is what makes the importance-weighted estimator's
    variance finite; the original overflow bug showed up first as a p that
    summed to NaN, so both halves are worth asserting together.
    """
    r = rng("exp3_simplex")
    for _ in range(60):
        n_arms = int(r.integers(2, 8))
        gamma = float(r.uniform(0.01, 1.0))
        algo = EXP3(n_arms, gamma=gamma, seed=int(r.integers(1 << 30)))
        for t in range(1, int(r.integers(50, 400))):
            arm = algo.select(t)
            algo.update(arm, float(r.random()))
        probs = algo._probs(gamma)
        assert sum(probs) == pytest.approx(1.0)
        assert min(probs) >= gamma / n_arms - 1e-12
        assert all(math.isfinite(w) for w in algo.weights)


def test_exp3_is_invariant_to_a_shift_of_all_log_weights():
    """Adding the same constant to every log-weight leaves the sampling
    distribution unchanged — the gauge freedom the log-space rewrite exploits
    to renormalize without changing the algorithm."""
    r = rng("exp3_gauge")
    for _ in range(60):
        n_arms = int(r.integers(2, 8))
        gamma = float(r.uniform(0.05, 0.9))
        algo = EXP3(n_arms, gamma=gamma, seed=7)
        for t in range(1, 120):
            arm = algo.select(t)
            algo.update(arm, float(r.random()))
        before = algo._probs(gamma)
        shift = float(r.normal(0.0, 50.0))
        algo._log_w = [lw + shift for lw in algo._log_w]
        assert algo._probs(gamma) == pytest.approx(before, abs=1e-12)


def test_anytime_gamma_decays_and_stays_a_probability():
    """gamma_t is in (0, 1] and non-increasing in t: exploration may slow
    down but must never turn negative or restart."""
    r = rng("anytime_gamma")
    for _ in range(DRAWS):
        n_arms = int(r.integers(2, 40))
        t = int(r.integers(1, 10 ** 6))
        g = anytime_gamma(n_arms, t)
        assert 0.0 < g <= 1.0
        assert anytime_gamma(n_arms, t + int(r.integers(1, 10_000))) <= g + 1e-15


# --------------------------------------------------------------------------
# Regret accounting
# --------------------------------------------------------------------------

def test_pseudo_regret_telescopes_into_the_gaps():
    """R(t) - R(t-1) is exactly the gap of the arm pulled at t, R is
    non-decreasing, and R(T) <= T * max gap.

    Pseudo-regret is a bookkeeping identity, not an estimate: if this ever
    drifts, every comparison against every bound in the repository is
    measuring something other than what it claims.
    """
    r = rng("regret_telescope")
    for _ in range(60):
        n_arms = int(r.integers(2, 9))
        probs = list(r.uniform(0.0, 1.0, size=n_arms))
        env = BernoulliBandit([float(p) for p in probs])
        arms = [int(a) for a in r.integers(0, n_arms, size=int(r.integers(5, 200)))]
        curve = cumulative_pseudo_regret(env, arms)
        gaps = [env.optimal_mean - env.mean(a) for a in arms]
        assert curve[0] == pytest.approx(gaps[0])
        for i in range(1, len(arms)):
            assert curve[i] - curve[i - 1] == pytest.approx(gaps[i])
        assert all(x >= -1e-12 for x in curve)
        assert curve[-1] <= len(arms) * max(gaps) + 1e-9


def test_lai_robbins_constant_is_relabelling_invariant():
    """Permuting the arms cannot change C(nu): the bound is a property of the
    multiset of reward distributions, not of the order they were listed in."""
    r = rng("lr_permutation")
    for _ in range(DRAWS):
        probs = [float(p) for p in r.uniform(0.02, 0.95, size=int(r.integers(2, 8)))]
        base = lai_robbins_lower_bound(BernoulliBandit(probs))
        shuffled = list(probs)
        r.shuffle(shuffled)
        assert lai_robbins_lower_bound(BernoulliBandit(shuffled)) == pytest.approx(base)
        assert base >= 0.0


def test_lai_robbins_gaussian_constant_scales_with_variance():
    """Scaling every sigma by c multiplies C(nu) by c^2, and shifting every
    mean by a constant leaves it alone. Both follow from the Gaussian KL and
    neither survives the value-range dispatch this bound used to do."""
    r = rng("lr_gaussian_scaling")
    for _ in range(120):
        k = int(r.integers(2, 6))
        mus = [float(m) for m in r.normal(size=k)]
        sigmas = [float(s) for s in r.uniform(0.2, 3.0, size=k)]
        if len(set(mus)) < k:
            continue
        base = lai_robbins_lower_bound(GaussianBandit(mus, sigmas=sigmas))
        c = float(r.uniform(0.5, 3.0))
        scaled = lai_robbins_lower_bound(
            GaussianBandit(mus, sigmas=[s * c for s in sigmas]))
        assert scaled == pytest.approx(base * c ** 2)
        shift = float(r.normal())
        shifted = lai_robbins_lower_bound(
            GaussianBandit([m + shift for m in mus], sigmas=sigmas))
        assert shifted == pytest.approx(base)


# --------------------------------------------------------------------------
# Policy bookkeeping
# --------------------------------------------------------------------------

@pytest.mark.parametrize("build", [
    lambda k, s: UCB1(k),
    lambda k, s: KLUCB(k),
    lambda k, s: EpsilonGreedy(k, eps=0.2, seed=s),
    lambda k, s: ThompsonBernoulli(k, seed=s),
], ids=["ucb1", "klucb", "eps-greedy", "thompson"])
def test_counts_conserve_pulls_and_means_stay_in_range(build):
    """Every policy's pull counts sum to the number of updates it received,
    and every running mean stays inside the reward range it was fed.

    Both are conservation laws: an off-by-one in the incremental mean or a
    dropped count would corrupt the index without raising anything.
    """
    r = rng("counts_conserved")
    for _ in range(30):
        n_arms = int(r.integers(2, 7))
        horizon = int(r.integers(20, 400))
        algo = build(n_arms, int(r.integers(1 << 30)))
        for t in range(1, horizon + 1):
            arm = algo.select(t)
            assert 0 <= arm < n_arms
            algo.update(arm, float(r.random()))
        counts = getattr(algo, "counts", None)
        if counts is None:  # Thompson keeps Beta parameters, not counts
            pulls = sum(a + b for a, b in zip(algo.alphas, algo.betas, strict=True))
            assert pulls == pytest.approx(2 * n_arms + horizon)
            assert all(0.0 <= a / (a + b) <= 1.0
                       for a, b in zip(algo.alphas, algo.betas, strict=True))
        else:
            assert sum(counts) == horizon
            assert all(0.0 <= v <= 1.0 for v in algo.values)


def test_reset_returns_every_policy_to_its_initial_behaviour():
    """reset() must be a true rewind: a policy replayed on the same rewards
    after a reset produces the same arms it did the first time. Anything left
    over from the previous run would make replications silently dependent."""
    r = rng("reset_rewind")
    rewards = [float(x) for x in r.random(200)]
    for build in (lambda k: UCB1(k), lambda k: KLUCB(k)):
        algo = build(4)
        first = []
        for t, reward in enumerate(rewards, start=1):
            arm = algo.select(t)
            algo.update(arm, reward)
            first.append(arm)
        algo.reset()
        second = []
        for t, reward in enumerate(rewards, start=1):
            arm = algo.select(t)
            algo.update(arm, reward)
            second.append(arm)
        assert first == second


def test_experiment_seeding_is_reproducible_and_seed_sensitive():
    """Same base_seed -> identical trajectories; different base_seed ->
    different ones. The second half is the regression: the runner used to
    overwrite the algorithm's RNG, so every seed produced the same run."""
    r = rng("runner_seeding")
    probs = [float(p) for p in r.uniform(0.1, 0.9, size=4)]

    def go(base_seed: int):
        return run_experiment(
            env_factory=lambda seed: BernoulliBandit(probs, seed=seed),
            algo_factory=lambda n_arms, seed: ThompsonBernoulli(n_arms, seed=seed),
            horizon=200, n_runs=3, base_seed=base_seed)

    assert go(11).runs[0].arms_pulled == go(11).runs[0].arms_pulled
    assert go(11).runs[0].arms_pulled != go(12).runs[0].arms_pulled


# --------------------------------------------------------------------------
# LinUCB's linear algebra
# --------------------------------------------------------------------------

def test_linucb_inverse_stays_symmetric_positive_definite():
    """A_inv is symmetric and positive definite after any sequence of rank-1
    updates, and A @ A_inv is the identity.

    Sherman-Morrison is an algebraic identity, so the only way this breaks is
    numerically — and a design matrix that drifts out of the PSD cone makes
    the confidence width sqrt(x^T A^-1 x) imaginary.
    """
    r = rng("linucb_spd")
    for _ in range(25):
        dim = int(r.integers(2, 9))
        algo = LinUCB(2, dim, alpha=1.0)
        for _ in range(int(r.integers(20, 300))):
            x = r.normal(size=dim) * r.uniform(0.1, 4.0)
            algo.update(int(r.integers(2)), float(r.normal()), x)
        for a in range(2):
            inv = algo.A_inv[a]
            assert np.allclose(inv, inv.T, atol=1e-10)
            assert np.min(np.linalg.eigvalsh(inv)) > 0.0
            assert np.allclose(algo.A[a] @ inv, np.eye(dim), atol=1e-8)
            assert np.allclose(inv, np.linalg.inv(algo.A[a]), atol=1e-8)


def test_linucb_confidence_width_shrinks_along_directions_it_has_seen():
    """Updating arm a with x can only shrink that arm's width in every
    direction: x^T A_new^-1 x <= x^T A^-1 x for all probes.

    A_new^-1 = A^-1 - u u^T / (1 + x^T A^-1 x) subtracts a PSD rank-1 term, so
    optimism is monotonically non-increasing in evidence — the property that
    makes an optimistic index policy converge at all.
    """
    r = rng("linucb_width_shrinks")
    for _ in range(40):
        dim = int(r.integers(2, 8))
        algo = LinUCB(1, dim)
        for _ in range(int(r.integers(5, 60))):
            x = r.normal(size=dim)
            before = algo.A_inv[0].copy()
            algo.update(0, float(r.normal()), x)
            for _ in range(5):
                probe = r.normal(size=dim)
                assert (probe @ algo.A_inv[0] @ probe
                        <= probe @ before @ probe + 1e-10)


def test_linucb_is_equivariant_under_rotation():
    """Rotate the contexts and every theta_a by the same orthogonal Q and
    LinUCB makes exactly the same choices.

    A_a starts at the identity, which commutes with every rotation, so the
    policy has no preferred basis. A regularizer or an inverse that quietly
    picked one — say by dropping the symmetry of A_inv — would break this.
    """
    r = rng("linucb_rotation")
    dim, n_arms, horizon = 5, 4, 300
    theta = r.normal(size=(n_arms, dim))
    q, _ = np.linalg.qr(r.normal(size=(dim, dim)))
    contexts = r.normal(size=(horizon, dim))
    noise = r.normal(size=horizon) * 0.05

    def play(th, xs):
        algo = LinUCB(n_arms, dim, alpha=1.0)
        picks = []
        for t in range(horizon):
            x = xs[t]
            a = algo.select(x)
            algo.update(a, float(x @ th[a] + noise[t]), x)
            picks.append(a)
        return picks

    assert play(theta, contexts) == play(theta @ q.T, contexts @ q.T)


def test_contextual_regret_is_non_negative_and_bounded_by_the_span():
    """Per-round contextual regret is optimal_mean(x) - mean(a, x), which is
    non-negative by construction and at most the spread of theta_a^T x across
    arms. The cumulative curve therefore never decreases."""
    r = rng("contextual_regret_range")
    for _ in range(20):
        dim, n_arms = int(r.integers(2, 7)), int(r.integers(2, 6))
        theta = r.normal(size=(n_arms, dim))
        env = LinearContextualBandit(theta, sigma=0.2, seed=int(r.integers(1 << 30)))
        algo = LinUCB(n_arms, dim)
        prev, span = 0.0, 0.0
        for _ in range(80):
            x = env.context()
            a = algo.select(x)
            algo.update(a, env.pull(a, x), x)
            step = env.optimal_mean(x) - env.mean(a, x)
            values = theta @ x
            assert -1e-12 <= step <= float(values.max() - values.min()) + 1e-12
            prev += step
            span += float(values.max() - values.min())
        assert 0.0 <= prev <= span + 1e-9

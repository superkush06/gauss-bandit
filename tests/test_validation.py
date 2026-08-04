"""The claims in docs/validation.md, at a horizon CI can afford.

`examples/validate.py` is the long form: 200,000-round sweeps, twenty reward
tables, a quarter of a million Monte-Carlo replications. These are the same
comparisons shrunk until the whole file runs in a few seconds, so a change
that quietly breaks agreement with a published bound fails the build instead
of waiting for someone to rerun the example.

Every bound here is either printed in a paper or derivable in two lines; none
of them is a threshold picked to make the current numbers pass. Where a test
does carry a tolerance, the docstring says which direction the slack points.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bandit import (
    KLUCB,
    UCB1,
    BernoulliBandit,
    ThompsonBernoulli,
    lai_robbins_lower_bound,
)
from examples.validate import (
    KLUCB_TOL,
    MODERATE,
    exact_expected_regret,
    exp3_weak_regret,
    klucb_closed_form_residual,
    linucb_potential_run,
    mean_regret,
    monte_carlo_regret,
    ucb1_upper_bound,
)

E_MINUS_1 = math.e - 1.0


# --------------------------------------------------------------------------
# UCB1 between two published constants
# --------------------------------------------------------------------------

def test_ucb1_upper_bound_matches_the_paper_formula():
    """Auer, Cesa-Bianchi & Fischer (2002), Theorem 1, evaluated by hand on a
    two-armed instance: 8 ln n / Delta + (1 + pi^2/3) Delta."""
    n, gap = 1_000, 0.25
    expected = 8 * math.log(n) / gap + (1 + math.pi ** 2 / 3) * gap
    assert ucb1_upper_bound([0.5, 0.25], n) == pytest.approx(expected)


def test_ucb1_regret_lies_between_lai_robbins_and_theorem_1():
    """Measured UCB1 regret must clear the Lai-Robbins floor and stay under
    the Auer et al. guarantee.

    The upper side is the real test: exceeding a proved worst-case bound would
    mean the index is wrong. The lower side has the weaker status of the two —
    the floor is asymptotic, so a policy can dip below it at finite T — but
    UCB1's constant is large enough that it clears the floor by 2x well before
    this horizon.
    """
    horizon = 20_000
    floor = lai_robbins_lower_bound(BernoulliBandit(MODERATE)) * math.log(horizon)
    upper = ucb1_upper_bound(MODERATE, horizon)
    measured = mean_regret(lambda k, s: UCB1(k), MODERATE, horizon, range(8))
    assert floor < measured < upper


# --------------------------------------------------------------------------
# The two asymptotically optimal policies
# --------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [
    lambda k, s: KLUCB(k),
    lambda k, s: ThompsonBernoulli(k, seed=s),
], ids=["klucb", "thompson"])
def test_optimal_policies_climb_toward_the_floor_from_below(factory):
    """R(T) / (C ln T) is increasing in T and does not overshoot.

    Garivier & Cappe (2011) and Kaufmann, Korda & Munos (2012) both prove the
    ratio tends to 1. Neither says anything about finite T, so the assertion
    is the direction of travel plus a ceiling loose enough that a genuinely
    optimal policy cannot trip it.
    """
    floor_c = lai_robbins_lower_bound(BernoulliBandit(MODERATE))
    ratios = []
    for horizon in (2_000, 20_000):
        r = mean_regret(factory, MODERATE, horizon, range(8))
        ratios.append(r / (floor_c * math.log(horizon)))
    assert ratios[0] < ratios[1]
    assert ratios[-1] < 1.2


def test_klucb_is_closer_to_the_floor_than_thompson_at_this_horizon():
    """A finding, not a theorem: both policies are asymptotically optimal, but
    KL-UCB's ratio is roughly 0.7 where Thompson's is roughly 0.46 at T = 20k.
    Thompson wins on raw regret and loses on how much of the ln T bill it has
    paid — which is why docs/validation.md reports both columns."""
    horizon = 20_000
    klucb = mean_regret(lambda k, s: KLUCB(k), MODERATE, horizon, range(8))
    thompson = mean_regret(lambda k, s: ThompsonBernoulli(k, seed=s),
                           MODERATE, horizon, range(8))
    assert thompson < klucb


# --------------------------------------------------------------------------
# EXP3 in the setting its theorem is stated for
# --------------------------------------------------------------------------

def _weak_regret(horizon: int, n_arms: int, gamma: float, runs: int):
    probs = np.array([0.5 + 0.03 * i for i in range(n_arms)])
    total, g_total = 0.0, 0.0
    for seed in range(runs):
        rng = np.random.default_rng(20_000 + seed)
        table = (rng.random((horizon, n_arms)) < probs).astype(float)
        wr, g_max = exp3_weak_regret(table, gamma, seed + 7919)
        total += wr
        g_total += g_max
    return total / runs, g_total / runs


def test_exp3_weak_regret_respects_theorem_3_1():
    """G_max - E[G_EXP3] <= (e-1) gamma G_max + K ln K / gamma for any fixed
    reward assignment (Auer, Cesa-Bianchi, Freund & Schapire 2002, Thm 3.1).

    The reward table is drawn once and held, which is what makes G_max the
    quantity the theorem bounds. Resampling rewards each round would measure
    something else and the comparison would be meaningless.
    """
    horizon, n_arms, gamma = 10_000, 10, 0.05
    wr, g_max = _weak_regret(horizon, n_arms, gamma, runs=5)
    bound = E_MINUS_1 * gamma * g_max + n_arms * math.log(n_arms) / gamma
    assert 0.0 < wr <= bound


def test_exp3_tuned_gamma_respects_corollary_3_2():
    """With gamma = min(1, sqrt(K ln K / ((e-1) g))) and g >= G_max, the same
    paper's Corollary 3.2 gives 2 sqrt(e-1) sqrt(g K ln K). Rewards are in
    [0, 1] so g = T is admissible."""
    horizon, n_arms = 10_000, 10
    k_ln_k = n_arms * math.log(n_arms)
    gamma = min(1.0, math.sqrt(k_ln_k / (E_MINUS_1 * horizon)))
    wr, _ = _weak_regret(horizon, n_arms, gamma, runs=5)
    assert wr <= 2.0 * math.sqrt(E_MINUS_1) * math.sqrt(horizon * k_ln_k)


# --------------------------------------------------------------------------
# LinUCB
# --------------------------------------------------------------------------

def test_linucb_satisfies_the_elliptical_potential_inequality():
    """sum_t min(1, ||x_t||^2_{A^-1}) <= 2 ln det A(T).

    This is the determinant lemma, not a quoted constant: det(A + xx^T) =
    det(A)(1 + ||x||^2_{A^-1}) and u <= 2 ln(1+u) on [0, 1]. It is the step
    the O(d sqrt(T)) analysis of Abbasi-Yadkori, Pal & Szepesvari (2011)
    turns on, and it holds for any sequence of contexts whatsoever.
    """
    for seed in (0, 1):
        ratio, det_ok, _ = linucb_potential_run(seed, horizon=3_000, dim=8,
                                                n_arms=5)
        assert ratio <= 1.0
        assert det_ok


def test_linucb_regret_exponent_is_well_under_one_half():
    """R(T) ~ T^p with p far below the 0.5 the sqrt(T) analysis allows.

    Recorded as a fact about this environment rather than as a certificate:
    isotropic Gaussian contexts fill every direction of the design matrix, so
    LinUCB does much better here than the worst case. A p above 0.5 would be
    a genuine problem; a p near 0 only means the bound is not tight.
    """
    _, _, exponent = linucb_potential_run(0, horizon=3_000, dim=8, n_arms=5)
    assert exponent < 0.5


# --------------------------------------------------------------------------
# Exact ground truth, no citation involved
# --------------------------------------------------------------------------

def test_path_enumeration_agrees_with_a_hand_computation():
    """UCB1 pulls each arm once before indexing, so over exactly two rounds
    its expected regret is the sum of the gaps: 0.2 on [0.6, 0.4], with no
    randomness left in it. Anchors the enumerator before it is trusted."""
    assert exact_expected_regret(lambda: UCB1(2), [0.6, 0.4], 2) == pytest.approx(0.2)


def test_exact_expected_regret_matches_monte_carlo():
    """Enumerating all 2^T reward paths and sampling them must agree.

    The two calculations share no code beyond the policy itself: one is a
    weighted sum over a binary tree, the other a seeded simulation. Tolerance
    is four standard errors of the Monte-Carlo mean, so this is a genuine
    two-sided check rather than a loose sanity bound.
    """
    probs, horizon = [0.6, 0.4], 10
    exact = exact_expected_regret(lambda: UCB1(2), probs, horizon)
    mc, se = monte_carlo_regret(lambda: UCB1(2), probs, horizon, runs=40_000)
    assert abs(mc - exact) < 4 * se


def test_exact_expected_regret_weights_paths_by_their_probability():
    """Third round on [0.6, 0.4], worked out by hand.

    Rounds 1 and 2 are the round-robin, costing 0 + 0.2. At t = 3 both arms
    have one pull, so the confidence bonuses are equal and UCB1 picks by
    empirical mean, breaking the tie towards arm 0. Only the path (miss,
    hit) — probability 0.4 x 0.4 — sends it to the wrong arm, so

        E[R(3)] = 0.2 + 0.16 x 0.2 = 0.232.

    A tree that enumerated the right paths with the wrong weights would land
    somewhere else; Monte Carlo would call the difference noise.
    """
    assert exact_expected_regret(lambda: UCB1(2), [0.6, 0.4], 3) == pytest.approx(0.232)
    assert exact_expected_regret(lambda: UCB1(3), [0.5, 0.5, 0.5], 6) == 0.0


def test_klucb_index_at_zero_stays_inside_its_advertised_tolerance():
    """Row 12 of docs/validation.md, recomputed rather than quoted.

    d(0, u) = -ln(1 - u) inverts exactly, so `klucb_index(0, L)` has a closed
    form to be held against: 1 - exp(-L). The published number is the worst
    residual over 300 random levels, 4.74e-07 against the 1e-6 the solver
    advertises. The two asserts below catch different mutation sizes, which is
    why both are here: a bisection stopping one halving early lands at
    9.51e-07, still inside KLUCB_TOL and still printing "holds", so only the
    pinned value notices; two halvings early lands at 1.90e-06 and trips both.
    """
    worst, level = klucb_closed_form_residual()
    assert worst <= KLUCB_TOL
    assert worst == pytest.approx(4.74e-07, rel=0.02)
    assert 0.0 < level <= 8.0

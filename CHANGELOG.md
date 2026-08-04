# Changelog

## [0.5.3] - 2026-08-03

### Fixed
- **The section 4 block in `docs/validation.md` was an abridged paste of
  `examples/validate.py`, not the paste it presented itself as.** The numbers
  in it were all correct — a rerun reproduces 932.7 / 3767.7 / 0.25 / 1537.5 /
  2489.3 / 0.62 / 2813.0 exactly, since the reward tables are drawn from seeded
  generators — but the block dropped the rule line separating the gamma rows
  from the Corollary 3.2 line, and with it the `measured at tuned gamma 1537.5
  (holds)` line, which is where section 4's verdict is actually printed. A
  reader diffing the block against `python3 examples/validate.py exp3` got a
  mismatch. Both lines are back.
- **The section 2 and section 3 block dropped the rules under its two title
  lines**, the same way and for no better reason. Restored.

### Changed
- `docs/validation.md` now states what its fenced blocks are: each is that
  section's output from the top, cut where the script stops printing numbers
  and starts explaining them, with a section title and the rule beneath it
  dropped where the Markdown heading already carries them. Every output block
  on the page satisfies that rule as of this release — sections 1, 2, 3, 4, 5
  and 6 each appear as a contiguous run of lines in a `--long` run.

## [0.5.2] - 2026-07-29

### Fixed
- **The mutation-sensitivity claim on row 12 of `docs/validation.md` was off by
  one halving.** Measured on the shipped tree and on copies with the bisection's
  stopping rule widened: `hi - lo < tol` gives a worst residual of 4.74e-07;
  `tol * 2` gives 9.51e-07, which is still under `KLUCB_TOL = 1e-6`, so
  `examples/validate.py` still prints "holds" and the row stays green; only
  `tol * 4` reaches 1.90e-06 and flips the verdict to VIOLATED. The docs, the
  test docstring and the 0.5.1 entry below all said one halving turned the row
  red. What one halving actually breaks is the pinned value
  `pytest.approx(4.74e-07, rel=0.02)`; two halvings break the printed verdict
  as well.
- **Section 4 of `examples/validate.py` claimed "Both bounds hold with room to
  spare" unconditionally**, three lines under a verdict derived from
  `tuned_wr <= cor32`. Any run that violated the bound printed VIOLATED and
  then contradicted itself. The narration is now taken from the same
  comparisons as the verdict — the worst Thm 3.1 ratio and the tuned run's
  share of Cor 3.2 (0.62 and 0.55 on the default table) — with a different
  paragraph when either bound fails.

## [0.5.1] - 2026-07-29

### Fixed
- **EXP3's default gamma schedule is no longer attributed to Corollary 3.2.**
  `anytime_gamma` is that corollary's tuned gamma with the horizon `g` replaced
  by the current round `t` — a standard adaptation that inherits none of the
  corollary's guarantee, since Cor 3.2 fixes a *single* gamma from a known
  bound on `G_max` and is neither time-varying nor horizon-free. The paper's
  own unknown-horizon algorithm is Exp3.1 (its Section 4), which doubles a
  guess at `G_max` instead. `docs/theory.md`, the `anytime_gamma` docstring and
  the README now all say this; previously only section 7 of
  `docs/validation.md` did, and the other three claimed the opposite.
- Row 12 of `docs/validation.md` carried a formula where every other row
  carries a number, so it could not be checked the way the rest of the table
  can. `examples/validate.py` now measures it — the worst error of
  `klucb_index(0, L)` against its closed form `1 - exp(-L)` over 300 random
  levels, 4.74e-07 against the solver's advertised 1e-6 — and
  `tests/test_validation.py` recomputes it. Widening the bisection's stopping
  rule by one halving pushes the residual to 1.9e-06, flips the printed
  verdict to VIOLATED and fails the test. [Corrected in 0.5.2: 1.9e-06 is the
  *two*-halving number. One halving gives 9.51e-07, which is still under the
  1e-6 tolerance, so the printed verdict stays "holds"; it fails the test only
  because the test pins the value.]
- `examples/execution_router.py` said the equal-split router wastes 0.168 bp
  in its docstring; the script and the README both say 0.169. 0.168 was the two
  displayed roundings subtracted by hand.

## [0.5.0] - 2026-07-27

### Added
- **`docs/validation.md`** — every regret claim in the library checked against
  something outside it: UCB1 against the finite-time bound of Auer,
  Cesa-Bianchi & Fischer (2002, Thm 1) above and Lai & Robbins (1985) below;
  KL-UCB and Thompson against the asymptotic optimality proved by Garivier &
  Cappe (2011) and Kaufmann, Korda & Munos (2012); EXP3's weak regret against
  Auer, Cesa-Bianchi, Freund & Schapire (2002, Thm 3.1 / Cor 3.2) on a fixed
  reward table, which is the setting those results are stated for; and
  LinUCB against the elliptical-potential inequality behind the linear-bandit
  sqrt(T) analysis. Includes a section on the three places the library does
  *not* match its references and why.
- **`examples/validate.py`** — the script that produces every number in that
  table, section by section (`--long` carries the asymptotic sections to
  T = 200,000). Its strongest check needs no citation at all: UCB1 and KL-UCB
  are deterministic given their rewards, so `exact_expected_regret` walks all
  2^T reward paths and returns E[R(T)] in closed form. It agrees with 200,000
  Monte-Carlo replications to about one standard error.
- **`tests/test_validation.py`** — the same comparisons at CI-sized horizons,
  so a regression against a published bound fails the build.
- **`tests/test_invariants.py`** — randomized property tests over hundreds of
  seeded draws: Pinsker's inequality, the KL's label symmetry, the KL-UCB
  index bracketing its root and matching the closed form `1 - exp(-L)` at
  p = 0, EXP3's simplex and gamma/K floor and its invariance to a shift of all
  log-weights, pseudo-regret telescoping into the gaps that produced it, the
  Lai-Robbins constant's permutation invariance and its sigma^2 scaling for
  Gaussian arms, and LinUCB's inverse staying symmetric positive definite,
  shrinking every confidence width it updates, and choosing identically when
  the whole problem is rotated.
- **`examples/execution_router.py`** — the library doing its job inside a
  pipeline: a 20,000-slice parent order routed across four venues with unknown
  passive-fill rates, scored in basis points, with the Lai-Robbins constant
  read back as the part of the shortfall that was never recoverable. Nothing
  imported; the upstream scorecard and downstream shortfall are inlined.
- README gains a "Does it meet the bounds it cites?" section and a "Where this
  sits" note placing the repository among its siblings.

### Fixed
- `klucb_index` dropped an unreachable branch: `d(p, 1)` is infinite for every
  `p < 1`, and `p >= 1` returns two lines earlier, so the ceiling shortcut
  could never fire. Its docstring now states the tolerance is on the *q* axis,
  which is where the bisection actually converges.
- `docs/exp3_longrun.png`: the overflow marker ran the full height of the axes
  and struck through all three legend labels. It now stops below them.
- CI no longer swallows pytest exit code 5. The bootstrap-phase escape hatch
  turned a collection error into a green build.
- Every example command in the README and in the example docstrings drops the
  `PYTHONPATH=.` prefix, which is unnecessary after `pip install -e`, and
  `python`/`python3` no longer alternate between adjacent blocks.

## [0.4.0] - 2026-07-27

### Added
- **KL-UCB** (`bandit.algos.KLUCB`, Garivier & Cappe 2011): the index
  `max{q : n_i d(mu_hat_i, q) <= ln t + c ln ln t}`, solved by bisection on
  the binary KL — the same divergence `lai_robbins_lower_bound` uses, which is
  why the two meet. `klucb_index` is exposed on its own for anyone who wants
  the solver without the policy. Default `c = 0`.
- `docs/optimality.png`, the new lead figure: mean regret divided by ln T for
  four policies against the Lai-Robbins constant, over 200,000 rounds in two
  reward regimes. At T = 200k, KL-UCB sits at 0.87 of the floor on the
  rare-reward instance where UCB1 is at 10.8x.
- `docs/contextual.png`: LinUCB against a context-blind UCB1 on the same
  linear instance — 49 regret against 11,091 over 4,000 rounds.
- `docs/figures.py`: one renderer for every figure in the README, printing the
  table behind each chart before it saves.
- `examples/optimality.py`: the same study at a horizon that finishes in
  seconds, reported as R(T) / (C ln T).
- `tests/test_optimality.py`: benchmarks that tie measured regret back to the
  bound — KL-UCB and Thompson under 1.2x the floor on rare rewards, UCB1 above
  4x, and the per-family dispatch pinned so the Bernoulli/Gaussian constants
  cannot converge again.
- `tests/test_klucb.py`: the index solves d(p, u) = level to 1e-4, is monotone
  in the budget, and never exceeds the Pinsker bound p + sqrt(level / 2).

### Changed
- **`LinUCB` maintains each arm's inverse design matrix by Sherman-Morrison
  rank-1 updates** instead of calling `np.linalg.inv` for every arm every
  round: O(K d^2) per round rather than O(K d^3). Tests pin `A_inv` against
  `np.linalg.inv(A)` to 1e-10 after 2,000 updates and assert identical arm
  choices against the direct-inverse implementation for 1,000 rounds.
- `examples/compare_algos.py` includes KL-UCB in the leaderboard.
- `examples/exp3_longrun.py` reports the table only; its chart now comes from
  `docs/figures.py exp3`.
- `docs/theory.md` rewritten around why the KL index attains the Lai-Robbins
  constant and the Hoeffding index cannot, with the Pinsker comparison
  tabulated, plus sections on EXP3's log-space weights and LinUCB's
  Sherman-Morrison update. Math now renders on GitHub.
- README leads with the optimality figure; every table in it is pasted from a
  real run of the command printed above it.

### Removed
- `examples/render_hero.py` and `examples/visualize_regret.py`, superseded by
  `docs/figures.py` (and `docs/demo.png` with them).

## [0.3.0] - 2026-07-XX

### Fixed
- **EXP3 no longer overflows on long horizons.** Weights are now stored in
  log-space and sampled through a max-shifted softmax; previously the
  linear-space weights hit `inf` around t~9,000 (gamma=0.2), probabilities
  went NaN, and `select()` silently returned the last arm forever.
- **EXP3 importance weights use the select-time distribution.** The
  probability used in the reward estimate is cached when the arm is drawn
  instead of being recomputed from possibly-changed weights at update time.
- **`lai_robbins_lower_bound` dispatches on the reward family.** Gaussian
  envs were being scored with Bernoulli KL whenever their means happened to
  lie in (0,1) — and arm sigmas were ignored entirely. Bernoulli edge cases
  (p* = 1, p_i = 0) fell into a quadratic fallback and returned nonzero
  constants where the exact answers are 0 and gap/KL(0||p*).
- **The runner honors algorithm seeds.** `run_one` used to overwrite
  `algo._rng` with its own `random.Random`, silently discarding any seed the
  factory set (seeds 123 and 999 produced identical trajectories) and
  crashing algorithms that hold a numpy `Generator`.
- **EpsilonGreedy explores before exploiting.** Every arm is pulled once
  before the eps/greedy rule applies, and value ties are broken uniformly at
  random; with eps=0 the old argmax locked onto arm 0 forever.

### Added
- `anytime_gamma(n_arms, t)` — the decaying EXP3 exploration schedule
  `gamma_t = sqrt(K ln K / ((e-1) t))`, now the default (`gamma=None`). At
  T=100k it beats fixed gamma=0.07 by ~8x (regret 234 vs 1798; see
  `examples/exp3_longrun.py` and README). [Attribution corrected in 0.5.1:
  this entry originally credited Auer et al. 2002, Cor. 3.2, which covers a
  single fixed gamma, not a schedule.]
- `bernoulli_kl` / `gaussian_kl` helpers, exported at the package root;
  `lai_robbins_lower_bound(env, kl=...)` accepts a custom divergence for
  user-defined env families.
- Long-horizon regression tests: 100k-step EXP3 (finite weights, valid
  distribution, decelerating regret) and 50k-step sublinearity checks for
  UCB1 and Thompson.
- `examples/exp3_longrun.py` + `docs/exp3_longrun.png`: fixed vs anytime
  gamma over 100,000 steps.

### Changed
- Runner contract: `algo_factory(n_arms, seed)`. Old single-argument
  factories still run (their own seeding now respected) but emit a
  `DeprecationWarning`.
- `EXP3` default exploration is now the anytime schedule instead of a fixed
  gamma=0.1.

## [0.2.0] - 2026-06-XX

### Added
- **Contextual bandits**: `LinUCB` (disjoint per-arm linear models with an
  optimism confidence bonus), `LinearContextualBandit` environment, and
  `run_contextual` for cumulative contextual regret.
- **Regret hero chart** (`examples/render_hero.py` → `docs/demo.png`):
  context-free algorithm comparison + LinUCB contextual regret.

## [0.1.0] - 2026-06-XX

### Added
- Environments: `BernoulliBandit`, `GaussianBandit`.
- Algorithms: `EpsilonGreedy` (constant + annealed schedule), `UCB1`,
  `ThompsonBernoulli`, `ThompsonGaussian`, `EXP3`.
- Metrics: `cumulative_regret`, `cumulative_pseudo_regret`,
  `lai_robbins_lower_bound`.
- Experiment runner: `run_one`, `run_experiment` with seeded replication.
- Two examples: `compare_algos.py`, `visualize_regret.py`.
- Theory primer in `docs/theory.md`.
- CI on Python 3.11 + 3.12.

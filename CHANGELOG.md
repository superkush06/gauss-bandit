# Changelog

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

# Validation

Every algorithm in this repository has a published guarantee attached to it.
This page checks whether the code actually obeys them.

The rule for what goes in the table: each row compares a number this library
produced against a number that came from somewhere else — a theorem, or a
calculation done a different way — and every value in the "ours" column is
pasted from a run of

```bash
python3 examples/validate.py --long        # ~3 min
python3 examples/validate.py exact         # one section, ~10 s
```

`tests/test_validation.py` asserts the same comparisons at a horizon CI can
afford, so a regression fails the build rather than waiting for someone to
rerun the script.

The fenced blocks are that script's output rather than a retyping of it. Each
one starts at the top of its section and is cut where the script stops printing
numbers and starts explaining them, because the explaining is what this page
does — with the sources attached. The only other edit is dropping a section's
title line and the rule under it where the Markdown heading above already says
the same thing; §2 and §3 share a heading here, so they keep theirs. Diffing a
block against a rerun should therefore show changed numbers or nothing at all.

Two things this page deliberately does *not* do. It does not tune an instance
until a bound binds — the stochastic sections all use the same fixed instance
`[0.50, 0.45, 0.40, 0.35, 0.30]`, chosen because it is where UCB1's Hoeffding
bonus is at its **tightest** and so flatters it, not the other way round. And
it does not hide the rows where our number and the reference disagree; those
are in [§7](#7-where-we-do-not-match).

---

## The table

| # | claim | ours | reference | source | agrees |
|---|---|---:|---:|---|:--:|
| 1 | UCB1 regret at T = 50,000 is below the finite-time guarantee | **552.6** | ≤ 3608.7 | Auer, Cesa-Bianchi & Fischer (2002), Thm 1 | yes |
| 2 | UCB1 regret at T = 50,000 is above the Lai-Robbins floor | **552.6** | ≥ 223.6 | Lai & Robbins (1985) | yes |
| 3 | Lai-Robbins constant for that instance | **20.66** | 20.66 | closed form, hand-computed below | yes |
| 4 | KL-UCB's R(T)/(C ln T) rises toward 1 | **0.53 → 0.88** | → 1 | Garivier & Cappé (2011) | direction only |
| 5 | Thompson's R(T)/(C ln T) rises toward 1 | **0.41 → 0.53** | → 1 | Kaufmann, Korda & Munos (2012) | direction only |
| 6 | EXP3 weak regret at γ = 0.05, T = 50,000 | **932.7** | ≤ 3767.7 | Auer, Cesa-Bianchi, Freund & Schapire (2002), Thm 3.1 | yes |
| 7 | EXP3 weak regret at the tuned γ = 0.01637 | **1537.5** | ≤ 2813.0 | same paper, Cor 3.2 | yes |
| 8 | LinUCB's elliptical potential ÷ 2 ln det A(T) | **0.457** | ≤ 1 | determinant lemma; the step behind Abbasi-Yadkori, Pál & Szepesvári (2011) | yes |
| 9 | LinUCB's measured regret exponent p | **0.006** | ≤ 0.5 | same | yes, vacuously |
| 10 | UCB1's exact E[R] at T = 14, by path enumeration | **1.118369** | 1.117497 ± 0.000850 | Monte Carlo, 200,000 replications | yes (z = 1.03) |
| 11 | KL-UCB's exact E[R] at T = 12, by path enumeration | **0.829564** | 0.834128 ± 0.003617 | Monte Carlo, 25,000 replications | yes (z = 1.26) |
| 12 | worst `klucb_index(0, L)` error against its analytic root, 300 random L | **4.74e−07** | ≤ 1e−6 | closed form 1 − e^−L, the solver's stated tolerance | yes |

---

## 1. UCB1, between two constants

Theorem 1 of Auer, Cesa-Bianchi & Fischer (2002) bounds UCB1's expected
regret after `n` plays on rewards in [0, 1] by

```
8 * sum_{i: mu_i < mu*} (ln n) / Delta_i  +  (1 + pi^2 / 3) * sum_j Delta_j
```

Lai & Robbins (1985) bound it from below by `C(nu) ln T + o(ln T)`. Both are
computable from the instance, so UCB1's measured regret has a two-sided
target and neither side involves a fitted constant.

```
instance [0.5, 0.45, 0.4, 0.35, 0.3], T = 50,000, 30 runs
  Lai-Robbins floor    C ln T                223.6   (C = 20.66)
  measured regret      R(T)                  552.6
  Auer et al. Thm 1    upper bound          3608.7
--------------------------------------------------------------------------
  floor <= measured                     holds   (measured / floor = 2.47)
  measured <= Thm 1                     holds   (measured / bound = 0.15)
```

Sitting at 0.15 of the upper bound is not a sign that anything is loose in
the code. Theorem 1 charges `8/Δ²` pulls of every suboptimal arm; a run that
gets unlucky in the first few hundred rounds needs them, and a typical run
does not. The falsifying observation would be `measured > bound`.

The `C = 20.66` is worth checking by hand, since every other row leans on it:

| arm | μ | Δ | d(μ, 0.5) | Δ / d |
|---|---:|---:|---:|---:|
| 1 | 0.45 | 0.05 | 0.005008 | 9.98 |
| 2 | 0.40 | 0.10 | 0.020136 | 4.97 |
| 3 | 0.35 | 0.15 | 0.045701 | 3.28 |
| 4 | 0.30 | 0.20 | 0.082283 | 2.43 |
| | | | **C** | **20.66** |

`lai_robbins_lower_bound(BernoulliBandit([0.50, 0.45, 0.40, 0.35, 0.30]))`
returns `20.662520018433387`, which is the same sum without the rounding.

## 2 & 3. The asymptotically optimal pair

Garivier & Cappé (2011) prove KL-UCB attains the Lai-Robbins constant itself
rather than a multiple of it, and Kaufmann, Korda & Munos (2012) prove the
same for Thompson sampling with a uniform prior on Bernoulli arms. Both are
statements about the limit, so what is measurable at finite T is the ratio
`R(T) / (C ln T)` and whether it is climbing.

```
2. KL-UCB against asymptotic optimality
--------------------------------------------------------------------------
instance [0.5, 0.45, 0.4, 0.35, 0.3], C = 20.66, 20 runs
         T        R(T)      C ln T   R(T)/(C ln T)
     5,000        93.9       176.0            0.53
    20,000       140.9       204.6            0.69
    50,000       178.0       223.6            0.80
   200,000       222.5       252.2            0.88

3. Thompson sampling against asymptotic optimality
--------------------------------------------------------------------------
instance [0.5, 0.45, 0.4, 0.35, 0.3], C = 20.66, 30 runs
         T        R(T)      C ln T   R(T)/(C ln T)
     5,000        72.4       176.0            0.41
    20,000        93.7       204.6            0.46
    50,000       109.3       223.6            0.49
   200,000       134.4       252.2            0.53
```

Both climb, neither arrives. See [§7](#7-where-we-do-not-match).

## 4. EXP3, in the setting the theorem is stated for

Theorem 3.1 of Auer, Cesa-Bianchi, Freund & Schapire (2002) bounds the *weak
regret* — the gap to the best single arm in hindsight — for an arbitrary
**fixed** assignment of rewards:

```
G_max - E[G_EXP3]  <=  (e - 1) gamma G_max + (K ln K) / gamma
```

and Corollary 3.2 tunes `gamma = min(1, sqrt(K ln K / ((e-1) g)))` for any
`g >= G_max`, giving `2 sqrt(e-1) sqrt(g K ln K)`. Rewards live in [0, 1], so
`g = T` is admissible.

Matching the setting matters here. The comparison is only faithful if the
reward table is drawn **once** and then held, so `G_max` is a well-defined
number the policy could in principle have earned. Resampling rewards every
round — the obvious thing to do — measures something the theorem says nothing
about.

```
K = 10, T = 50,000, 20 reward tables, tuned gamma = 0.01637
     gamma     weak regret   Thm 3.1 bound     ratio
   0.05000           932.7          3767.7      0.25
   0.01637          1537.5          2489.3      0.62
--------------------------------------------------------------------------
  Cor 3.2, g = T:      2 sqrt(e-1) sqrt(T K ln K) = 2813.0
  measured at tuned gamma                          1537.5   (holds)
```

Note that the *tuned* γ does worse than γ = 0.05 on this table. That is not a
contradiction: Corollary 3.2 minimises a worst-case bound, and these rewards
are i.i.d. rather than adversarial. EXP3 is paying for an adversary it never
meets, which is precisely the trade it exists to make.

## 5. LinUCB and the inequality behind √T

Our `LinUCB` is the disjoint model of Li, Chu, Langford & Schapire (2010)
with a fixed α, not the OFUL algorithm of Abbasi-Yadkori, Pál & Szepesvári
(2011) with its time-varying confidence radius, so it does not inherit their
high-probability regret constant. What it does inherit is the step that
constant is built on — the elliptical potential:

```
det(A + x x^T) = det(A) (1 + ||x||^2_{A^-1})          [determinant lemma]
u <= 2 ln(1 + u)  for u in [0, 1]
=>  sum_t min(1, ||x_t||^2_{A^-1})  <=  2 ln det A(T)
```

with `A(0) = I`, whose log-determinant is zero. This is a deterministic
statement about any sequence of contexts, so it is checked against the
determinant the run actually produced rather than against a quoted constant:

```
K = 10, d = 12, T = 20,000, 3 runs
  sum_t min(1, ||x_t||^2_{A^-1})  /  2 ln det A(T)
      worst run                              0.457   (holds, must be <= 1)
  ln det A(T) <= sum_a d ln((d + n_a L^2)/d)        holds
  measured regret exponent p in R(T) ~ T^p
      mean over runs                         0.006   (the sqrt(T) analysis allows p <= 0.5)
```

The second line is the AM-GM consequence `det A_a <= (tr A_a / d)^d`; both
are exact. The exponent is reported and flagged, not celebrated — see
[§7](#7-where-we-do-not-match).

## 6. Exact expected regret, with no citation involved

The strongest check in this file needs no literature at all. UCB1 and KL-UCB
are deterministic functions of the rewards they observe, so a run is a binary
tree: at each node the next arm is fixed, and the two branches carry that
arm's success and failure probabilities. Summing `gap x P(path)` over the
leaves gives `E[R(T)]` **exactly**.

```
instance [0.6, 0.4]; both policies are deterministic given the rewards
policy       T    exact E[R]   Monte Carlo  +/- 1 s.e.       z
UCB1        14      1.118369      1.117497    0.000850    1.03
KL-UCB      12      0.829564      0.834128    0.003617    1.26

KL-UCB index at an arm that has never paid out, vs 1 - exp(-L)
  max |klucb_index(0, L) - (1 - exp(-L))|, 300 random L in (0, 8]
      worst residual                      4.74e-07   (holds, tol 1e-06, at L = 5.566)
```

Two calculations sharing nothing but the policy object agree to within about
one standard error. That simultaneously exercises the environment's sampling,
the policy's index, and the regret accounting — a disagreement here could not
be explained away as a loose bound.

The last line is row 12, and it is the cheapest ground truth in the file. At
p = 0 the Bernoulli divergence collapses to `d(0, u) = -ln(1 - u)`, so the
index solving `d(0, u) = L` is exactly `1 - exp(-L)` — no bisection on the
other side of the comparison, no paper. The reported number is the *worst*
residual over 300 random levels, not a typical one, and at 4.74e-07 it sits
just inside the 1e-6 the solver advertises. Stopping the bisection one halving
early (`hi - lo < tol * 2`) doubles it to 9.51e-07 — still under the tolerance,
so the printed verdict stays "holds" and this row stays green, but far enough
from 4.74e-07 that the value pinned in `tests/test_validation.py` fails. It
takes two halvings (`tol * 4`) to reach 1.90e-06 and turn the row red.

A hand-checkable anchor, in `tests/test_validation.py`: on `[0.6, 0.4]` over
three rounds, the round-robin costs 0 + 0.2, and at t = 3 both arms have one
pull, so the bonuses are equal and UCB1 picks by empirical mean, breaking
ties towards arm 0. Only the path (miss, hit) — probability 0.16 — misroutes
it, giving `E[R(3)] = 0.2 + 0.16 x 0.2 = 0.232`, which is what the enumerator
returns.

---

## 7. Where we do not match

**KL-UCB and Thompson do not reach the Lai-Robbins constant.** At
T = 200,000, KL-UCB is at 0.88 of the floor and Thompson at 0.53, and neither
theorem is violated by that: both are asymptotic, and both policies approach
the constant from below. The honest reading is that at any horizon a person
would actually run, "asymptotically optimal" describes the slope of the
ratio, not its value. Thompson in particular is the better policy on raw
regret at every horizon here *and* the further from its own asymptote — two
facts that a single-column benchmark table would force you to conflate.

**Our LinUCB has no √T certificate.** The measured exponent p ≈ 0.006 clears
the 0.5 the analysis allows by so much that the check certifies nothing: any
correct implementation passes, and so would several incorrect ones. The
reason is the environment, not the policy — isotropic Gaussian contexts grow
every eigenvalue of `A_a` linearly, so the confidence widths shrink like
1/√t and regret is near-logarithmic. The √T rate needs contexts chosen to
keep one direction starved, which `LinearContextualBandit` never does. A
worst-case context sequence would be the way to make this row mean something.

**EXP3's default schedule is not Corollary 3.2 and does not inherit it.**
The library's `anytime_gamma(K, t) = sqrt(K ln K / ((e-1) t))` is Corollary
3.2's tuned γ with the horizon `g` replaced by the current round `t`. The
corollary is stated for a *single fixed* γ chosen from a known bound on
`G_max`; it is neither time-varying nor horizon-free, and the paper's own
answer to an unknown horizon is a different algorithm — Exp3.1, in its
Section 4, which doubles a guess at `G_max` and restarts rather than decaying
γ. The time-varying version is standard practice and behaves as advertised in
`examples/exp3_longrun.py`, but it is an adaptation carrying no cited bound,
and §4 above therefore validates the fixed-γ algorithm the theorem actually
covers. `docs/theory.md` and the `anytime_gamma` docstring say the same
thing; if you find a claim in this repository that the schedule *is* Cor 3.2,
it is a bug in the prose.

**The Lai-Robbins comparisons assume Bernoulli rewards.** For rewards merely
bounded in [0, 1], the binary KL under-states the true divergence, so
`lai_robbins_lower_bound(BernoulliBandit(means))` over-states the constant
and stops being a floor. `examples/execution_router.py` stays Bernoulli for
exactly this reason.

**Nothing here is non-stationary.** Every guarantee checked above assumes a
fixed reward distribution or a fixed reward table. The library has no drifting
environment, so EXP3's reason for existing is asserted rather than measured.

---

## References

- Lai, T. L. & Robbins, H. (1985). *Asymptotically efficient adaptive
  allocation rules.* Advances in Applied Mathematics 6(1), 4–22. — the
  `C(nu) ln T` lower bound in §1.
- Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). *Finite-time analysis of
  the multiarmed bandit problem.* Machine Learning 47, 235–256. — Theorem 1,
  the UCB1 upper bound in §1.
- Auer, P., Cesa-Bianchi, N., Freund, Y. & Schapire, R. (2002). *The
  nonstochastic multiarmed bandit problem.* SIAM Journal on Computing 32(1),
  48–77. — Theorem 3.1 and Corollary 3.2, the EXP3 bounds in §4 above; its
  Section 4 is Exp3.1, the unknown-horizon algorithm this library does *not*
  implement.
- Garivier, A. & Cappé, O. (2011). *The KL-UCB algorithm for bounded
  stochastic bandits and beyond.* COLT. — KL-UCB's asymptotic optimality, §2.
- Kaufmann, E., Korda, N. & Munos, R. (2012). *Thompson sampling: an
  asymptotically optimal finite-time analysis.* ALT. — Thompson's asymptotic
  optimality, §3.
- Li, L., Chu, W., Langford, J. & Schapire, R. (2010). *A contextual-bandit
  approach to personalized news article recommendation.* WWW. — the LinUCB
  this library implements.
- Abbasi-Yadkori, Y., Pál, D. & Szepesvári, C. (2011). *Improved algorithms
  for linear stochastic bandits.* NeurIPS. — the linear-bandit analysis whose
  elliptical-potential step is checked in §5.

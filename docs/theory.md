# Bandit theory primer

A *K-armed bandit* is a sequential decision problem. At each round
$t = 1, 2, \ldots, T$ the agent picks an arm $a_t \in \{1, \ldots, K\}$ and
observes a stochastic reward $X_{a_t, t}$ drawn from that arm's unknown
distribution. Nothing else is revealed: the rewards of the arms you did not
pull stay hidden. The goal is to maximise cumulative reward.

The benchmark is the (unknown) best arm $a^\*$ with mean $\mu^\*$, and the
currency is **regret**:

$$
R(T) \;=\; T\mu^\* - \mathbb{E}\Big[\textstyle\sum_{t=1}^{T} X_{a_t, t}\Big].
$$

[`docs/validation.md`](validation.md) is the companion to this page: the same
bounds, but measured. Everything derived below is checked there against a run
of the code, including the places where the two do not agree.

Regret is the price of not knowing which arm is best. Two versions live in
this codebase and they are not interchangeable. `cumulative_regret` subtracts
the rewards you actually observed — what a practitioner measures.
`cumulative_pseudo_regret` subtracts the *means* of the arms you pulled — what
the theory bounds. Pseudo-regret has far lower variance and is the right thing
to average over replications; observed regret is the right thing to quote when
someone asks what a deployment cost.

---

## 1. The floor: Lai & Robbins (1985)

Call a policy *consistent* if its regret is $o(T^\alpha)$ for every
$\alpha > 0$ on **every** instance in the family — that is, it is not secretly
hard-coded for one problem. Lai & Robbins showed that any consistent policy
must pull each suboptimal arm at least

$$
\mathbb{E}[N_i(T)] \;\ge\; \frac{\ln T}{D(\nu_i, \nu^\*)} - o(\ln T)
$$

times, where $D$ is the Kullback-Leibler divergence between arm $i$'s reward
distribution and the best arm's. Summing the gaps,

$$
R(T) \;\ge\; C(\nu)\ln T + o(\ln T),
\qquad
C(\nu) \;=\; \sum_{i\,:\,\mu_i < \mu^\*} \frac{\mu^\* - \mu_i}{D(\nu_i, \nu^\*)}.
$$

The intuition is a hypothesis test. To be confident arm $i$ is worse than
$a^\*$ you must gather enough evidence to reject "arm $i$ is really $\nu^\*$ in
disguise", and the sample cost of that test is $1/D$. Arms that are nearly
indistinguishable are expensive; arms that are obviously bad are cheap.

`lai_robbins_lower_bound(env)` returns $C(\nu)$. The divergence is chosen by
the environment's **family**, never by the numeric range of its means:

| family | divergence used | edge cases |
|---|---|---|
| `BernoulliBandit` | binary KL $d(\mu_i, \mu^\*)$ | $\mu^\* = 1$ or $\mu_i = 0$ give $D = \infty$; such an arm is separated after finitely many pulls and contributes $0$ to $C$ |
| `GaussianBandit` | $(\mu^\* - \mu_i)^2 / 2\sigma_i^2$ | uses each arm's own $\sigma_i$ — an arm five times wider is twenty-five times more expensive to rule out |
| anything else | `kl=lambda arm: ...` | raises `TypeError` rather than guessing |

Two instances used throughout these docs, for calibration:

| instance | $C(\nu)$ | floor at $T = 10^5$ |
|---|---:|---:|
| `[0.50, 0.45, 0.40, 0.35, 0.30]` | 20.66 | 238 |
| `[0.10, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]` | 17.45 | 201 |

---

## 2. The ceiling: index policies

An *index policy* scores every arm with an optimistic estimate of its mean and
pulls the argmax. The entire design question is how optimistic to be.

**UCB1** (Auer, Cesa-Bianchi & Fischer 2002) answers with Hoeffding's
inequality, which knows only that rewards lie in $[0, 1]$:

$$
\text{UCB}_i(t) \;=\; \hat\mu_i + \sqrt{\frac{2\ln t}{n_i}}.
$$

**KL-UCB** (Garivier & Cappé 2011) answers with the exact Chernoff rate:

$$
u_i(t) \;=\; \max\\{\, q \in [\hat\mu_i, 1] \;:\; n_i \\, d(\hat\mu_i, q) \le \ln t + c\ln\ln t \,\\}.
$$

Both say "assume each arm is as good as the data still permits". They differ
only in what *permits* means. Since $q \mapsto d(\hat\mu_i, q)$ is continuous
and strictly increasing on $[\hat\mu_i, 1]$, the KL index has no closed form
but yields to bisection in about twenty halvings — that is all `klucb_index`
does.

### Why KL-UCB is optimal and UCB1 is not

Pinsker's inequality states $d(p, q) \ge 2(p - q)^2$, with equality only at
$p = q = 1/2$. Inverting it turns the KL index into exactly the Hoeffding
bonus. **UCB1's confidence width is the Pinsker relaxation of KL-UCB's** —
never tighter, and badly loose whenever the means sit far from $1/2$:

| $p$ vs $q$ | $d(p, q)$ | $2(p-q)^2$ | KL tighter by |
|---|---:|---:|---:|
| 0.45 vs 0.50 | 0.0050 | 0.0050 | 1.00x |
| 0.30 vs 0.50 | 0.0823 | 0.0800 | 1.03x |
| 0.02 vs 0.10 | 0.0513 | 0.0128 | 4.01x |
| 0.01 vs 0.10 | 0.0713 | 0.0162 | 4.40x |

An index policy pulls arm $i$ roughly $\ln t$ divided by its divergence times,
so a divergence that is 4.4x tighter means 4.4x fewer wasted pulls. Near
$p = 1/2$ the two agree and UCB1 is perfectly good; in the click-through-rate
regime that motivates most deployments — means of one or two percent — it
over-explores by close to an order of magnitude. That gap is the whole content
of the two panels in `docs/optimality.png`.

Garivier & Cappé prove $R(T) \le C(\nu)\ln T + O(\ln\ln T)$ for KL-UCB:
it matches the Lai-Robbins constant itself, not a multiple of it. Their
finite-time analysis wants $c \ge 3$ on the $\ln\ln t$ term; in practice
$c = 0$ is uniformly better, so that is this library's default.

**Thompson sampling** reaches the same constant by a different route
(Kaufmann, Korda & Munos 2012). Draw $\tilde\mu_i$ from each posterior and
pull the argmax, so each arm is explored in proportion to the posterior
probability that it is best. It approaches $C(\nu)$ from below, and slowly: at
$T = 2\times10^5$ it is still running at roughly half the floor.

### Bounds at a glance

| algorithm | regret bound | notes |
|---|---|---|
| EpsilonGreedy (fixed $\varepsilon$) | $\Theta(T)$ | linear — the exploration rate never decays |
| EpsilonGreedy ($\varepsilon_t \propto 1/t$) | $O(\log T)$ | right rate, loose constant |
| UCB1 | $O(\sqrt{KT\log T})$ | problem-dependent $\sum_i 8\ln T/\Delta_i$: a multiple of $C(\nu)$ |
| KL-UCB | $C(\nu)\ln T + O(\ln\ln T)$ | asymptotically optimal for Bernoulli rewards |
| Thompson sampling | $C(\nu)\ln T + O(1)$ | asymptotically optimal; slow finite-time approach |
| EXP3 (fixed $\gamma$ tuned to $T$) | $O(\sqrt{KT\ln K})$ | adversarial — no stochastic assumption at all |

---

## 3. Adversarial rewards: EXP3

Everything above assumes each arm has a *fixed* distribution. Drop that and
the lower bound changes shape: against an adversary who chooses the reward
sequence, the best achievable regret is $\Theta(\sqrt{KT})$ and no policy can
be logarithmic. EXP3 (Auer, Cesa-Bianchi, Freund & Schapire 2002) attains it
with exponential weights over importance-weighted reward estimates:

$$
p_i(t) = (1 - \gamma)\frac{w_i(t)}{\sum_j w_j(t)} + \frac{\gamma}{K},
\qquad
w_i \leftarrow w_i \exp\\!\Big(\frac{\gamma}{K}\cdot\frac{X_t}{p_i(t)}\Big).
$$

Two implementation facts matter more than they look:

- **Keep the weights in log-space.** The recursion multiplies by
  $\exp(\gamma\hat x / K)$ every round with no normalisation, so linear-space
  weights overflow to `inf` within a few thousand rounds — at $\gamma = 0.2$,
  around $t \approx 9{,}000$. After that the probabilities are `NaN` and the
  policy silently degenerates to a single arm. `EXP3` stores $\log w$ and
  samples from a max-shifted softmax; the distribution is invariant to the
  shift, so nothing about the algorithm changes except that it survives.
- **The importance weight must be the probability the arm was actually drawn
  from.** Recomputing $p_i$ at update time is correct only when `select` and
  `update` strictly alternate. Caching it inside `select` keeps the estimator
  unbiased under batching too.

This library's default is the decaying schedule
$\gamma_t = \min\\{1, \sqrt{K\ln K / ((e-1)t)}\\}$ — Corollary 3.2's tuned
$\gamma$ with the horizon $g$ replaced by the current round $t$. Worth being
precise about what that inherits, which is nothing: Cor 3.2 fixes a *single*
$\gamma$ from a known bound $g \ge G_{\max}$ and bounds the weak regret by
$2\sqrt{e-1}\sqrt{gK\ln K}$. It is not time-varying and it is not
horizon-free. The paper's own answer to an unknown horizon is a different
algorithm, Exp3.1 (§4), which doubles a guess at $G_{\max}$ and restarts
rather than decaying $\gamma$. The substitution above is standard practice
and is justified here by measurement, not by citation: §4 of
[`docs/validation.md`](validation.md) checks the fixed-$\gamma$ algorithm the
theorem actually covers, and `docs/exp3_longrun.png` shows what the decay
buys. A fixed $\gamma$ is optimal only when tuned to a known $T$; left
running, it pays a $\gamma t$ exploration tax forever — linear regret in slow
motion, 5,014 against 234 at $T = 10^5$.

---

## 4. Contexts: LinUCB

The context-free setting assumes one arm is best, full stop. In recommendation
or dose-finding the best arm depends on a feature vector $x_t$ revealed before
the choice. LinUCB (Li, Chu, Langford & Schapire 2010) assumes
$\mathbb{E}[r \mid a, x] = \theta_a^{\top}x$ and applies optimism inside that
linear model:

$$
p_a = \hat\theta_a^{\top}x + \alpha\sqrt{x^{\top}A_a^{-1}x},
\qquad
A_a = I + \sum_{s\,:\,a_s = a} x_s x_s^{\top},
\qquad
\hat\theta_a = A_a^{-1}b_a.
$$

The bonus is the width of the confidence ellipsoid *in the direction of the
current context* — large when $x$ points somewhere arm $a$ has rarely been
tried. Regret is $\tilde O(d\sqrt{T})$: sublinear in $T$, and independent of
$K$ except through the model. The sharpest version of that analysis is
Abbasi-Yadkori, Pál & Szepesvári (2011), whose central step — the elliptical
potential $\sum_t \min(1, \|x_t\|^2_{A_{t-1}^{-1}}) \le 2\ln\det A_T$ — is
checked against this implementation in
[`docs/validation.md`](validation.md) §5.

$A_a$ only ever changes by a rank-1 update, so its inverse follows from
Sherman-Morrison:

$$
(A + xx^{\top})^{-1} = A^{-1} - \frac{(A^{-1}x)(x^{\top}A^{-1})}{1 + x^{\top}A^{-1}x}.
$$

That is $O(d^2)$ per arm per round instead of $O(d^3)$ for a fresh
factorisation, and it is numerically safe here because the denominator is at
least 1 whenever $A$ is positive definite — and $A$ starts at $I$. The tests
pin `A_inv` to `np.linalg.inv(A)` within `1e-10` after 2,000 updates and
assert the two implementations choose identical arms for 1,000 rounds.

The comparison worth internalising is `docs/contextual.png`. On a linear
contextual instance with random $\theta_a$, every arm has the same *marginal*
mean, so a context-free policy has no signal at all and accrues linear regret:
11,091 over 4,000 rounds, against LinUCB's 49.

---

## 5. Choosing

- **Stationary, Bernoulli-ish, means well below 1/2** → KL-UCB. This is where
  the shape of the index is worth real money.
- **Stationary, and you care about behaviour at $T \le 10^5$** → Thompson.
  Simple, fast, and empirically the strongest at practical horizons.
- **Bounded rewards, unknown family, want something defensible** → UCB1. It is
  loose, but it is never wrong.
- **Rewards may be adversarial or drift** → EXP3 with `gamma=None`.
- **The best arm depends on side information** → LinUCB. Nothing context-free
  can help you.

## References

- Lai, T. L. & Robbins, H. (1985). *Asymptotically efficient adaptive
  allocation rules.* Advances in Applied Mathematics 6(1), 4–22.
- Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). *Finite-time analysis of the
  multiarmed bandit problem.* Machine Learning 47, 235–256.
- Auer, P., Cesa-Bianchi, N., Freund, Y. & Schapire, R. (2002). *The
  nonstochastic multiarmed bandit problem.* SIAM Journal on Computing 32(1),
  48–77.
- Garivier, A. & Cappé, O. (2011). *The KL-UCB algorithm for bounded stochastic
  bandits and beyond.* COLT.
- Li, L., Chu, W., Langford, J. & Schapire, R. (2010). *A contextual-bandit
  approach to personalized news article recommendation.* WWW.
- Kaufmann, E., Korda, N. & Munos, R. (2012). *Thompson sampling: an
  asymptotically optimal finite-time analysis.* ALT.
- Abbasi-Yadkori, Y., Pál, D. & Szepesvári, C. (2011). *Improved algorithms
  for linear stochastic bandits.* NeurIPS.
- Lattimore, T. & Szepesvári, C. (2020). *Bandit Algorithms.* Cambridge
  University Press — the standard modern reference for all of the above.

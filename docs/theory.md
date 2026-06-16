# Bandit theory primer

A *K-armed bandit* is a sequential decision-making problem. At each round
\(t = 1, 2, \ldots, T\), the agent picks an arm \(a_t \in \{1, \ldots, K\}\)
and observes a stochastic reward \(X_{a_t, t}\). The arm's reward distribution
is unknown. The goal: maximize cumulative reward.

The benchmark is the (unknown) optimal arm \(a^*\), with mean \(\mu^*\).
**Cumulative regret** is

\[
R(T) = T \cdot \mu^* - \mathbb{E}\left[ \sum_{t=1}^T X_{a_t, t} \right].
\]

## Lower bound (Lai-Robbins 1985)

For any consistent policy on K Bernoulli arms, the asymptotic regret is at least

\[
R(T) \geq \sum_{i : \mu_i < \mu^*} \frac{\mu^* - \mu_i}{\mathrm{KL}(\mu_i \| \mu^*)} \ln T + o(\ln T)
\]

This sets the bar for "optimal" — no algorithm can do better asymptotically.

## Upper bounds achieved by `gauss-bandit` algorithms

| algorithm           | regret bound          | notes                                       |
|---------------------|-----------------------|---------------------------------------------|
| EpsilonGreedy (fix) | \(\Theta(T)\)      | linear unless annealed                      |
| EpsilonGreedy (1/t) | \(O(\log T)\)      | matches LR up to constants                  |
| UCB1                | \(O(\sqrt{KT \log T})\) | distribution-free; problem-dependent \(O(\log T)\) |
| Thompson sampling   | \(O(\log T)\)      | asymptotically optimal in many regimes      |
| EXP3                | \(O(\sqrt{KT \log K})\) | adversarial setting                  |

## When to use which

- **Bernoulli rewards, stationary** -> Thompson is usually best in practice.
- **Bounded but unknown distribution** -> UCB1 is a safe default.
- **Reward distribution may change adversarially** -> EXP3.
- **You want a baseline / homework solution** -> EpsilonGreedy with annealing.

## References

- Lai & Robbins (1985), "Asymptotically efficient adaptive allocation rules."
- Auer, Cesa-Bianchi, Fischer (2002), "Finite-time analysis of the multi-armed
  bandit problem."
- Auer, Cesa-Bianchi, Freund, Schapire (2002), "The nonstochastic multi-armed
  bandit problem."
- Russo, Van Roy, Kazerouni, Osband, Wen (2018), "A tutorial on Thompson sampling."

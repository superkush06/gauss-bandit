"""KL-UCB (Garivier & Cappe 2011) — the index that meets the Lai-Robbins floor.

UCB1 bounds the deviation of an empirical mean with Hoeffding's inequality,
which only knows that rewards live in [0, 1]. For Bernoulli arms that is
wasteful: near p = 0 or p = 1 the true concentration is far sharper than the
quadratic bound. KL-UCB replaces the sqrt bonus with the exact Chernoff rate,

    u_i(t) = max { q in [p_hat_i, 1] : n_i * d(p_hat_i, q) <= ln t + c ln ln t }

where d is the Bernoulli KL divergence — the *same* divergence that appears in
the Lai-Robbins lower bound. That is why KL-UCB is asymptotically optimal:
its exploration budget for arm i is exactly ln t / d(mu_i, mu*) pulls, which
is the bound, not a constant multiple of it.

The index has no closed form, but q -> d(p_hat, q) is continuous and strictly
increasing on [p_hat, 1], so bisection converges in ~20 halvings to 1e-6.
"""

from __future__ import annotations

import math

from ..metrics import bernoulli_kl
from .base import Algorithm


def klucb_index(p_hat: float, level: float, tol: float = 1e-6,
                max_iter: int = 40) -> float:
    """Largest q in [p_hat, 1] with ``bernoulli_kl(p_hat, q) <= level``.

    Solved by bisection on the monotone map q -> d(p_hat, q). ``level`` is the
    per-arm exploration budget (ln t + c ln ln t) / n_i.

    The root is located to ``tol`` **on the q axis**, which is the axis the
    policy compares on. Near q = 1 the divergence is near-vertical, so the
    residual d(p_hat, q) - level at the returned point can still be large;
    that is a property of the KL, not slack in the solver.
    """
    if not 0.0 <= p_hat <= 1.0:
        raise ValueError("p_hat must be in [0, 1]")
    if level <= 0.0:
        return p_hat
    if p_hat >= 1.0:
        return 1.0
    # d(p, 1) = inf for every p < 1, so no finite budget ever reaches the
    # ceiling and the bracket [p_hat, 1] is always valid from here.
    lo, hi = p_hat, 1.0
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if bernoulli_kl(p_hat, mid) > level:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


class KLUCB(Algorithm):
    """KL-UCB for Bernoulli (or [0, 1]-bounded) rewards.

    Args:
        n_arms: number of arms.
        c: coefficient on the ``ln ln t`` term. The finite-time proof needs
            c >= 3; Garivier & Cappe report — and we reproduce — that c = 0 is
            uniformly better in practice, so that is the default.

    Untried arms are pulled first, exactly as in :class:`~bandit.algos.UCB1`,
    so the two policies differ only in the shape of their confidence bonus.
    """

    def __init__(self, n_arms: int, c: float = 0.0) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        if c < 0:
            raise ValueError("c must be >= 0")
        self.n_arms = n_arms
        self.c = c
        self.reset()

    def reset(self) -> None:
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms

    def _budget(self, t: int) -> float:
        """ln t + c ln ln t, floored at 0 so the first rounds stay well-defined."""
        log_t = math.log(max(t, 2))
        if self.c == 0.0:
            return log_t
        return log_t + self.c * math.log(max(log_t, 1.0))

    def index(self, arm: int, t: int) -> float:
        """The KL upper confidence bound for `arm` at round `t`."""
        n = self.counts[arm]
        if n == 0:
            return math.inf
        return klucb_index(self.values[arm], self._budget(t) / n)

    def select(self, t: int) -> int:
        for arm, n in enumerate(self.counts):
            if n == 0:
                return arm
        best_arm, best_idx = 0, -math.inf
        for arm in range(self.n_arms):
            idx = self.index(arm, t)
            if idx > best_idx:
                best_arm, best_idx = arm, idx
        return best_arm

    def update(self, arm: int, reward: float) -> None:
        if not 0.0 <= reward <= 1.0:
            raise ValueError("KL-UCB expects reward in [0, 1]")
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

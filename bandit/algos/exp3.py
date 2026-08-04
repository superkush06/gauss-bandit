"""EXP3 — adversarial bandits (Auer, Cesa-Bianchi, Freund, Schapire 2002).

Maintains one weight per arm **in log-space** and samples from a max-shifted
softmax, so weights never overflow no matter how long the run. Updates use
importance-weighted reward estimates computed against the distribution that
was actually used at select() time.
"""

from __future__ import annotations

import math
import random

from .base import Algorithm

_E_MINUS_1 = math.e - 1.0

# Renormalize log-weights once the max exceeds this; subtracting a common
# constant from every log-weight leaves the sampling distribution unchanged.
_LOG_SHIFT_THRESHOLD = 50.0


def anytime_gamma(n_arms: int, t: int) -> float:
    """Decaying exploration rate: gamma_t = min(1, sqrt(K ln K / ((e-1) t))).

    This is the tuned gamma of Auer et al. (2002), Corollary 3.2, with the
    horizon g replaced by the current round t. The corollary covers a *fixed*
    gamma chosen from a known bound g >= G_max and says nothing about a
    time-varying schedule; the paper's own answer to an unknown horizon is a
    different algorithm, Exp3.1 (Section 4), which doubles a guess at G_max
    and restarts. So this schedule does not inherit Corollary 3.2's bound and
    is not claimed to: it is the standard adaptation, justified here by
    measurement (examples/exp3_longrun.py, and section 7 of
    docs/validation.md) rather than by citation. What it buys is that
    exploration decays instead of costing a linear gamma*t forever.
    """
    if t < 1:
        raise ValueError("t must be >= 1")
    if n_arms < 2:
        return 1.0
    return min(1.0, math.sqrt(n_arms * math.log(n_arms) / (_E_MINUS_1 * t)))


class EXP3(Algorithm):
    """EXP3 for adversarial bandits, rewards in [0, 1].

    With exploration rate gamma in (0, 1], the algorithm picks arm i with
    probability  p_i = (1-gamma) * softmax(log_w)_i + gamma/K.

    Pass ``gamma=None`` (the default) to use the anytime schedule
    ``anytime_gamma`` — the right choice when the horizon is unknown.
    A fixed gamma keeps exploring at a constant rate forever, which is only
    optimal when tuned to a known horizon.
    """

    def __init__(self, n_arms: int, gamma: float | None = None,
                 seed: int | None = None) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        if gamma is not None and not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1] (or None for anytime)")
        self.n_arms = n_arms
        self.gamma = gamma
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self._log_w = [0.0] * self.n_arms
        self._t = 1
        # Distribution (and gamma) cached at select() time so update() uses
        # the probabilities the arm was *actually* drawn from.
        self._cached: tuple[list[float], float] | None = None

    @property
    def weights(self) -> list[float]:
        """Arm weights in linear space (max-shifted; relative scale only)."""
        m = max(self._log_w)
        shift = m if m > _LOG_SHIFT_THRESHOLD else 0.0
        return [math.exp(lw - shift) for lw in self._log_w]

    def _gamma_at(self, t: int) -> float:
        return self.gamma if self.gamma is not None else anytime_gamma(self.n_arms, t)

    def _probs(self, gamma: float | None = None) -> list[float]:
        if gamma is None:
            gamma = self._gamma_at(self._t)
        m = max(self._log_w)
        ws = [math.exp(lw - m) for lw in self._log_w]
        total = sum(ws)
        return [(1.0 - gamma) * (w / total) + gamma / self.n_arms for w in ws]

    def select(self, t: int) -> int:
        self._t = max(t, 1)
        gamma = self._gamma_at(self._t)
        probs = self._probs(gamma)
        self._cached = (probs, gamma)
        r = self._rng.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return i
        return self.n_arms - 1

    def update(self, arm: int, reward: float) -> None:
        if not 0.0 <= reward <= 1.0:
            raise ValueError("EXP3 expects reward in [0, 1]")
        if self._cached is not None:
            probs, gamma = self._cached
            self._cached = None  # one update per select; then fall back to fresh
        else:
            gamma = self._gamma_at(self._t)
            probs = self._probs(gamma)
        x_hat = reward / probs[arm]
        self._log_w[arm] += gamma * x_hat / self.n_arms
        m = max(self._log_w)
        if m > _LOG_SHIFT_THRESHOLD:
            self._log_w = [lw - m for lw in self._log_w]

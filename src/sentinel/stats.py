"""Constant-memory online estimators.

Nothing here stores an unbounded history. A detector that needs the 99.5th
percentile of its own score distribution over 10 years of ticks uses ~200
bytes, not 10 years of ticks. That is the difference between a research
notebook and something you can leave running on a feed handler.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable

import numpy as np

BENFORD_P1 = np.array([math.log10(1 + 1 / d) for d in range(1, 10)])


def _alpha_from_halflife(halflife: float) -> float:
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    return 1.0 - math.exp(math.log(0.5) / halflife)


class EWMA:
    """Exponentially weighted mean with an explicit warm-up gate."""

    __slots__ = ("alpha", "warmup", "_value", "n")

    def __init__(self, halflife: float, warmup: int = 10) -> None:
        self.alpha = _alpha_from_halflife(halflife)
        self.warmup = warmup
        self._value: float | None = None
        self.n = 0

    def update(self, x: float) -> float:
        self.n += 1
        if self._value is None:
            self._value = float(x)
        else:
            self._value += self.alpha * (float(x) - self._value)
        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    @property
    def ready(self) -> bool:
        return self.n >= self.warmup


class EWMVariance:
    """West's incremental exponentially weighted mean/variance."""

    __slots__ = ("alpha", "warmup", "mean", "var", "n")

    def __init__(self, halflife: float, warmup: int = 20) -> None:
        self.alpha = _alpha_from_halflife(halflife)
        self.warmup = warmup
        self.mean: float | None = None
        self.var = 0.0
        self.n = 0

    def update(self, x: float) -> None:
        x = float(x)
        self.n += 1
        if self.mean is None:
            self.mean = x
            self.var = 0.0
            return
        diff = x - self.mean
        incr = self.alpha * diff
        self.mean += incr
        self.var = (1.0 - self.alpha) * (self.var + diff * incr)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 0.0))

    @property
    def ready(self) -> bool:
        return self.n >= self.warmup


class P2Quantile:
    """Jain & Chlamtac's P-square algorithm: streaming quantiles in O(1).

    Used for two things a rolling window cannot do cheaply:
      * a robust *center* (the running median) that a fat-finger print cannot
        drag around the way a mean can;
      * self-calibrating thresholds, e.g. "alert above the 99.5th percentile
        of this detector's own historical score".
    """

    __slots__ = ("q", "count", "_buf", "heights", "n", "ns", "dn")

    def __init__(self, q: float) -> None:
        if not 0.0 < q < 1.0:
            raise ValueError("q must be in (0, 1)")
        self.q = q
        self.count = 0
        self._buf: list[float] = []
        self.heights: list[float] = []
        self.n: list[float] = []
        self.ns: list[float] = []
        self.dn: list[float] = []

    def update(self, x: float) -> float | None:
        x = float(x)
        self.count += 1
        if self.count <= 5:
            self._buf.append(x)
            if self.count == 5:
                self.heights = sorted(self._buf)
                self.n = [0.0, 1.0, 2.0, 3.0, 4.0]
                q = self.q
                self.ns = [0.0, 2.0 * q, 4.0 * q, 2.0 + 2.0 * q, 4.0]
                self.dn = [0.0, q / 2.0, q, (1.0 + q) / 2.0, 1.0]
            return self.value

        h = self.heights
        if x < h[0]:
            h[0] = x
            k = 0
        elif x >= h[4]:
            h[4] = x
            k = 3
        else:
            k = 3
            for i in range(1, 5):
                if x < h[i]:
                    k = i - 1
                    break

        for i in range(k + 1, 5):
            self.n[i] += 1.0
        for i in range(5):
            self.ns[i] += self.dn[i]

        for i in range(1, 4):
            d = self.ns[i] - self.n[i]
            if (d >= 1 and self.n[i + 1] - self.n[i] > 1) or (
                d <= -1 and self.n[i - 1] - self.n[i] < -1
            ):
                sign = 1.0 if d > 0 else -1.0
                cand = self._parabolic(i, sign)
                if h[i - 1] < cand < h[i + 1]:
                    h[i] = cand
                else:
                    h[i] = self._linear(i, sign)
                self.n[i] += sign
        return self.value

    def _parabolic(self, i: int, d: float) -> float:
        h, n = self.heights, self.n
        return h[i] + d / (n[i + 1] - n[i - 1]) * (
            (n[i] - n[i - 1] + d) * (h[i + 1] - h[i]) / (n[i + 1] - n[i])
            + (n[i + 1] - n[i] - d) * (h[i] - h[i - 1]) / (n[i] - n[i - 1])
        )

    def _linear(self, i: int, d: float) -> float:
        h, n = self.heights, self.n
        j = i + int(d)
        return h[i] + d * (h[j] - h[i]) / (n[j] - n[i])

    @property
    def value(self) -> float | None:
        if self.count == 0:
            return None
        if self.count < 5:
            return float(np.median(self._buf))
        return self.heights[2]


class RobustScale:
    """Online median + MAD-equivalent scale.

    scale = 1.4826 * EWMA(|x - median|), which is a consistent estimator of
    sigma under normality but has a ~50% breakdown point. A single 20-sigma
    print inflates a rolling stdev enough to mask the *next* one; it barely
    moves this.
    """

    __slots__ = ("_median", "_dev", "floor", "warmup", "n")

    def __init__(self, halflife: float = 100.0, warmup: int = 50, floor: float = 1e-9):
        self._median = P2Quantile(0.5)
        self._dev = EWMA(halflife)
        self.floor = floor
        self.warmup = warmup
        self.n = 0

    def score(self, x: float) -> float | None:
        """Return |x - center| / scale using state from *before* x, then update.

        Order matters: scoring first and updating second is what makes the
        detector point-in-time correct.
        """
        med = self._median.value
        dev = self._dev.value
        out: float | None = None
        if self.n >= self.warmup and med is not None and dev is not None:
            scale = max(1.4826 * dev, self.floor)
            out = abs(x - med) / scale
        self._median.update(x)
        self._dev.update(abs(x - (med if med is not None else x)))
        self.n += 1
        return out

    @property
    def center(self) -> float | None:
        return self._median.value

    @property
    def scale(self) -> float | None:
        dev = self._dev.value
        return None if dev is None else max(1.4826 * dev, self.floor)


class Latch:
    """Schmitt-trigger hysteresis.

    Fires once when a score crosses `enter` and stays latched until it falls
    back under `exit_`. Without this, a 90-minute volatility event produces
    4,000 alerts instead of one, and the desk stops reading alerts.
    """

    __slots__ = ("enter", "exit_", "_on")

    def __init__(self, enter: float, exit_: float) -> None:
        if exit_ > enter:
            raise ValueError("exit threshold must be <= enter threshold")
        self.enter = enter
        self.exit_ = exit_
        self._on = False

    def push(self, score: float) -> bool:
        """True only on the rising edge of an episode."""
        if self._on:
            if score < self.exit_:
                self._on = False
            return False
        if score >= self.enter:
            self._on = True
            return True
        return False

    @property
    def latched(self) -> bool:
        return self._on


def ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage toward a scaled identity target.

    With T=250 daily returns and N=40 names, the sample covariance is close to
    singular and its smallest eigenvalues are noise. Inverting it (which is
    exactly what a Mahalanobis distance does) turns that noise into enormous
    fake anomaly scores. Shrinkage is the cheap fix.

    Returns (covariance, shrinkage_intensity).
    """
    x = np.asarray(x, dtype=float)
    t, n = x.shape
    if t < 2:
        raise ValueError("need at least 2 observations")
    xc = x - x.mean(axis=0, keepdims=True)
    sample = xc.T @ xc / t
    mu = float(np.trace(sample) / n)
    target = mu * np.eye(n)

    d2 = float(((sample - target) ** 2).sum())
    if d2 <= 0:
        return target, 1.0

    b2 = 0.0
    for row in xc:
        outer = np.outer(row, row)
        b2 += float(((outer - sample) ** 2).sum())
    b2 /= t * t
    b2 = min(b2, d2)

    shrink = b2 / d2
    return (1.0 - shrink) * sample + shrink * target, shrink


def benford_chi2(values: Iterable[float]) -> tuple[float, int]:
    """Chi-square statistic of leading-digit frequencies vs Benford's law."""
    digits: list[int] = []
    for v in values:
        v = abs(float(v))
        if v <= 0 or not math.isfinite(v):
            continue
        d = int(str(v).replace(".", "").lstrip("0")[:1] or 0)
        if 1 <= d <= 9:
            digits.append(d)
    m = len(digits)
    if m == 0:
        return 0.0, 0
    observed = np.bincount(np.array(digits), minlength=10)[1:10].astype(float)
    expected = BENFORD_P1 * m
    return float(((observed - expected) ** 2 / expected).sum()), m


class RingBuffer:
    """Fixed-capacity float buffer with O(1) append and array export."""

    __slots__ = ("_dq", "capacity")

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._dq: deque[float] = deque(maxlen=capacity)

    def append(self, x: float) -> None:
        self._dq.append(float(x))

    def __len__(self) -> int:
        return len(self._dq)

    @property
    def full(self) -> bool:
        return len(self._dq) == self.capacity

    def array(self) -> np.ndarray:
        return np.fromiter(self._dq, dtype=float, count=len(self._dq))

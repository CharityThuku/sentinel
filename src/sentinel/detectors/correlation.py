"""Correlation-structure breakdown.

Every leg of the portfolio moves within one sigma and the portfolio still
blows up, because the legs that always moved together stopped. Univariate
detectors are blind to this by construction.

Mahalanobis distance on the joint return vector catches it, but only with
three pieces of hygiene that a notebook version would skip:

  * Ledoit-Wolf shrinkage. A Mahalanobis distance inverts the covariance
    matrix; with T=250 and N=10 the smallest eigenvalues are mostly noise,
    and inverting noise manufactures enormous fake anomaly scores.

  * A threshold learned from the detector's own score history (P-square
    99.5th percentile) rather than a chi-square critical value. The
    chi-square value assumes Gaussian returns and is therefore wrong every
    single day. The detector instead learns what "normal weird" looks like
    for this book.

  * Quarantine. The covariance is refitted on a rolling window, so a
    sustained correlation break contaminates its own baseline within a few
    dozen ticks and the detector quietly goes blind mid-event. While an
    episode is latched, the estimator stops learning.

Two channels, because a correlation break is not always a single violent
tick: an instantaneous channel for joint outliers, and a smoothed channel
that catches a regime where distances run persistently hot without any
single tick being extreme.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..stats import EWMA, Latch, P2Quantile, ledoit_wolf
from ..types import MARKET_WIDE, Alert, Snapshot, make_alert
from .base import Detector


class CorrelationBreakdown(Detector):
    name = "correlation"

    def __init__(
        self,
        window: int = 250,
        refit_every: int = 100,
        quantile: float = 0.995,
        multiplier: float = 1.8,
        sustained_halflife: float = 60.0,
        sustained_quantile: float = 0.99,
        sustained_multiplier: float = 1.3,
        max_quarantine: int = 400,
        warmup: int = 500,
        min_universe: int = 4,
    ) -> None:
        self.window = window
        self.refit_every = refit_every
        self.multiplier = multiplier
        self.sustained_multiplier = sustained_multiplier
        self.max_quarantine = max_quarantine
        self.warmup = warmup
        self._quarantine_for = 0
        self.min_universe = min_universe
        self._hist: deque[np.ndarray] = deque(maxlen=window)
        self._last_price: dict[str, float] = {}
        self._symbols: tuple[str, ...] | None = None
        self._precision: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._shrink = 0.0
        self._since_fit = 0
        self._n = 0
        self._q = P2Quantile(quantile)
        self._smooth = EWMA(sustained_halflife)
        self._q_smooth = P2Quantile(sustained_quantile)
        self._latch = Latch(1.0, 0.7)
        self._latch_smooth = Latch(1.0, 0.75)

    @property
    def quarantined(self) -> bool:
        """True while an episode is live: stop *learning* from anomalous data.

        Measurement never freezes - only the baseline does. Freezing the
        measurement too was the first version of this, and it deadlocked: the
        smoothed statistic could no longer fall back below its exit level, so
        the latch never released and the detector went permanently blind after
        its first alert. Quarantine also expires, because a genuine regime
        change must eventually become the new normal.
        """
        if self._quarantine_for >= self.max_quarantine:
            return False
        return self._latch.latched or self._latch_smooth.latched

    def _refit(self) -> None:
        x = np.vstack(self._hist)
        cov, shrink = ledoit_wolf(x)
        self._shrink = shrink
        self._mean = x.mean(axis=0)
        self._precision = np.linalg.pinv(cov)
        self._since_fit = 0

    def _returns(self, snap: Snapshot) -> np.ndarray | None:
        assert self._symbols is not None
        vals: list[float] = []
        ok = True
        for sym in self._symbols:
            tick = snap.ticks.get(sym)
            prev = self._last_price.get(sym)
            if tick is None or prev is None or prev <= 0 or tick.price <= 0:
                ok = False
                vals.append(0.0)
            else:
                vals.append(math.log(tick.price / prev))
        for sym in self._symbols:
            tick = snap.ticks.get(sym)
            if tick is not None:
                self._last_price[sym] = tick.price
        return np.array(vals) if ok else None

    def _drivers(self, d: np.ndarray, d2: float) -> str:
        assert self._symbols is not None and self._precision is not None
        contrib = d * (self._precision @ d)
        order = np.argsort(-np.abs(contrib))[:3]
        return ", ".join(f"{self._symbols[i]} ({contrib[i] / d2:.0%})" for i in order)

    def update(self, snap: Snapshot) -> list[Alert]:
        if self._symbols is None:
            self._symbols = snap.symbols()
        if len(self._symbols) < self.min_universe:
            return []
        r = self._returns(snap)
        if r is None:
            return []
        self._n += 1

        out: list[Alert] = []
        if self._precision is None or self._mean is None:
            self._hist.append(r)
            self._since_fit += 1
            if len(self._hist) == self.window:
                self._refit()
            return out

        d = r - self._mean
        d2 = float(d @ self._precision @ d)
        thr = self._q.value
        thr_smooth = self._q_smooth.value

        learning = not self.quarantined
        self._quarantine_for = 0 if learning else self._quarantine_for + 1

        # measurement always advances; only the baseline is quarantined
        smooth = self._smooth.update(d2)
        if learning:
            # winsorise before learning: a genuine 5,000-sigma print must not
            # be allowed to redefine what "extreme" means for the next month
            self._q.update(min(d2, 3.0 * thr) if thr else d2)
            self._q_smooth.update(
                min(smooth, 3.0 * thr_smooth) if thr_smooth else smooth
            )

        if self._n >= self.warmup and thr is not None and thr > 0:
            norm = d2 / (thr * self.multiplier)
            if self._latch.push(norm):
                out.append(
                    make_alert(
                        snap, MARKET_WIDE, self.name, "correlation_breakdown",
                        norm, 1.0,
                        f"joint move inconsistent with covariance structure "
                        f"(d2={d2:.1f} vs adaptive limit {thr * self.multiplier:.1f}); "
                        f"drivers: {self._drivers(d, d2)}",
                        mahalanobis_d2=round(d2, 3),
                        shrinkage=round(self._shrink, 3),
                    )
                )
            if (
                smooth is not None
                and thr_smooth is not None
                and thr_smooth > 0
                and not self._latch.latched
            ):
                norm_s = smooth / (thr_smooth * self.sustained_multiplier)
                if self._latch_smooth.push(norm_s):
                    out.append(
                        make_alert(
                            snap, MARKET_WIDE, self.name, "correlation_regime_break",
                            norm_s, 1.0,
                            f"joint distances running persistently hot "
                            f"(smoothed d2={smooth:.1f} vs "
                            f"{thr_smooth * self.sustained_multiplier:.1f}) - "
                            f"the structure has changed, not one print; "
                            f"drivers: {self._drivers(d, d2)}",
                            smoothed_d2=round(smooth, 3),
                            shrinkage=round(self._shrink, 3),
                        )
                    )

        if learning:
            self._hist.append(r)
            self._since_fit += 1
            if len(self._hist) == self.window and self._since_fit >= self.refit_every:
                self._refit()
        return out

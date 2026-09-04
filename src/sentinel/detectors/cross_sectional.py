"""Idiosyncratic move detection.

A single-name detector cannot tell you anything a PM cares about. If the name
is down 4% and the market is down 4%, nothing happened. If the name is down
4% and the market is flat, someone knows something you don't.

So: strip the factor. Estimate the market return robustly (cross-sectional
median, not mean - the mean is contaminated by the very outlier we're hunting),
regress each name on it with an online EWMA beta, and score the residual
against its own robust scale.
"""

from __future__ import annotations

import math
import statistics

from ..stats import EWMA, Latch, RobustScale
from ..types import Alert, Snapshot, make_alert
from .base import Detector


class CrossSectionalResidual(Detector):
    name = "cross_sectional"

    def __init__(
        self,
        enter: float = 5.0,
        exit_: float = 3.0,
        beta_halflife: float = 250.0,
        scale_halflife: float = 150.0,
        warmup: int = 300,
        min_universe: int = 4,
    ) -> None:
        self.enter = enter
        self.exit_ = exit_
        self.beta_halflife = beta_halflife
        self.warmup = warmup
        self.min_universe = min_universe
        self.scale_halflife = scale_halflife
        self._last_price: dict[str, float] = {}
        self._cov: dict[str, EWMA] = {}
        self._var: dict[str, EWMA] = {}
        self._resid_scale: dict[str, RobustScale] = {}
        self._latch: dict[str, Latch] = {}

    def _returns(self, snap: Snapshot) -> dict[str, float]:
        rets: dict[str, float] = {}
        for sym, tick in snap.ticks.items():
            prev = self._last_price.get(sym)
            self._last_price[sym] = tick.price
            if prev is not None and prev > 0 and tick.price > 0:
                rets[sym] = math.log(tick.price / prev)
        return rets

    def update(self, snap: Snapshot) -> list[Alert]:
        rets = self._returns(snap)
        if len(rets) < self.min_universe:
            return []

        mkt = statistics.median(rets.values())
        out: list[Alert] = []

        for sym, r in rets.items():
            if sym not in self._cov:
                self._cov[sym] = EWMA(self.beta_halflife)
                self._var[sym] = EWMA(self.beta_halflife)
                self._resid_scale[sym] = RobustScale(
                    halflife=self.scale_halflife, warmup=self.warmup
                )
                self._latch[sym] = Latch(self.enter, self.exit_)

            cov, var = self._cov[sym].value, self._var[sym].value
            beta = 1.0
            if cov is not None and var is not None and var > 1e-18:
                beta = cov / var
                beta = max(-3.0, min(3.0, beta))  # winsorise a noisy estimate

            resid = r - beta * mkt
            score = self._resid_scale[sym].score(resid)

            # Update beta *after* scoring: point-in-time discipline.
            self._cov[sym].update(r * mkt)
            self._var[sym].update(mkt * mkt)

            if score is None or not self._latch[sym].push(score):
                continue

            out.append(
                make_alert(
                    snap,
                    sym,
                    self.name,
                    "idiosyncratic_move",
                    score,
                    self.enter,
                    f"{sym} residual {resid * 1e4:+.0f}bp after stripping "
                    f"beta {beta:.2f} x market {mkt * 1e4:+.0f}bp "
                    f"= {score:.1f} sigma idiosyncratic",
                    beta=round(beta, 3),
                    market_return_bp=round(mkt * 1e4, 2),
                    residual_bp=round(resid * 1e4, 2),
                )
            )
        return out

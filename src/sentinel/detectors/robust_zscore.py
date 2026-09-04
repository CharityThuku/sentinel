"""Adaptive, regime-aware threshold breach detection.

The naive version of this problem ("alert if the move exceeds 3%") fails in
both directions: it screams all day in March 2020 and stays silent when a
normally-dead 30-year municipal bond moves 40bp. The threshold has to be
denominated in the instrument's *current* volatility regime, and the scale
estimate has to survive the very outliers it is trying to detect.
"""

from __future__ import annotations

import math

from ..stats import EWMVariance, Latch, RobustScale
from ..types import Alert, Snapshot, make_alert
from .base import Detector


class RobustZScore(Detector):
    name = "robust_z"

    def __init__(
        self,
        enter: float = 5.0,
        exit_: float = 3.0,
        halflife: float = 120.0,
        warmup: int = 300,
        vol_halflife: float = 60.0,
    ) -> None:
        self.enter = enter
        self.exit_ = exit_
        self.halflife = halflife
        self.warmup = warmup
        self.vol_halflife = vol_halflife
        self._scales: dict[str, RobustScale] = {}
        self._vol: dict[str, EWMVariance] = {}
        self._latches: dict[str, Latch] = {}
        self._last_price: dict[str, float] = {}

    def _state(self, sym: str) -> tuple[RobustScale, EWMVariance, Latch]:
        if sym not in self._scales:
            self._scales[sym] = RobustScale(halflife=self.halflife, warmup=self.warmup)
            self._vol[sym] = EWMVariance(halflife=self.vol_halflife, warmup=self.warmup)
            self._latches[sym] = Latch(self.enter, self.exit_)
        return self._scales[sym], self._vol[sym], self._latches[sym]

    def update(self, snap: Snapshot) -> list[Alert]:
        out: list[Alert] = []
        for sym, tick in snap.ticks.items():
            prev = self._last_price.get(sym)
            self._last_price[sym] = tick.price
            if prev is None or prev <= 0 or tick.price <= 0:
                continue
            ret = math.log(tick.price / prev)

            scale, vol, latch = self._state(sym)
            score = scale.score(ret)
            vol.update(ret)
            if score is None:
                continue

            fired = latch.push(score)
            if not fired:
                continue

            ann_vol = vol.std * math.sqrt(252 * 390)  # 1-min bars -> annualised
            out.append(
                make_alert(
                    snap,
                    sym,
                    self.name,
                    "threshold_breach",
                    score,
                    self.enter,
                    f"{sym} moved {ret * 1e4:+.0f}bp = {score:.1f} robust sigma "
                    f"(regime vol {ann_vol:.0%} annualised)",
                    return_bp=round(ret * 1e4, 2),
                    robust_scale_bp=round((scale.scale or 0) * 1e4, 3),
                    regime_vol_annualised=round(ann_vol, 4),
                )
            )
        return out

"""Feed integrity: the anomalies that actually cost money.

Statistical detectors assume the data is real. Most production incidents are
not exotic market events - they are a vendor feed that froze at 09:47 and kept
serving the same print, a quote that crossed, a corporate action applied twice.
A model fed a frozen price reports zero volatility and zero risk, which is
precisely when it is most dangerous.

These checks are deterministic, cheap, and they fire before the statistical
layer has anything to say.
"""

from __future__ import annotations

from ..stats import EWMA
from ..types import Alert, Snapshot, make_alert
from .base import Detector


class FeedIntegrity(Detector):
    name = "integrity"

    def __init__(
        self,
        stale_ticks: int = 20,
        volume_spike_mult: float = 12.0,
        volume_halflife: float = 200.0,
        volume_warmup: int = 100,
        max_spread_bp: float = 250.0,
    ) -> None:
        self.stale_ticks = stale_ticks
        self.volume_spike_mult = volume_spike_mult
        self.max_spread_bp = max_spread_bp
        self.volume_halflife = volume_halflife
        self.volume_warmup = volume_warmup
        self._last_price: dict[str, float] = {}
        self._repeat: dict[str, int] = {}
        self._stale_fired: dict[str, bool] = {}
        self._vol_ewma: dict[str, EWMA] = {}
        self._last_ts: float | None = None

    def update(self, snap: Snapshot) -> list[Alert]:
        out: list[Alert] = []

        if self._last_ts is not None and snap.ts < self._last_ts:
            out.append(
                make_alert(
                    snap,
                    "*",
                    self.name,
                    "timestamp_regression",
                    1.0,
                    1.0,
                    f"snapshot timestamp went backwards "
                    f"({snap.ts:.0f} < {self._last_ts:.0f}) - out-of-order feed",
                )
            )
        self._last_ts = snap.ts

        for sym, tick in snap.ticks.items():
            if tick.price <= 0 or tick.price != tick.price:
                out.append(
                    make_alert(
                        snap, sym, self.name, "invalid_price", 1.0, 1.0,
                        f"{sym} non-positive or NaN price ({tick.price})",
                    )
                )
                continue

            if tick.bid is not None and tick.ask is not None:
                if tick.bid > tick.ask:
                    bp = (tick.bid - tick.ask) / tick.price * 1e4
                    out.append(
                        make_alert(
                            snap, sym, self.name, "crossed_quote", 1.0, 1.0,
                            f"{sym} crossed market: bid {tick.bid:.4f} > "
                            f"ask {tick.ask:.4f} ({bp:.0f}bp inverted)",
                            bid=tick.bid, ask=tick.ask,
                        )
                    )
                else:
                    spread_bp = (tick.ask - tick.bid) / tick.price * 1e4
                    if spread_bp > self.max_spread_bp:
                        out.append(
                            make_alert(
                                snap, sym, self.name, "wide_spread",
                                spread_bp, self.max_spread_bp,
                                f"{sym} spread {spread_bp:.0f}bp exceeds "
                                f"{self.max_spread_bp:.0f}bp - liquidity gap",
                                spread_bp=round(spread_bp, 1),
                            )
                        )

            prev = self._last_price.get(sym)
            if prev is not None and prev == tick.price:
                self._repeat[sym] = self._repeat.get(sym, 0) + 1
                n = self._repeat[sym]
                if n >= self.stale_ticks and not self._stale_fired.get(sym, False):
                    self._stale_fired[sym] = True
                    out.append(
                        make_alert(
                            snap, sym, self.name, "stale_price",
                            float(n), float(self.stale_ticks),
                            f"{sym} price frozen at {tick.price:.4f} for {n} "
                            f"consecutive ticks with live volume - suspect feed",
                            repeat_count=n,
                        )
                    )
            else:
                self._repeat[sym] = 0
                self._stale_fired[sym] = False
            self._last_price[sym] = tick.price

            ew = self._vol_ewma.setdefault(
                sym, EWMA(self.volume_halflife, warmup=self.volume_warmup)
            )
            baseline = ew.value
            if ew.ready and baseline and tick.volume > baseline * self.volume_spike_mult:
                out.append(
                    make_alert(
                        snap, sym, self.name, "volume_spike",
                        tick.volume / baseline, self.volume_spike_mult,
                        f"{sym} volume {tick.volume:,.0f} is "
                        f"{tick.volume / baseline:.0f}x its baseline",
                        baseline_volume=round(baseline, 1),
                    )
                )
            ew.update(tick.volume)

        return out

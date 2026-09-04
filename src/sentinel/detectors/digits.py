"""Leading-digit forensics on trade notionals.

The genuinely off-piste layer. It says nothing about whether a market moved;
it says whether the *numbers* are still the kind of numbers this feed has
always produced. A stub service plumbed into production, a broker sending
round-lot placeholders, a fat-finger pattern of suspiciously clean notionals,
a vendor silently switching units - none of these move a price enough to trip
a sigma threshold, and all of them change the digit distribution immediately.

An honest note on Benford's law, which is what everyone reaches for here:
it only holds for quantities spanning several orders of magnitude. A single
liquid name's notionals span maybe one and a half, so a raw Benford test on
one symbol fires constantly on perfectly healthy data - measured chi2 of ~200
against a p=0.001 critical value of 26.1. Deploying that would be worse than
deploying nothing.

So the detector tests against the symbol's *own* long-run digit distribution
instead, and reports the Benford divergence alongside as context. The
reference and the test window are kept strictly disjoint - a digit only
enters the reference once it has aged out of the window - which keeps the
test point-in-time correct and stops the anomaly from contaminating its own
null hypothesis.

The trigger level (75) is also empirical rather than tabulated. Consecutive
notionals are strongly autocorrelated, so 400 observations carry far fewer
than 400 independent digits and the statistic is inflated relative to a
textbook chi2 with 8 degrees of freedom. Measured on clean tapes it peaks
near 58; injected tampering scores 95-101. The threshold sits in the gap.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..stats import benford_chi2
from ..types import Alert, Snapshot, make_alert
from .base import Detector


def _leading_digit(v: float) -> int | None:
    v = abs(float(v))
    if v <= 0 or not math.isfinite(v):
        return None
    d = int(str(v).replace(".", "").lstrip("0")[:1] or 0)
    return d if 1 <= d <= 9 else None


class DigitDistribution(Detector):
    name = "digits"

    def __init__(
        self,
        window: int = 400,
        threshold: float = 75.0,
        check_every: int = 50,
        min_reference: int = 1000,
        decay: float = 0.9999,
    ) -> None:
        self.window = window
        self.threshold = threshold
        self.check_every = check_every
        self.min_reference = min_reference
        self.decay = decay
        self._win: dict[str, deque[int]] = {}
        self._ref: dict[str, np.ndarray] = {}
        self._since: dict[str, int] = {}
        self._firing: dict[str, bool] = {}

    def update(self, snap: Snapshot) -> list[Alert]:
        out: list[Alert] = []
        for sym, tick in snap.ticks.items():
            d = _leading_digit(tick.notional)
            if d is None:
                continue

            win = self._win.setdefault(sym, deque(maxlen=self.window))
            ref = self._ref.setdefault(sym, np.zeros(9))
            if len(win) == self.window:
                # the digit aging out of the window joins the reference:
                # window and reference never share an observation
                ref *= self.decay
                ref[win[0] - 1] += 1.0
            win.append(d)

            self._since[sym] = self._since.get(sym, 0) + 1
            if self._since[sym] < self.check_every:
                continue
            if len(win) < self.window or ref.sum() < self.min_reference:
                continue
            self._since[sym] = 0

            probs = (ref + 0.5) / (ref.sum() + 4.5)
            observed = np.bincount(np.array(win) - 1, minlength=9).astype(float)
            expected = probs * len(win)
            chi2 = float(((observed - expected) ** 2 / expected).sum())

            if chi2 >= self.threshold:
                if not self._firing.get(sym, False):
                    self._firing[sym] = True
                    bench, _ = benford_chi2([t for t in win])
                    top = int(np.argmax(observed - expected)) + 1
                    out.append(
                        make_alert(
                            snap, sym, self.name, "digit_distribution",
                            chi2, self.threshold,
                            f"{sym} notional leading digits diverge from this "
                            f"symbol's own history (chi2={chi2:.0f}, df=8, "
                            f"n={len(win)}; excess of leading '{top}') - "
                            f"data provenance issue, not a market event",
                            chi2=round(chi2, 1),
                            benford_chi2=round(bench, 1),
                            sample=len(win),
                        )
                    )
            else:
                self._firing[sym] = False
        return out

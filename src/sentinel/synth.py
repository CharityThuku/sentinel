"""A labelled market, because anomaly detection without labels is opinion.

You cannot tune an anomaly detector on real data: nobody hands you the ground
truth of which of 6,000 ticks were "wrong". So we generate a market with a
realistic factor structure and a stochastic volatility regime, inject a fixed
schedule of *known* faults, and keep the labels. Every threshold in this repo
was then chosen against measured recall and false-alert rate rather than taste.

The generator is seeded and deterministic: the numbers in the README reproduce
exactly.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .types import Snapshot, Tick

SYMBOLS = (
    "ALPH", "BRVO", "CHRL", "DLTA", "ECHO",
    "FXTR", "GOLF", "HTEL", "INDX", "JULT",
)

# kind -> the detector(s) the design expects to own this failure mode
EXPECTED: dict[str, tuple[str, ...]] = {
    "fat_finger": ("robust_z", "cross_sectional"),
    "stale_feed": ("integrity",),
    "directional_drift": ("drift",),
    "idio_shock": ("cross_sectional", "robust_z"),
    "crossed_quote": ("integrity",),
    "decouple": ("correlation", "cross_sectional"),
    "digit_tamper": ("benford",),
    "vol_regime": ("robust_z", "drift"),
    "volume_spike": ("integrity",),
}


@dataclass(frozen=True)
class InjectedEvent:
    symbol: str
    kind: str
    start: int
    end: int

    @property
    def expected(self) -> tuple[str, ...]:
        return EXPECTED[self.kind]

    def covers(self, idx: int, grace: int = 0) -> bool:
        return self.start <= idx <= self.end + grace


# (kind, symbol, start, length). Hand-placed so windows never overlap on a
# symbol and every detector gets at least two independent chances.
SCHEDULE: tuple[tuple[str, str, int, int], ...] = (
    ("fat_finger",       "ALPH", 1200, 1),
    ("stale_feed",       "BRVO", 1500, 40),
    ("directional_drift", "CHRL", 1900, 400),
    ("idio_shock",       "DLTA", 2400, 1),
    ("crossed_quote",    "ECHO", 2700, 8),
    ("decouple",         "FXTR", 2900, 200),
    ("digit_tamper",     "GOLF", 3300, 400),
    ("vol_regime",       "HTEL", 3900, 250),
    ("volume_spike",     "INDX", 4300, 3),
    ("fat_finger",       "JULT", 4600, 1),
    ("stale_feed",       "ALPH", 4800, 30),
    ("directional_drift", "DLTA", 5100, 400),
    ("idio_shock",       "BRVO", 5500, 1),
    ("decouple",         "ECHO", 5600, 200),
)


def generate(
    n_ticks: int = 6000,
    seed: int = 20260904,
    schedule: tuple[tuple[str, str, int, int], ...] = SCHEDULE,
) -> tuple[list[Snapshot], list[InjectedEvent]]:
    rng = np.random.default_rng(seed)
    syms = list(SYMBOLS)
    n = len(syms)
    col = {s: i for i, s in enumerate(syms)}

    beta = rng.uniform(0.6, 1.4, size=n)
    idio_sigma = rng.uniform(0.0004, 0.0010, size=n)
    start_px = rng.uniform(18.0, 240.0, size=n)
    base_vol = rng.uniform(8_000, 40_000, size=n)
    spread_bp = rng.uniform(3.0, 18.0, size=n)

    # Two-state market volatility regime (calm / stressed).
    regime = np.zeros(n_ticks, dtype=int)
    p_switch = np.array([0.0015, 0.006])
    for t in range(1, n_ticks):
        regime[t] = (
            1 - regime[t - 1] if rng.random() < p_switch[regime[t - 1]] else regime[t - 1]
        )
    mkt_sigma = np.where(regime == 1, 0.0016, 0.00055)
    factor = rng.normal(0.0, 1.0, size=n_ticks) * mkt_sigma

    idio = rng.normal(0.0, 1.0, size=(n_ticks, n)) * idio_sigma
    rets = factor[:, None] * beta[None, :] + idio

    # per-symbol total volatility of the *unperturbed* tape, so injected
    # drifts are specified in units the detector actually sees
    total_sigma = rets.std(axis=0)

    events: list[InjectedEvent] = []
    price_ops: list[tuple[str, str, int, int]] = []

    for kind, sym, start, length in schedule:
        j = col[sym]
        end = start + length - 1
        if end >= n_ticks:
            continue  # short tape: drop events that would not fit
        if kind == "directional_drift":
            rets[start : end + 1, j] += 0.25 * total_sigma[j]
        elif kind == "idio_shock":
            rets[start, j] += 0.045
        elif kind == "decouple":
            # a hedged pair coming apart: both legs drop their factor
            # loading and their idiosyncratic vol widens. A single name
            # losing its beta is a weak multivariate signal; a *pair*
            # decorrelating is the scenario this detector exists for.
            partner = (j + 1) % n
            for leg in (j, partner):
                rets[start : end + 1, leg] = idio[start : end + 1, leg] * 3.0
        elif kind == "vol_regime":
            rets[start : end + 1, j] = (
                factor[start : end + 1] * beta[j] + idio[start : end + 1, j] * 4.0
            )
        else:
            price_ops.append((kind, sym, start, end))
        # a stale feed hides the truth until it recovers: the catch-up print
        # belongs to the event, so extend the window by two ticks
        events.append(
            InjectedEvent(sym, kind, start, end + (2 if kind == "stale_feed" else 0))
        )

    prices = start_px[None, :] * np.exp(np.cumsum(rets, axis=0))
    volumes = base_vol[None, :] * np.exp(rng.normal(0.0, 0.45, size=(n_ticks, n)))
    bids = prices * (1 - spread_bp[None, :] / 2e4)
    asks = prices * (1 + spread_bp[None, :] / 2e4)

    for kind, sym, start, end in price_ops:
        j = col[sym]
        if kind == "fat_finger":
            prices[start, j] *= 1.055
            bids[start, j] = prices[start, j] * (1 - spread_bp[j] / 2e4)
            asks[start, j] = prices[start, j] * (1 + spread_bp[j] / 2e4)
        elif kind == "stale_feed":
            frozen = prices[start - 1, j]
            prices[start : end + 1, j] = frozen
            bids[start : end + 1, j] = frozen * (1 - spread_bp[j] / 2e4)
            asks[start : end + 1, j] = frozen * (1 + spread_bp[j] / 2e4)
        elif kind == "crossed_quote":
            b = bids[start : end + 1, j].copy()
            bids[start : end + 1, j] = asks[start : end + 1, j]
            asks[start : end + 1, j] = b
        elif kind == "volume_spike":
            volumes[start : end + 1, j] *= 25.0
        elif kind == "digit_tamper":
            # a stub feed emitting only round notionals: 1e5, 5e5, 1e6 ...
            mant = rng.choice([1.0, 5.0], size=end - start + 1)
            mag = rng.choice([1e5, 1e6], size=end - start + 1)
            volumes[start : end + 1, j] = mant * mag / prices[start : end + 1, j]

    snapshots: list[Snapshot] = []
    t0 = 1_757_000_000.0
    for t in range(n_ticks):
        ticks = {
            s: Tick(
                symbol=s,
                price=float(prices[t, i]),
                volume=float(volumes[t, i]),
                bid=float(bids[t, i]),
                ask=float(asks[t, i]),
            )
            for i, s in enumerate(syms)
        }
        snapshots.append(Snapshot(idx=t, ts=t0 + 60.0 * t, ticks=ticks))

    return snapshots, sorted(events, key=lambda e: e.start)


def stream(
    snapshots: list[Snapshot], upto: int | None = None
) -> Iterator[Snapshot]:
    yield from snapshots[: upto if upto is not None else len(snapshots)]

"""Behavioural tests: each detector catches its fault, and - just as important
- stays quiet on a clean tape and on the faults that belong to someone else."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.detectors import (
    CorrelationBreakdown,
    CrossSectionalResidual,
    DigitDistribution,
    DriftDetector,
    FeedIntegrity,
    RobustZScore,
)
from sentinel.synth import SCHEDULE, generate
from sentinel.types import Snapshot, Tick

SYMS = [f"S{i}" for i in range(8)]


def clean_tape(n=3000, seed=0, sigma=0.0008):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.0007, size=n)
    idio = rng.normal(0, sigma, size=(n, len(SYMS)))
    rets = f[:, None] + idio
    px = 100.0 * np.exp(np.cumsum(rets, axis=0))
    vol = 10000 * np.exp(rng.normal(0, 0.4, size=(n, len(SYMS))))
    out = []
    for t in range(n):
        out.append(
            Snapshot(
                idx=t,
                ts=1e9 + 60 * t,
                ticks={
                    s: Tick(s, float(px[t, i]), float(vol[t, i]),
                            float(px[t, i]) * 0.9995, float(px[t, i]) * 1.0005)
                    for i, s in enumerate(SYMS)
                },
            )
        )
    return out


def run(det, snaps):
    return [a for s in snaps for a in det.update(s)]


def test_robust_z_catches_price_spike():
    snaps = clean_tape()
    t = 2000
    spiked = list(snaps)
    tick = spiked[t].ticks["S0"]
    ticks = dict(spiked[t].ticks)
    ticks["S0"] = Tick("S0", tick.price * 1.05, tick.volume, tick.bid, tick.ask)
    spiked[t] = Snapshot(t, spiked[t].ts, ticks)

    alerts = run(RobustZScore(), spiked)
    assert any(a.idx == t and a.symbol == "S0" for a in alerts)


def test_robust_z_quiet_on_clean_tape():
    alerts = run(RobustZScore(), clean_tape(seed=9))
    assert len(alerts) <= 3, [a.message for a in alerts]


def test_drift_catches_what_threshold_misses():
    """A 0.25-sigma-per-tick drift: no single tick is unusual."""
    snaps = clean_tape(n=4000, seed=2)
    sigma = 0.0008
    mutated, carry = [], 0.0
    for s in snaps:
        ticks = dict(s.ticks)
        if 2500 <= s.idx < 3200:
            carry += 0.25 * sigma
        t = ticks["S0"]
        ticks["S0"] = Tick("S0", t.price * np.exp(carry), t.volume, t.bid, t.ask)
        mutated.append(Snapshot(s.idx, s.ts, ticks))

    z_alerts = run(RobustZScore(), mutated)
    d_alerts = run(DriftDetector(), mutated)
    assert not any(a.symbol == "S0" and 2500 <= a.idx < 3300 for a in z_alerts)
    assert any(a.symbol == "S0" and 2500 <= a.idx < 3300 for a in d_alerts)


def test_cross_sectional_ignores_market_moves_but_flags_idiosyncratic():
    snaps = clean_tape(n=3000, seed=4)

    # (a) everything moves 3% together -> not an anomaly
    together = list(snaps)
    t = 2000
    together[t] = Snapshot(
        t, snaps[t].ts,
        {s: Tick(s, k.price * 1.03, k.volume, k.bid, k.ask)
         for s, k in snaps[t].ticks.items()},
    )
    alerts = run(CrossSectionalResidual(), together)
    assert not any(a.idx == t for a in alerts)

    # (b) one name moves 3% alone -> anomaly
    alone = list(snaps)
    ticks = dict(snaps[t].ticks)
    k = ticks["S3"]
    ticks["S3"] = Tick("S3", k.price * 1.03, k.volume, k.bid, k.ask)
    alone[t] = Snapshot(t, snaps[t].ts, ticks)
    alerts = run(CrossSectionalResidual(), alone)
    assert any(a.idx == t and a.symbol == "S3" for a in alerts)


def test_integrity_catches_stale_price():
    snaps = clean_tape(n=1200, seed=6)
    mutated = []
    frozen = None
    for s in snaps:
        ticks = dict(s.ticks)
        if 800 <= s.idx < 860:
            frozen = frozen or ticks["S1"].price
            t = ticks["S1"]
            ticks["S1"] = Tick("S1", frozen, t.volume, t.bid, t.ask)
        mutated.append(Snapshot(s.idx, s.ts, ticks))
    alerts = run(FeedIntegrity(), mutated)
    stale = [a for a in alerts if a.kind == "stale_price"]
    assert len(stale) == 1, "must fire once, not once per frozen tick"
    assert stale[0].symbol == "S1"


def test_integrity_catches_crossed_quote():
    snaps = clean_tape(n=300)
    ticks = dict(snaps[100].ticks)
    t = ticks["S2"]
    ticks["S2"] = Tick("S2", t.price, t.volume, bid=t.ask, ask=t.bid)
    snaps[100] = Snapshot(100, snaps[100].ts, ticks)
    alerts = run(FeedIntegrity(), snaps)
    assert any(a.kind == "crossed_quote" and a.symbol == "S2" for a in alerts)


def test_digits_flags_tampered_notionals_only():
    rng = np.random.default_rng(8)
    snaps = clean_tape(n=2000, seed=8)
    mutated = []
    for s in snaps:
        ticks = dict(s.ticks)
        if s.idx >= 1000:
            t = ticks["S4"]
            notional = float(rng.choice([1.0, 5.0]) * rng.choice([1e5, 1e6]))
            ticks["S4"] = Tick("S4", t.price, notional / t.price, t.bid, t.ask)
        mutated.append(Snapshot(s.idx, s.ts, ticks))
    alerts = run(DigitDistribution(), mutated)
    assert {a.symbol for a in alerts} == {"S4"}


def test_correlation_detects_decoupling():
    snaps, events = generate(n_ticks=6000, seed=20260904)
    alerts = run(CorrelationBreakdown(), snaps)
    decouples = [e for e in events if e.kind == "decouple"]
    assert decouples
    assert any(
        any(e.start <= a.idx <= e.end + 60 for a in alerts) for e in decouples
    )


@pytest.mark.parametrize("kind", sorted({k for k, *_ in SCHEDULE}))
def test_every_injected_kind_is_covered_by_some_detector(kind):
    from sentinel.detectors import default_stack
    from sentinel.engine import Engine
    from sentinel.evaluate import evaluate

    snaps, events = generate(n_ticks=6000, seed=20260904)
    alerts = list(Engine(default_stack()).run(snaps))
    report = evaluate(alerts, events, len(snaps))
    for r in report.results:
        if r.event.kind == kind:
            assert r.detected, f"{kind} on {r.event.symbol} @ {r.event.start} missed"

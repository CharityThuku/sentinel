from __future__ import annotations

from sentinel.detectors.base import Detector
from sentinel.engine import AlertPolicy, Engine
from sentinel.types import MARKET_WIDE, Severity, Snapshot, Tick, make_alert


class Always(Detector):
    """A detector that fires on every symbol every tick - worst case for the
    governance layer."""

    name = "always"

    def __init__(self, symbols, score=1.0):
        self.symbols = symbols
        self.score = score

    def update(self, snap):
        return [
            make_alert(snap, s, self.name, "noise", self.score, 1.0, "boom")
            for s in self.symbols
        ]


def snaps(n, symbols):
    return [
        Snapshot(i, 1e9 + i, {s: Tick(s, 100.0, 1.0) for s in symbols})
        for i in range(n)
    ]


def test_cooldown_suppresses_repeats():
    eng = Engine([Always(["A"])], AlertPolicy(cooldown=10, storm_symbols=99))
    alerts = list(eng.run(snaps(100, ["A"])))
    assert len(alerts) == 10
    assert eng.stats.suppressed_cooldown == 90


def test_storm_collapses_to_one_market_wide_alert():
    syms = ["A", "B", "C", "D", "E"]
    eng = Engine([Always(syms)], AlertPolicy(cooldown=0, storm_symbols=4))
    out = list(eng.run(snaps(1, syms)))
    assert len(out) == 1
    assert out[0].symbol == MARKET_WIDE
    assert out[0].kind.endswith("_storm")
    assert out[0].severity is Severity.CRITICAL
    assert out[0].context["symbols"] == syms


def test_severity_floor_filters():
    eng = Engine(
        [Always(["A"], score=1.0)],
        AlertPolicy(cooldown=0, storm_symbols=99, min_severity=Severity.CRITICAL),
    )
    assert list(eng.run(snaps(50, ["A"]))) == []
    assert eng.stats.suppressed_severity == 50


def test_severity_scales_with_threshold_ratio():
    assert Severity.from_ratio(5.1, 5.0) is Severity.INFO
    assert Severity.from_ratio(6.5, 5.0) is Severity.WARN
    assert Severity.from_ratio(11.0, 5.0) is Severity.CRITICAL
    # a Mahalanobis distance and a z-score at the same relative level agree
    assert Severity.from_ratio(80.0, 40.0) is Severity.CRITICAL


def test_stats_report_noise_reduction():
    eng = Engine([Always(["A"])], AlertPolicy(cooldown=25, storm_symbols=99))
    list(eng.run(snaps(1000, ["A"])))
    d = eng.stats.as_dict()
    assert d["raw_alerts"] == 1000
    assert d["emitted"] == 40
    assert d["noise_reduction"] == 0.96

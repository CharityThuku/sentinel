"""Orchestration and alert governance.

The hard part of an anomaly system is not detection, it is the second-order
problem: a detector that is right 200 times during one event has produced 199
pieces of noise, and after two weeks of that nobody reads the channel. So the
engine owns suppression policy explicitly rather than leaving it implicit in
each detector's thresholds.

Three mechanisms:
  * per-(detector, symbol, kind) cooldown - repeat suppression
  * storm collapse - if one detector lights up N+ symbols in a single
    snapshot, that is a market event or a feed outage, not N anomalies; it
    is escalated into one market-wide alert
  * severity floor - route INFO to a log, WARN to a dashboard, CRITICAL to
    a human
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .detectors.base import Detector
from .types import MARKET_WIDE, Alert, Severity, Snapshot


@dataclass(frozen=True)
class AlertPolicy:
    cooldown: int = 30
    storm_symbols: int = 4
    min_severity: Severity = Severity.INFO


@dataclass
class EngineStats:
    snapshots: int = 0
    raw_alerts: int = 0
    suppressed_cooldown: int = 0
    suppressed_storm: int = 0
    suppressed_severity: int = 0
    emitted: int = 0
    by_detector: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "snapshots": self.snapshots,
            "raw_alerts": self.raw_alerts,
            "emitted": self.emitted,
            "suppressed_cooldown": self.suppressed_cooldown,
            "suppressed_storm": self.suppressed_storm,
            "suppressed_severity": self.suppressed_severity,
            "noise_reduction": (
                round(1 - self.emitted / self.raw_alerts, 3) if self.raw_alerts else 0.0
            ),
            "alerts_per_1k_snapshots": (
                round(1000 * self.emitted / self.snapshots, 2) if self.snapshots else 0.0
            ),
            "by_detector": dict(sorted(self.by_detector.items())),
        }


class Engine:
    def __init__(
        self,
        detectors: list[Detector],
        policy: AlertPolicy | None = None,
    ) -> None:
        self.detectors = detectors
        self.policy = policy or AlertPolicy()
        self.stats = EngineStats()
        self._last_fired: dict[tuple[str, str, str], int] = {}

    def run(self, stream: Iterable[Snapshot]) -> Iterator[Alert]:
        for snap in stream:
            yield from self.step(snap)

    def step(self, snap: Snapshot) -> list[Alert]:
        self.stats.snapshots += 1
        emitted: list[Alert] = []

        for det in self.detectors:
            raw = det.update(snap)
            if not raw:
                continue
            self.stats.raw_alerts += len(raw)
            for alert in self._govern(det.name, raw, snap):
                self.stats.emitted += 1
                self.stats.by_detector[alert.detector] = (
                    self.stats.by_detector.get(alert.detector, 0) + 1
                )
                emitted.append(alert)
        return emitted

    def _govern(
        self, detector: str, raw: list[Alert], snap: Snapshot
    ) -> list[Alert]:
        symbols = {a.symbol for a in raw if a.symbol != MARKET_WIDE}
        if len(symbols) >= self.policy.storm_symbols:
            self.stats.suppressed_storm += len(raw) - 1
            worst = max(raw, key=lambda a: a.score / max(a.threshold, 1e-9))
            collapsed = Alert(
                idx=snap.idx,
                ts=snap.ts,
                symbol=MARKET_WIDE,
                detector=detector,
                kind=f"{worst.kind}_storm",
                score=worst.score,
                threshold=worst.threshold,
                severity=Severity.CRITICAL,
                message=(
                    f"{len(symbols)} symbols breached simultaneously "
                    f"({', '.join(sorted(symbols)[:6])}"
                    f"{'...' if len(symbols) > 6 else ''}) - "
                    f"market-wide event or feed outage, not {len(symbols)} anomalies"
                ),
                context={"symbols": sorted(symbols), "example": worst.message},
            )
            raw = [collapsed]

        out: list[Alert] = []
        for alert in raw:
            if alert.severity < self.policy.min_severity:
                self.stats.suppressed_severity += 1
                continue
            key = (alert.detector, alert.symbol, alert.kind)
            last = self._last_fired.get(key)
            if last is not None and alert.idx - last < self.policy.cooldown:
                self.stats.suppressed_cooldown += 1
                continue
            self._last_fired[key] = alert.idx
            out.append(alert)
        return out

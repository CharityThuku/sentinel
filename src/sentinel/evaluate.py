"""Scoring the detectors against known injected faults.

Three numbers matter, in this order:

  1. recall - did we catch the fault at all
  2. detection latency - how many ticks after onset (a drift caught 400 ticks
     late is a post-mortem, not an alert)
  3. false alerts per 1,000 snapshots - the number that decides whether
     anyone still reads the channel in month two

Precision is reported but is the weakest of the four: an "unmatched" alert on
synthetic data is often a real tail event the generator produced honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .synth import InjectedEvent
from .types import MARKET_WIDE, Alert


@dataclass
class EventResult:
    event: InjectedEvent
    detected_by: tuple[str, ...]
    latency: int | None

    @property
    def detected(self) -> bool:
        return self.latency is not None

    @property
    def by_expected_detector(self) -> bool:
        return any(d in self.event.expected for d in self.detected_by)


@dataclass
class Report:
    results: list[EventResult]
    n_alerts: int
    n_matched: int
    n_unmatched: int
    n_snapshots: int

    @property
    def recall(self) -> float:
        return sum(r.detected for r in self.results) / max(len(self.results), 1)

    @property
    def precision(self) -> float:
        return self.n_matched / max(self.n_alerts, 1)

    @property
    def false_alerts_per_1k(self) -> float:
        return 1000.0 * self.n_unmatched / max(self.n_snapshots, 1)

    @property
    def median_latency(self) -> float:
        lats = sorted(r.latency for r in self.results if r.latency is not None)
        if not lats:
            return float("nan")
        mid = len(lats) // 2
        return float(lats[mid] if len(lats) % 2 else (lats[mid - 1] + lats[mid]) / 2)


def evaluate(
    alerts: list[Alert],
    events: list[InjectedEvent],
    n_snapshots: int,
    grace: int = 60,
) -> Report:
    """Match alerts to events. An alert matches if it lands inside the event
    window (plus grace) and either names the symbol or is market-wide."""
    results: list[EventResult] = []
    matched_alerts: set[int] = set()

    for ev in events:
        hits: list[tuple[int, str]] = []
        for i, a in enumerate(alerts):
            if not ev.covers(a.idx, grace):
                continue
            if a.symbol != ev.symbol and a.symbol != MARKET_WIDE:
                continue
            hits.append((a.idx, a.detector))
            matched_alerts.add(i)
        if hits:
            first = min(h[0] for h in hits)
            results.append(
                EventResult(ev, tuple(sorted({h[1] for h in hits})), first - ev.start)
            )
        else:
            results.append(EventResult(ev, (), None))

    return Report(
        results=results,
        n_alerts=len(alerts),
        n_matched=len(matched_alerts),
        n_unmatched=len(alerts) - len(matched_alerts),
        n_snapshots=n_snapshots,
    )


def format_report(report: Report) -> str:
    lines = []
    head = f"{'kind':<18}{'symbol':<8}{'start':>7}{'lat':>6}  detectors"
    lines.append(head)
    lines.append("-" * len(head))
    for r in sorted(report.results, key=lambda x: (x.event.kind, x.event.start)):
        lat = "MISS" if r.latency is None else str(r.latency)
        dets = ", ".join(r.detected_by) or "-"
        flag = "" if r.by_expected_detector else "  <- unexpected owner"
        lines.append(
            f"{r.event.kind:<18}{r.event.symbol:<8}{r.event.start:>7}{lat:>6}  {dets}{flag}"
        )
    lines.append("")
    lines.append(
        f"recall                {report.recall:.1%}  "
        f"({sum(r.detected for r in report.results)}/{len(report.results)} injected faults)"
    )
    lines.append(f"median latency        {report.median_latency:.0f} ticks")
    lines.append(f"alerts emitted        {report.n_alerts}")
    lines.append(f"precision             {report.precision:.1%}")
    lines.append(
        f"false alerts / 1k     {report.false_alerts_per_1k:.2f} snapshots"
    )
    return "\n".join(lines)

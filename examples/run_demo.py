"""Minimal embedding example: build a stack, stream a tape, route alerts.

    python examples/run_demo.py
"""

from __future__ import annotations

from sentinel import AlertPolicy, Engine, evaluate, format_report, generate
from sentinel.detectors import (
    CorrelationBreakdown,
    CrossSectionalResidual,
    DriftDetector,
    FeedIntegrity,
    RobustZScore,
)
from sentinel.types import Severity


def route(alert) -> None:
    """Stand-in for the thing that actually pages someone."""
    if alert.severity is Severity.CRITICAL:
        print(f"  PAGE  [{alert.idx}] {alert.symbol} {alert.message}")
    elif alert.severity is Severity.WARN:
        print(f"  DASH  [{alert.idx}] {alert.symbol} {alert.message}")


def main() -> None:
    snapshots, events = generate(n_ticks=6000, seed=20260904)

    # a custom stack: skip the digit forensics and tighten the z threshold
    engine = Engine(
        [
            FeedIntegrity(),
            RobustZScore(enter=4.5, exit_=3.0),
            CrossSectionalResidual(),
            DriftDetector(),
            CorrelationBreakdown(),
        ],
        AlertPolicy(cooldown=30, storm_symbols=4, min_severity=Severity.INFO),
    )

    alerts = []
    for snap in snapshots:
        for alert in engine.step(snap):
            alerts.append(alert)
            route(alert)

    print()
    print(format_report(evaluate(alerts, events, n_snapshots=len(snapshots))))


if __name__ == "__main__":
    main()

"""Core domain types.

Everything downstream is built on a single streaming primitive: a `Snapshot`
is the complete state of the universe at one point in time. Detectors are
handed snapshots in order and may never look forward. That constraint is
enforced by the API shape, not by convention.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

MARKET_WIDE = "*"


@dataclass(frozen=True, slots=True)
class Tick:
    """One symbol's state at one instant."""

    symbol: str
    price: float
    volume: float
    bid: float | None = None
    ask: float | None = None

    @property
    def notional(self) -> float:
        return self.price * self.volume

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The universe at time index `idx`. `ts` is an epoch-seconds float."""

    idx: int
    ts: float
    ticks: Mapping[str, Tick]

    def symbols(self) -> tuple[str, ...]:
        return tuple(self.ticks.keys())


class Severity(IntEnum):
    INFO = 10
    WARN = 20
    CRITICAL = 30

    @classmethod
    def from_ratio(cls, score: float, threshold: float) -> Severity:
        """Severity is a function of how far past the threshold we are.

        Deliberately not a function of the raw score: a Mahalanobis distance of
        40 and a z-score of 6 are both 'about 1.3x the trigger level'.
        """
        if threshold <= 0:
            return cls.WARN
        ratio = score / threshold
        if ratio >= 2.0:
            return cls.CRITICAL
        if ratio >= 1.25:
            return cls.WARN
        return cls.INFO


@dataclass(frozen=True, slots=True)
class Alert:
    idx: int
    ts: float
    symbol: str
    detector: str
    kind: str
    score: float
    threshold: float
    severity: Severity
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "symbol": self.symbol,
            "detector": self.detector,
            "kind": self.kind,
            "score": round(self.score, 4),
            "threshold": round(self.threshold, 4),
            "severity": self.severity.name,
            "message": self.message,
        }


def make_alert(
    snap: Snapshot,
    symbol: str,
    detector: str,
    kind: str,
    score: float,
    threshold: float,
    message: str,
    **context: Any,
) -> Alert:
    return Alert(
        idx=snap.idx,
        ts=snap.ts,
        symbol=symbol,
        detector=detector,
        kind=kind,
        score=score,
        threshold=threshold,
        severity=Severity.from_ratio(score, threshold),
        message=message,
        context=context,
    )

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Alert, Snapshot


class Detector(ABC):
    """A detector consumes snapshots in order and emits zero or more alerts.

    Contract (enforced by tests/test_pit.py):
      1. `update` may only use the current snapshot and its own internal state.
      2. Alerts emitted at index i must never change if data after i changes.
      3. Memory is O(1) in the length of the stream.
    """

    name: str = "detector"

    @abstractmethod
    def update(self, snap: Snapshot) -> list[Alert]:  # pragma: no cover - abstract
        ...

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"

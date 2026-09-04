"""CSV ingestion, so the same stack runs on a real tape.

Expected columns: ts, symbol, price[, volume, bid, ask]
Rows are grouped into snapshots by `ts`, in file order. Any tape that can be
flattened to this shape - equity bars, bond marks, an internal position file -
runs through the identical detector stack.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from .types import Snapshot, Tick


def _f(row: dict[str, str], key: str) -> float | None:
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_csv(path: str | Path) -> Iterator[Snapshot]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        idx = 0
        current_ts: str | None = None
        bucket: dict[str, Tick] = {}
        for row in reader:
            ts = row["ts"]
            if current_ts is not None and ts != current_ts:
                yield Snapshot(idx=idx, ts=float(current_ts), ticks=bucket)
                idx += 1
                bucket = {}
            current_ts = ts
            price = _f(row, "price")
            if price is None:
                continue
            sym = row["symbol"]
            bucket[sym] = Tick(
                symbol=sym,
                price=price,
                volume=_f(row, "volume") or 0.0,
                bid=_f(row, "bid"),
                ask=_f(row, "ask"),
            )
        if bucket and current_ts is not None:
            yield Snapshot(idx=idx, ts=float(current_ts), ticks=bucket)


def write_csv(snapshots: list[Snapshot], path: str | Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts", "symbol", "price", "volume", "bid", "ask"])
        for snap in snapshots:
            for t in snap.ticks.values():
                w.writerow(
                    [
                        f"{snap.ts:.0f}", t.symbol, f"{t.price:.6f}",
                        f"{t.volume:.2f}",
                        "" if t.bid is None else f"{t.bid:.6f}",
                        "" if t.ask is None else f"{t.ask:.6f}",
                    ]
                )

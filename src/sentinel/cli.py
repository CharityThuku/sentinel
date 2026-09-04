from __future__ import annotations

import argparse
import json
import sys

from .detectors import default_stack
from .engine import AlertPolicy, Engine
from .evaluate import evaluate, format_report
from .io import load_csv, write_csv
from .synth import generate
from .types import Severity


def _engine(cooldown: int, min_severity: str) -> Engine:
    return Engine(
        default_stack(),
        AlertPolicy(cooldown=cooldown, min_severity=Severity[min_severity.upper()]),
    )


def cmd_demo(args: argparse.Namespace) -> int:
    snaps, events = generate(n_ticks=args.ticks, seed=args.seed)
    eng = _engine(args.cooldown, args.min_severity)
    alerts = list(eng.run(snaps))
    print(f"{len(snaps)} snapshots x {len(snaps[0].ticks)} symbols "
          f"-> {len(alerts)} alerts\n")
    for a in alerts:
        print(f"[{a.idx:>5}] {a.severity.name:<8} {a.detector:<15} "
              f"{a.symbol:<6} {a.message}")
    print("\nengine stats:")
    print(json.dumps(eng.stats.as_dict(), indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    snaps, events = generate(n_ticks=args.ticks, seed=args.seed)
    eng = _engine(args.cooldown, args.min_severity)
    alerts = list(eng.run(snaps))
    report = evaluate(alerts, events, n_snapshots=len(snaps), grace=args.grace)
    print(format_report(report))
    print("\nengine stats:")
    print(json.dumps(eng.stats.as_dict(), indent=2))
    return 0 if report.recall >= args.min_recall else 1


def cmd_run(args: argparse.Namespace) -> int:
    eng = _engine(args.cooldown, args.min_severity)
    n = 0
    for snap in load_csv(args.csv):
        n += 1
        for a in eng.step(snap):
            if args.json:
                print(json.dumps(a.to_row()))
            else:
                print(f"[{a.idx:>6}] {a.severity.name:<8} {a.detector:<15} "
                      f"{a.symbol:<6} {a.message}")
    print(f"\n{n} snapshots processed", file=sys.stderr)
    print(json.dumps(eng.stats.as_dict(), indent=2), file=sys.stderr)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    snaps, _ = generate(n_ticks=args.ticks, seed=args.seed)
    write_csv(snaps, args.out)
    print(f"wrote {len(snaps) * len(snaps[0].ticks)} rows to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("sentinel", description="streaming anomaly detection")
    p.add_argument("--cooldown", type=int, default=30)
    p.add_argument("--min-severity", default="INFO", choices=["info", "warn",
                   "critical", "INFO", "WARN", "CRITICAL"])
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the labelled synthetic tape")
    d.add_argument("--ticks", type=int, default=6000)
    d.add_argument("--seed", type=int, default=20260904)
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("eval", help="score detectors against injected faults")
    e.add_argument("--ticks", type=int, default=6000)
    e.add_argument("--seed", type=int, default=20260904)
    e.add_argument("--grace", type=int, default=60)
    e.add_argument("--min-recall", type=float, default=0.0)
    e.set_defaults(func=cmd_eval)

    r = sub.add_parser("run", help="run against a CSV tape")
    r.add_argument("csv")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_run)

    x = sub.add_parser("export", help="write the synthetic tape to CSV")
    x.add_argument("out")
    x.add_argument("--ticks", type=int, default=6000)
    x.add_argument("--seed", type=int, default=20260904)
    x.set_defaults(func=cmd_export)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

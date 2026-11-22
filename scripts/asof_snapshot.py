"""Build an as-of snapshot (F0) from a synthetic JSON ledger and print or save the report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from early_trace.evidence.events import LedgerContractError
from early_trace.evidence.ledger import load_ledger_json
from early_trace.temporal.availability import TimestampError
from early_trace.temporal.snapshot import make_asof_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True, help="synthetic ledger JSON file")
    parser.add_argument(
        "--cutoff", required=True, help="decision instant, ISO-8601 with explicit UTC offset"
    )
    parser.add_argument("--output", type=Path, help="write the JSON report here instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ledger = load_ledger_json(args.ledger, require_synthetic=True)
        snapshot = make_asof_snapshot(args.cutoff, ledger)
    except (LedgerContractError, TimestampError, json.JSONDecodeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = json.dumps(snapshot.to_dict(), indent=2, sort_keys=True)
    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"snapshot_hash={snapshot.snapshot_hash} written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

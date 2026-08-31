"""Split a CPU manifest into two balanced, nonoverlapping node shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlaopt_experiments.sharding import shard_cpu_records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-a", type=Path, required=True)
    parser.add_argument("--output-b", type=Path, required=True)
    return parser


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records))


def main() -> None:
    args = _parser().parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    first, second = shard_cpu_records(records)
    _write(args.output_a, first)
    _write(args.output_b, second)
    print(f"wrote {len(first)} jobs to {args.output_a}")
    print(f"wrote {len(second)} jobs to {args.output_b}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create bounded real-data ERM production batches and submission waves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlaopt_experiments.suites.real_erm.production import plan_real_production


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/real_erm.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-jobs-per-batch", type=int, default=10)
    parser.add_argument("--max-batches-per-node-per-wave", type=int, default=6)
    args = parser.parse_args()
    plan = plan_real_production(
        args.config,
        args.output,
        max_jobs_per_batch=args.max_jobs_per_batch,
        max_batches_per_node_per_wave=args.max_batches_per_node_per_wave,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

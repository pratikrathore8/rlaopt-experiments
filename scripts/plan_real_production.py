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
    parser.add_argument("--max-jobs-per-batch", type=int, default=7)
    parser.add_argument("--max-batches-per-node-per-wave", type=int, default=6)
    parser.add_argument("--gpu-concurrency", type=int, default=1)
    parser.add_argument("--gpu-cpu-threads", type=int, default=64)
    args = parser.parse_args()
    plan = plan_real_production(
        args.config,
        args.output,
        max_jobs_per_batch=args.max_jobs_per_batch,
        max_batches_per_node_per_wave=args.max_batches_per_node_per_wave,
        gpu_concurrency=args.gpu_concurrency,
        gpu_cpu_threads=args.gpu_cpu_threads,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

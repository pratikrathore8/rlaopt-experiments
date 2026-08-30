"""Optional W&B logging with local JSON as the source of truth."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from rlaopt_experiments.records import TrialRecord


@contextmanager
def wandb_run(record: TrialRecord, output_dir: Path) -> Iterator[object | None]:
    try:
        import wandb
    except ImportError:
        yield None
        return
    os.environ.setdefault("WANDB_MODE", "offline")
    run = wandb.init(project="rlaopt-synthetic-ridge", id=record.run_id,
                     dir=str(output_dir), config=asdict(record), resume="allow")
    try:
        yield run
    finally:
        run.finish()


def log_record(run: object | None, record: TrialRecord, trace: list[dict[str, float]]) -> None:
    if run is None:
        return
    for point in trace:
        run.log({f"convergence/{key}": value for key, value in point.items()})
    run.log({
        "summary/runtime_seconds": record.runtime_seconds,
        "summary/relative_kkt": record.relative_kkt,
        "summary/relative_solution_error": record.relative_solution_error,
        "summary/success": int(record.success),
    })

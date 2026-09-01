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
    config = asdict(record)
    config.pop("trace")
    try:
        run = wandb.init(
            project="rlaopt-experiments",
            group=record.suite,
            id=record.run_id,
            dir=str(output_dir),
            config=config,
            resume="allow",
        )
    except Exception:
        yield None
        return
    try:
        yield run
    finally:
        try:
            run.finish()
        except Exception:
            pass


def log_record(run: object | None, record: TrialRecord) -> None:
    if run is None:
        return
    try:
        for point in record.trace:
            run.log({f"convergence/{key}": value for key, value in point.items()})
        summary = {
            f"summary/{key}": value for key, value in record.metrics.items() if value is not None
        }
        summary |= {f"summary/{key}": value for key, value in record.timings.items()}
        summary["summary/success"] = int(record.success)
        run.log(summary)
    except Exception:
        # The atomic JSON record is authoritative; telemetry must not stop a job.
        return

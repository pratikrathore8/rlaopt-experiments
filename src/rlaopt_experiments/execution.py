"""Execute indexed jobs from suite-aware JSONL manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rlaopt_experiments.records import TrialRecord
from rlaopt_experiments.suites.synthetic_erm.config import load_synthetic_erm_config
from rlaopt_experiments.suites.synthetic_erm.execution import (
    run_bounded_elastic_net_job,
    run_multinomial_job,
    run_vanilla_elastic_net_job,
)


def read_manifest_job(path: Path, index: int) -> dict[str, Any]:
    """Read one zero-based JSON object without loading the full manifest."""
    if index < 0:
        raise ValueError("manifest index must be nonnegative")
    with path.open() as manifest:
        for current, line in enumerate(manifest):
            if current == index:
                try:
                    job = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at manifest index {index}") from error
                if not isinstance(job, dict):
                    raise ValueError(f"manifest index {index} is not a JSON object")
                return job
    raise IndexError(f"manifest index {index} is out of range")


def run_manifest_job(
    job: dict[str, Any],
    config_path: Path,
    output_dir: Path,
) -> list[TrialRecord]:
    """Dispatch one self-contained manifest job to its suite executor."""
    suite = job.get("suite")
    if suite == "synthetic_erm":
        config = load_synthetic_erm_config(config_path)
        problem_type = job.get("problem_type")
        if problem_type == "multinomial":
            controls = config.multinomial.execution
            runner = run_multinomial_job
        elif problem_type == "vanilla_elastic_net":
            controls = config.elastic_net.vanilla_execution
            runner = run_vanilla_elastic_net_job
        elif problem_type == "bounded_elastic_net":
            controls = config.elastic_net.bounded_execution
            runner = run_bounded_elastic_net_job
        else:
            raise ValueError(f"unsupported synthetic ERM problem type: {problem_type}")
        native_tolerance = controls.native_tolerances.for_solver(
            job.get("backend"),
            job.get("solver"),
        )
        return runner(
            job,
            config,
            native_tolerance=native_tolerance,
            max_iterations=controls.max_iterations,
            batch_size=controls.batch_size,
            output_dir=output_dir,
        )
    raise ValueError(f"manifest execution is not implemented for suite: {suite}")

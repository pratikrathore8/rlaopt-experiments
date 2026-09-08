"""Native-tolerance calibration for synthetic ERM solver adapters."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

from rlaopt_experiments.records import TrialRecord
from rlaopt_experiments.suites.synthetic_erm.config import (
    SolverExecution,
    SyntheticErmConfig,
)
from rlaopt_experiments.suites.synthetic_erm.execution import (
    run_bounded_elastic_net_job,
    run_multinomial_job,
)
from rlaopt_experiments.suites.synthetic_erm.manifest import build_synthetic_erm_manifest

_PROBLEM_TYPES = {
    "multinomial",
    "bounded_elastic_net",
}

ErmRunner = Callable[..., list[TrialRecord]]


def _execution(
    config: SyntheticErmConfig,
    problem_type: str,
) -> tuple[SolverExecution, ErmRunner]:
    if problem_type == "multinomial":
        return config.multinomial.execution, run_multinomial_job
    if problem_type == "bounded_elastic_net":
        return config.elastic_net.bounded_execution, run_bounded_elastic_net_job
    raise ValueError(f"unsupported synthetic ERM problem type: {problem_type}")


def _candidates(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("at least one native-tolerance candidate is required")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("native-tolerance candidates must be finite and positive")
    if len(set(values)) != len(values):
        raise ValueError("native-tolerance candidates must not contain duplicates")
    return sorted(values, reverse=True)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically persist a calibration summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def calibrate_synthetic_erm(
    *,
    problem_type: str,
    solver: str,
    backend: str,
    candidates: list[float],
    config: SyntheticErmConfig,
    output_dir: Path,
) -> float:
    """Select the loosest native tolerance passing every configured case.

    Each candidate/case pair runs in a fresh worker process. Problem generation
    is deterministic and excluded from solver timing, while fresh processes
    prevent solver state, JIT state, or warm starts from leaking across
    tolerance candidates.
    """
    if problem_type not in _PROBLEM_TYPES:
        raise ValueError(f"unsupported synthetic ERM problem type: {problem_type}")
    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be cpu or cuda")
    ordered_candidates = _candidates(candidates)
    controls, runner = _execution(config, problem_type)
    jobs = [
        job
        for job in build_synthetic_erm_manifest(config, backend)
        if job["problem_type"] == problem_type and job["solver"] == solver
    ]
    if not jobs:
        raise ValueError(f"solver {solver!r} is not configured for {backend}/{problem_type}")

    # Calibration measures solution qualification, not runtime variability.
    # Suppress timing warmups and repetitions while preserving all mathematical
    # and solver controls from the calibration configuration.
    calibration_config = replace(config, warmups=0, repetitions=1)
    outcomes: dict[str, list[dict[str, Any]]] = {}
    valid_candidates: list[float] = []
    for candidate_index, candidate in enumerate(ordered_candidates):
        candidate_results: list[dict[str, Any]] = []
        candidate_directory = output_dir / "candidates" / f"{candidate_index:02d}-{candidate:.0e}"
        for case_index, job in enumerate(jobs):
            records = runner(
                job,
                calibration_config,
                native_tolerance=candidate,
                max_iterations=controls.max_iterations,
                batch_size=controls.batch_size,
                output_dir=candidate_directory,
                record_run_key=(f"{problem_type}/calibration/native-tolerance-{candidate:.17g}"),
                record_metadata={
                    "benchmark_phase": "calibration",
                    "calibration_candidate": candidate,
                    "calibration_case_index": case_index,
                },
            )
            if len(records) != 1:
                raise RuntimeError("calibration must produce exactly one record per case")
            record = records[0]
            passed = record.native_success and record.external_success
            candidate_results.append(
                {
                    "case_index": case_index,
                    "problem_id": record.problem_id,
                    "run_id": record.run_id,
                    "native_status": record.native_status,
                    "native_success": record.native_success,
                    "external_success": record.external_success,
                    "passed": passed,
                    "iterations": record.iterations,
                    "runtime_seconds": record.runtime_seconds,
                    "metrics": record.metrics,
                }
            )
        outcomes[str(candidate)] = candidate_results
        if all(result["passed"] for result in candidate_results):
            valid_candidates.append(candidate)

    selected = max(valid_candidates) if valid_candidates else None
    summary = {
        "suite": config.suite,
        "problem_type": problem_type,
        "solver": solver,
        "backend": backend,
        "selection_rule": (
            "loosest candidate with native and external success on every "
            "configured calibration case"
        ),
        "selected_tolerance": selected,
        "accuracy_thresholds": asdict(config.accuracy),
        "candidate_order": ordered_candidates,
        "case_count": len(jobs),
        "outcomes": outcomes,
    }
    _write_json(output_dir / "calibration.json", summary)
    if selected is None:
        raise RuntimeError("no native tolerance passed every calibration case")
    return selected

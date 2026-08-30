"""Native-tolerance calibration against the external KKT contract."""

from __future__ import annotations

import json
from pathlib import Path

from rlaopt_experiments.config import ExperimentConfig
from rlaopt_experiments.diagnostics import Accuracy, choose_native_tolerance
from rlaopt_experiments.runner import run_job


def calibrate(*, solver: str, backend: str, candidates: list[float],
              config: ExperimentConfig, output_dir: Path) -> float:
    """Run fixed easy/middle/hard cases and return the loosest passing tolerance."""
    ordered_shapes = sorted(config.shapes, key=lambda shape: shape.n * shape.p)
    cases = (
        (ordered_shapes[0], min(config.alphas), max(config.lambdas)),
        (ordered_shapes[len(ordered_shapes) // 2], 1.0, 1e-4),
        (ordered_shapes[-1], max(config.alphas), min(config.lambdas)),
    )
    outcomes: dict[float, list[Accuracy]] = {}
    for candidate in sorted(set(candidates), reverse=True):
        values: list[Accuracy] = []
        for case_index, (shape, alpha, ridge) in enumerate(cases):
            records = run_job(
                n=shape.n, p=shape.p, alpha=alpha, seed=0, solver=solver,
                backend=backend, ridges=[ridge], native_tolerance=candidate,
                kkt_tolerance=config.kkt_tolerance,
                timeout_seconds=config.timeout_seconds, rank=config.nystrom_rank,
                warmups=config.warmups, repetitions=1,
                output_dir=output_dir / f"tol-{candidate:g}" / f"case-{case_index}",
            )
            record = records[0]
            if record.relative_kkt is not None and record.relative_solution_error is not None:
                values.append(Accuracy(record.relative_kkt, record.relative_solution_error,
                                       record.success))
            else:
                values.append(Accuracy(float("inf"), float("inf"), False))
        outcomes[candidate] = values
    selected = choose_native_tolerance(outcomes, config.kkt_tolerance)
    summary = {
        "solver": solver, "backend": backend, "selected_tolerance": selected,
        "candidates": {str(key): [item.relative_kkt for item in value]
                       for key, value in outcomes.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration.json").write_text(json.dumps(summary, indent=2) + "\n")
    return selected

"""Native-tolerance calibration against the external KKT contract."""

from __future__ import annotations

import json
from pathlib import Path

from rlaopt_experiments.config import ExperimentConfig
from rlaopt_experiments.diagnostics import Accuracy, choose_native_tolerance
from rlaopt_experiments.runner import run_tolerance_sweep


def calibrate(
    *,
    solver: str,
    backend: str,
    candidates: list[float],
    config: ExperimentConfig,
    output_dir: Path,
) -> float:
    """Run fixed easy/middle/hard cases and return the loosest passing tolerance."""
    ordered_shapes = sorted(config.shapes, key=lambda shape: shape.n * shape.p)
    cases = (
        (ordered_shapes[0], min(config.alphas), max(config.lambdas)),
        (ordered_shapes[len(ordered_shapes) // 2], 1.0, 1e-4),
        (ordered_shapes[-1], max(config.alphas), min(config.lambdas)),
    )
    unique_candidates = sorted(set(candidates), reverse=True)
    outcomes: dict[float, list[Accuracy]] = {candidate: [] for candidate in unique_candidates}
    for case_index, (shape, alpha, ridge) in enumerate(cases):
        records = run_tolerance_sweep(
            n=shape.n,
            p=shape.p,
            alpha=alpha,
            ridge=ridge,
            seed=0,
            solver=solver,
            backend=backend,
            candidates=unique_candidates,
            kkt_tolerance=config.kkt_tolerance,
            timeout_seconds=config.timeout_seconds,
            startup_timeout_seconds=config.startup_timeout_seconds,
            rank=config.nystrom_rank,
            output_dir=output_dir,
            case_index=case_index,
        )
        for candidate, record in records.items():
            if record.relative_kkt is not None and record.relative_solution_error is not None:
                accuracy = Accuracy(
                    record.relative_kkt, record.relative_solution_error, record.success
                )
            else:
                accuracy = Accuracy(float("inf"), float("inf"), False)
            outcomes[candidate].append(accuracy)
    try:
        selected = choose_native_tolerance(outcomes, config.kkt_tolerance)
    except RuntimeError:
        selected = None
    summary = {
        "solver": solver,
        "backend": backend,
        "selected_tolerance": selected,
        "candidates": {
            str(key): [item.relative_kkt for item in value] for key, value in outcomes.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration.json").write_text(json.dumps(summary, indent=2) + "\n")
    if selected is None:
        raise RuntimeError("no native tolerance passed every calibration case")
    return selected

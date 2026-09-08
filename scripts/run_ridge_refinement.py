"""Execute one frozen ridge refinement trial, retaining every attempted tolerance."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

from rlaopt_experiments.execution import read_manifest_job
from rlaopt_experiments.records import write_record
from rlaopt_experiments.runner import run_job


def run_trial(job: dict, output: Path, runner=run_job) -> dict:
    source = Path(job["original_record"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != job["original_record_sha256"]:
        raise ValueError("Original record checksum mismatch")
    actual_image = os.environ.get("RLAOPT_CUDA_IMAGE_SHA256")
    if actual_image != job["image_sha256"]:
        raise ValueError("Refinement requires the frozen container image")
    trial = output / job["trial_id"]
    trial.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = {"trial_id": job["trial_id"], "original_record": str(source),
               "original_runtime_seconds": job["original_runtime_seconds"], "attempts": []}
    problem = dict(job["problem"])
    ridge = problem.pop("ridge")
    try:
        for tolerance in job["native_tolerances"]:
            attempt_output = trial / f"native-{tolerance:.0e}"
            attempt_started = time.perf_counter()
            records = runner(**problem, **job["controls"], ridges=[ridge],
                             native_tolerance=tolerance, output_dir=attempt_output)
            if len(records) != 1:
                raise RuntimeError("Expected exactly one measurement per refinement attempt")
            record = records[0]
            record = replace(record, metadata=record.metadata | {
                "benchmark_phase": "post-production accuracy refinement",
                "original_record_sha256": job["original_record_sha256"],
                "original_image_sha256": job["original_image_sha256"],
                "native_tolerance": tolerance,
                "native_tolerances_calibrated": False,
            })
            write_record(attempt_output / "records" / f"{record.run_id}.json", record)
            summary["attempts"].append({
                "native_tolerance": tolerance, "native_status": record.native_status,
                "native_success": record.native_success, "external_success": record.external_success,
                "runtime_seconds": record.runtime_seconds,
                "attempt_wall_seconds": time.perf_counter() - attempt_started,
                "relative_kkt": record.metrics.get("relative_kkt"),
                "record": str(attempt_output / "records" / f"{record.run_id}.json"),
            })
            if not (record.native_success and not record.external_success):
                break
    finally:
        summary["qualifying_runtime_seconds"] = next(
            (a["runtime_seconds"] for a in summary["attempts"]
             if a["native_success"] and a["external_success"]), None
        )
        summary["refinement_wall_seconds"] = time.perf_counter() - started
        summary["total_measured_solver_seconds_including_original"] = (
            job["original_runtime_seconds"] + sum(a["runtime_seconds"] for a in summary["attempts"])
        )
        (trial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_trial(read_manifest_job(args.manifest, args.index), args.output), indent=2))


if __name__ == "__main__":
    main()

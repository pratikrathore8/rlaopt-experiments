"""Single-job benchmark runner with isolated, hard-timeout solver calls."""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import uuid
from importlib.metadata import version
from pathlib import Path
from typing import Any

from rlaopt_experiments.isolation import ProblemWorker
from rlaopt_experiments.problem import ProblemSpec
from rlaopt_experiments.records import TrialRecord, write_record
from rlaopt_experiments.seeds import derive_seed
from rlaopt_experiments.tracking import log_record, wandb_run


def _environment_metadata() -> dict[str, Any]:
    git_commit = os.environ.get("RLAOPT_GIT_COMMIT")
    if git_commit is None:
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = None
    dirty_environment = os.environ.get("RLAOPT_GIT_DIRTY")
    if dirty_environment is not None:
        git_dirty = dirty_environment == "1"
    else:
        try:
            git_dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
        except (OSError, subprocess.CalledProcessError):
            git_dirty = None
    try:
        driver_version = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver_version = None
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "cuda_base_image_digest": os.environ.get("CUDA_BASE_IMAGE_DIGEST"),
        "cuda_image_sha256": os.environ.get("RLAOPT_CUDA_IMAGE_SHA256"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "nvidia_driver_version": driver_version,
        "rlaopt_version": version("rlaopt"),
        "jax_version": version("jax"),
        "jaxopt_version": version("jaxopt"),
        "numpy_version": version("numpy"),
        "scipy_version": version("scipy"),
        "wandb_version": version("wandb"),
    }


def record_outcome(
    *,
    suite: str,
    problem_id: str,
    run_key: str,
    problem: dict[str, Any],
    solver: str,
    backend: str,
    seed: int,
    repetition: int,
    outcome: dict[str, Any],
    runtimes: list[float],
    output_dir: Path,
    worker_metadata: dict[str, Any],
    timing_scope: str,
) -> TrialRecord:
    """Persist and optionally log one suite-neutral worker outcome."""
    run_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{suite}/{problem_id}/{solver}/{backend}/{run_key}/{repetition}",
    ).hex
    accuracy = dict(outcome.get("accuracy", {}))
    if outcome["kind"] == "result":
        external_success = bool(accuracy.pop("external_success"))
        native_success = bool(outcome["native_success"])
        runtime_eligible = bool(outcome["runtime_eligible"])
    else:
        external_success = False
        native_success = False
        runtime_eligible = False
    metadata = (
        _environment_metadata()
        | worker_metadata
        | outcome.get("solver_metadata", {})
        | {
            "worker_outcome": outcome["kind"],
            "timing_scope": timing_scope,
        }
    )
    for key in ("error_type", "error_message", "traceback"):
        if key in outcome:
            metadata[key] = outcome[key]
    record = TrialRecord(
        run_id=run_id,
        problem_id=problem_id,
        suite=suite,
        solver=solver,
        backend=backend,
        seed=seed,
        repetition=repetition,
        timings={
            "runtime_seconds": outcome["runtime_seconds"],
            "runtime_median_seconds": statistics.median(runtimes),
            "runtime_min_seconds": min(runtimes),
            "runtime_max_seconds": max(runtimes),
        },
        iterations=outcome.get("iterations"),
        native_status=outcome["native_status"],
        metrics=accuracy,
        native_success=native_success,
        external_success=external_success,
        runtime_eligible=runtime_eligible,
        timed_out=outcome["kind"] == "timeout" or outcome["native_status"] == "timeout",
        problem=problem | {"diagnostics": outcome.get("diagnostics", {})},
        peak_memory_bytes=outcome.get("peak_memory_bytes"),
        trace=outcome.get("trace", []),
        metadata=metadata,
    )
    write_record(output_dir / "records" / f"{run_id}.json", record)
    with wandb_run(record, output_dir) as run:
        log_record(run, record)
    return record


def _record(
    *,
    suite: str,
    problem_spec: ProblemSpec,
    solver: str,
    backend: str,
    ridge: float,
    seed: int,
    repetition: int,
    outcome: dict[str, Any],
    runtimes: list[float],
    output_dir: Path,
    worker_metadata: dict[str, Any],
) -> TrialRecord:
    """Preserve the synthetic-ridge record identity and problem fields."""
    return record_outcome(
        suite=suite,
        problem_id=problem_spec.problem_id,
        run_key=str(ridge),
        problem={
            "n": problem_spec.n,
            "p": problem_spec.p,
            "alpha": problem_spec.alpha,
            "ridge": ridge,
        },
        solver=solver,
        backend=backend,
        seed=seed,
        repetition=repetition,
        outcome=outcome,
        runtimes=runtimes,
        output_dir=output_dir,
        worker_metadata=worker_metadata,
        timing_scope="device-resident; includes setup, excludes data generation",
    )


def run_job(
    *,
    n: int,
    p: int,
    alpha: float,
    seed: int,
    solver: str,
    backend: str,
    ridges: list[float],
    native_tolerance: float,
    kkt_tolerance: float,
    timeout_seconds: int,
    startup_timeout_seconds: int,
    rank: int,
    warmups: int,
    repetitions: int,
    output_dir: Path,
    suite: str,
) -> list[TrialRecord]:
    problem_spec = ProblemSpec(
        n, p, alpha, derive_seed(seed, "factors"), derive_seed(seed, "response")
    )
    specification = {
        "n": n,
        "p": p,
        "alpha": alpha,
        "factor_seed": problem_spec.factor_seed,
        "response_seed": problem_spec.response_seed,
    }
    command = {
        "solver": solver,
        "native_tolerance": native_tolerance,
        "kkt_tolerance": kkt_tolerance,
        "timeout_seconds": timeout_seconds,
        "rank": rank,
        "nystrom_seed": derive_seed(seed, "nystrom"),
    }
    records: list[TrialRecord] = []

    def start_worker() -> tuple[ProblemWorker, dict[str, Any]]:
        new_worker = ProblemWorker(specification, backend, suite)
        return new_worker, new_worker.wait_until_ready(startup_timeout_seconds)

    worker, ready = start_worker()
    if ready["kind"] != "ready":
        # Startup errors have no ridge-specific solve; record one failure per ridge.
        for ridge in ridges:
            outcome = ready | {"runtime_seconds": 0.0}
            records.append(
                _record(
                    suite=suite,
                    problem_spec=problem_spec,
                    solver=solver,
                    backend=backend,
                    ridge=ridge,
                    seed=seed,
                    repetition=0,
                    outcome=outcome,
                    runtimes=[0.0],
                    output_dir=output_dir,
                    worker_metadata=ready.get("worker_metadata", {}),
                )
            )
        worker.close()
        return records

    try:
        for ridge in ridges:
            if not worker.alive:
                worker.close()
                worker, ready = start_worker()
                if ready["kind"] != "ready":
                    outcome = ready | {"runtime_seconds": 0.0}
                    records.append(
                        _record(
                            suite=suite,
                            problem_spec=problem_spec,
                            solver=solver,
                            backend=backend,
                            ridge=ridge,
                            seed=seed,
                            repetition=0,
                            outcome=outcome,
                            runtimes=[0.0],
                            output_dir=output_dir,
                            worker_metadata=ready.get("worker_metadata", {}),
                        )
                    )
                    continue
            maximum_iterations = 2 * p
            warmup_failed = False
            for _ in range(warmups):
                warmup = worker.solve(
                    command
                    | {
                        "ridge": ridge,
                        "max_iters": min(maximum_iterations, 10),
                    },
                    timeout_seconds,
                )
                if warmup["kind"] != "result":
                    records.append(
                        _record(
                            suite=suite,
                            problem_spec=problem_spec,
                            solver=solver,
                            backend=backend,
                            ridge=ridge,
                            seed=seed,
                            repetition=0,
                            outcome=warmup,
                            runtimes=[warmup["runtime_seconds"]],
                            output_dir=output_dir,
                            worker_metadata=ready["worker_metadata"],
                        )
                    )
                    warmup_failed = True
                    break
            if warmup_failed:
                continue

            first = worker.solve(
                command
                | {
                    "ridge": ridge,
                    "max_iters": maximum_iterations,
                },
                timeout_seconds,
            )
            outcomes = [first]
            if first["kind"] == "result" and first["runtime_eligible"]:
                for _ in range(repetitions - 1):
                    repeated = worker.solve(
                        command
                        | {
                            "ridge": ridge,
                            "max_iters": maximum_iterations,
                        },
                        timeout_seconds,
                    )
                    outcomes.append(repeated)
                    if not worker.alive:
                        break
            runtimes = [item["runtime_seconds"] for item in outcomes]
            for repetition, outcome in enumerate(outcomes):
                records.append(
                    _record(
                        suite=suite,
                        problem_spec=problem_spec,
                        solver=solver,
                        backend=backend,
                        ridge=ridge,
                        seed=seed,
                        repetition=repetition,
                        outcome=outcome,
                        runtimes=runtimes,
                        output_dir=output_dir,
                        worker_metadata=ready["worker_metadata"],
                    )
                )
    finally:
        worker.close()
    return records


def run_tolerance_sweep(
    *,
    n: int,
    p: int,
    alpha: float,
    ridge: float,
    seed: int,
    solver: str,
    backend: str,
    candidates: list[float],
    kkt_tolerance: float,
    timeout_seconds: int,
    startup_timeout_seconds: int,
    rank: int,
    output_dir: Path,
    case_index: int,
    suite: str,
) -> dict[float, TrialRecord]:
    """Generate one problem and evaluate several native tolerances against it."""
    problem_spec = ProblemSpec(
        n, p, alpha, derive_seed(seed, "factors"), derive_seed(seed, "response")
    )
    specification = {
        "n": n,
        "p": p,
        "alpha": alpha,
        "factor_seed": problem_spec.factor_seed,
        "response_seed": problem_spec.response_seed,
    }
    maximum_iterations = 2 * p
    records: dict[float, TrialRecord] = {}

    def start_worker() -> tuple[ProblemWorker, dict[str, Any]]:
        new_worker = ProblemWorker(specification, backend, suite)
        return new_worker, new_worker.wait_until_ready(startup_timeout_seconds)

    worker, ready = start_worker()
    try:
        for candidate in sorted(set(candidates), reverse=True):
            if not worker.alive:
                worker.close()
                worker, ready = start_worker()
            command = {
                "solver": solver,
                "native_tolerance": candidate,
                "kkt_tolerance": kkt_tolerance,
                "timeout_seconds": timeout_seconds,
                "rank": rank,
                "nystrom_seed": derive_seed(seed, "nystrom"),
                "ridge": ridge,
                "max_iters": maximum_iterations,
            }
            if ready["kind"] == "ready":
                outcome = worker.solve(command, timeout_seconds)
            else:
                outcome = ready | {"runtime_seconds": 0.0}
            solver_metadata = outcome.get("solver_metadata", {}) | {
                "native_tolerance": candidate,
            }
            outcome = outcome | {"solver_metadata": solver_metadata}
            candidate_output = output_dir / f"tol-{candidate:g}" / f"case-{case_index}"
            records[candidate] = _record(
                suite=suite,
                problem_spec=problem_spec,
                solver=solver,
                backend=backend,
                ridge=ridge,
                seed=seed,
                repetition=0,
                outcome=outcome,
                runtimes=[outcome["runtime_seconds"]],
                output_dir=candidate_output,
                worker_metadata=ready.get("worker_metadata", {}),
            )
    finally:
        worker.close()
    return records

"""Single-job benchmark runner."""

from __future__ import annotations

import gc
import platform
import statistics
import uuid
from pathlib import Path

import torch

from rlaopt_experiments.diagnostics import adjudicate
from rlaopt_experiments.problem import ProblemSpec, generate_problem
from rlaopt_experiments.records import TrialRecord, write_record
from rlaopt_experiments.seeds import derive_seed
from rlaopt_experiments.solvers import solve
from rlaopt_experiments.tracking import log_record, wandb_run


def run_job(*, n: int, p: int, alpha: float, seed: int, solver: str, backend: str,
            ridges: list[float], native_tolerance: float, kkt_tolerance: float,
            timeout_seconds: int, rank: int, warmups: int, fast_repetitions: int,
            fast_threshold_seconds: int, output_dir: Path) -> list[TrialRecord]:
    device = torch.device("cuda" if backend == "cuda" else "cpu")
    torch.set_default_dtype(torch.float64)
    problem = generate_problem(ProblemSpec(n, p, alpha, derive_seed(seed, "factors"),
                                           derive_seed(seed, "response")))
    problem.X = problem.X.pin_memory() if backend == "cuda" else problem.X
    problem.y = problem.y.pin_memory() if backend == "cuda" else problem.y
    if backend == "cuda":
        problem.X = problem.X.to(device, non_blocking=True)
        problem.y = problem.y.to(device, non_blocking=True)
        problem.U = problem.U.to(device, non_blocking=True)
        problem.V = problem.V.to(device, non_blocking=True)
        problem.singular_values = problem.singular_values.to(device, non_blocking=True)
        problem.response_coordinates = problem.response_coordinates.to(device, non_blocking=True)
        torch.cuda.synchronize()

    records: list[TrialRecord] = []
    nystrom_seed = derive_seed(seed, "nystrom")

    def run_once(ridge: float, maximum_iterations: int):
        torch.manual_seed(nystrom_seed)
        if backend == "cuda":
            torch.cuda.manual_seed_all(nystrom_seed)
        return solve(solver, problem, ridge, native_tolerance, maximum_iterations,
                     timeout_seconds, rank)

    for ridge in ridges:
        maximum_iterations = 2 * p
        for _ in range(warmups):
            run_once(ridge, min(maximum_iterations, 10))
        if backend == "cuda":
            torch.cuda.reset_peak_memory_stats()
        first = run_once(ridge, maximum_iterations)
        results = [first]
        if first.runtime_seconds < fast_threshold_seconds:
            results.extend(run_once(ridge, maximum_iterations)
                           for _ in range(fast_repetitions - 1))
        runtimes = [result.runtime_seconds for result in results]
        for repetition, result in enumerate(results):
            accuracy = adjudicate(problem, ridge, result, kkt_tolerance)
            run_id = uuid.uuid5(uuid.NAMESPACE_URL, (
                f"{problem.spec.problem_id}/{solver}/{backend}/{ridge}/{repetition}"
            )).hex
            record = TrialRecord(
                run_id=run_id, problem_id=problem.spec.problem_id, solver=solver,
                backend=backend, n=n, p=p, alpha=alpha, ridge=ridge, seed=seed,
                repetition=repetition, runtime_seconds=result.runtime_seconds,
                iterations=result.iterations, native_status=result.native_status,
                relative_kkt=accuracy.relative_kkt,
                relative_solution_error=accuracy.relative_solution_error,
                success=accuracy.success, timed_out=result.native_status == "timeout",
                peak_memory_bytes=(torch.cuda.max_memory_allocated() if backend == "cuda" else None),
                diagnostics=problem.diagnostics(ridge), metadata={
                    **result.metadata, "runtime_median_seconds": statistics.median(runtimes),
                    "runtime_min_seconds": min(runtimes), "runtime_max_seconds": max(runtimes),
                    "hostname": platform.node(), "device": str(device),
                    "timing_scope": "device-resident; includes setup, excludes data generation and H2D",
                })
            destination = output_dir / "records" / f"{run_id}.json"
            write_record(destination, record)
            with wandb_run(record, output_dir) as run:
                log_record(run, record, result.trace)
            records.append(record)
        gc.collect()
    return records

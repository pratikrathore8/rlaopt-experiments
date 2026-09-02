"""Smoke-test the production Clarabel adapters on one bounded elastic-net QP."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from rlaopt_experiments.clarabel_bridge import (
    ClarabelRuntime,
    initialize_clarabel_runtime,
)

if TYPE_CHECKING:
    from rlaopt_experiments.problems.synthetic_erm import ElasticNetProblem
    from rlaopt_experiments.suites.synthetic_erm.bounded_elastic_net_solvers import (
        BoundedElasticNetSolverResult,
    )


def make_problem(
    *,
    device: str,
    n_samples: int = 32,
    n_features: int = 8,
) -> ElasticNetProblem:
    """Construct the same deterministic bounded elastic-net problem on either device."""
    from rlaopt_experiments.problems.synthetic_erm import (
        ElasticNetSpec,
        generate_elastic_net_problem,
    )

    return generate_elastic_net_problem(
        ElasticNetSpec(
            n=n_samples,
            p=n_features,
            feature_seed=20260902,
            target_seed=20260903,
            teacher_density=0.25,
            noise_ratio=0.02,
            teacher_intercept=0.4,
            regularization_fraction=0.05,
        ),
        bounded=True,
        device=device,
    )


def solve(
    runtime: ClarabelRuntime,
    problem: ElasticNetProblem,
) -> BoundedElasticNetSolverResult:
    from rlaopt_experiments.suites.synthetic_erm.bounded_elastic_net_solvers import (
        solve_clarabel_qdldl,
        solve_cuclarabel_cudss,
    )

    adapter = solve_clarabel_qdldl if runtime.backend == "cpu" else solve_cuclarabel_cudss
    return adapter(
        problem,
        runtime=runtime,
        native_tolerance=1e-10,
        max_iterations=200,
    )


def validate(
    problem: ElasticNetProblem,
    result: BoundedElasticNetSolverResult,
) -> tuple[float, float, float]:
    if result.native_status != "converged":
        raise RuntimeError(f"{result.metadata['linear_solver']} returned {result.native_status}")
    objective = float(problem.objective(result.weights, result.intercept))
    stationarity = float(
        problem.kkt_residual(result.weights, result.intercept, activity_tolerance=1e-8)
    )
    feasibility = float(problem.constraint_violation(result.weights))
    if stationarity > 1e-7:
        raise RuntimeError(f"stationarity is {stationarity:.3e}")
    if feasibility > 1e-8:
        raise RuntimeError(f"constraint violation is {feasibility:.3e}")
    return objective, stationarity, feasibility


def summary(
    result: BoundedElasticNetSolverResult,
    metrics: tuple[float, float, float],
) -> dict[str, Any]:
    objective, stationarity, feasibility = metrics
    return {
        "backend": result.metadata["linear_solver"],
        "status": result.native_status,
        "iterations": result.iterations,
        "elapsed_seconds": result.runtime_seconds,
        "native_setup_seconds": result.metadata["native_setup_time_seconds"],
        "native_solve_seconds": result.metadata["native_solve_time_seconds"],
        "preparation_seconds": result.metadata["conic_preparation_seconds"],
        "extraction_seconds": result.metadata["solution_extraction_seconds"],
        "objective": objective,
        "stationarity": stationarity,
        "constraint_violation": feasibility,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "cuda", "both"), default="both")
    args = parser.parse_args()

    backends = ["cpu", "cuda"] if args.backend == "both" else [args.backend]
    # Initialize every requested Julia backend before make_problem imports PyTorch.
    runtimes = {backend: initialize_clarabel_runtime(backend) for backend in backends}
    problems = {backend: make_problem(device=backend) for backend in backends}

    results: dict[str, BoundedElasticNetSolverResult] = {}
    metrics: dict[str, tuple[float, float, float]] = {}
    for backend in backends:
        # The first solve is an untimed JIT warmup. The measured solve creates
        # fresh native solver state and fresh CuPy CSR buffers.
        validate(problems[backend], solve(runtimes[backend], problems[backend]))
        result = solve(runtimes[backend], problems[backend])
        result_metrics = validate(problems[backend], result)
        results[backend] = result
        metrics[backend] = result_metrics
        print(json.dumps(summary(result, result_metrics), sort_keys=True))

    if len(results) == 2:
        cpu_result = results["cpu"]
        cuda_result = results["cuda"]
        objective_difference = abs(metrics["cpu"][0] - metrics["cuda"][0])
        cpu_solution = np.concatenate(
            (cpu_result.weights.cpu().numpy(), [float(cpu_result.intercept)])
        )
        cuda_solution = np.concatenate(
            (cuda_result.weights.cpu().numpy(), [float(cuda_result.intercept)])
        )
        coefficient_difference = float(
            np.linalg.norm(cpu_solution - cuda_solution, ord=np.inf)
        )
        if objective_difference > 1e-9 or coefficient_difference > 1e-6:
            raise RuntimeError(
                "CPU/GPU disagreement: "
                f"objective={objective_difference:.3e}, solution={coefficient_difference:.3e}"
            )
        print(
            json.dumps(
                {
                    "cpu_gpu_objective_difference": objective_difference,
                    "cpu_gpu_solution_inf_difference": coefficient_difference,
                },
                sort_keys=True,
            )
        )

    print("CUCLARABEL_DIRECT_INTERFACE_OK=true")


if __name__ == "__main__":
    main()

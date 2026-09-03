"""Smoke-test the production SCS CUDA adapter and compiled backend identity."""

from __future__ import annotations

import json

import torch
from scs import _scs_gpu

from rlaopt_experiments.problems.synthetic_erm import (
    ElasticNetSpec,
    generate_elastic_net_problem,
)
from rlaopt_experiments.suites.synthetic_erm.bounded_elastic_net_solvers import (
    solve_scs_cuda,
)

COMMON_ACCURACY_TOLERANCE = 1e-6


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SCS CUDA smoke test requires a visible GPU")
    if _scs_gpu.sizeof_float() != 8:
        raise RuntimeError("SCS CUDA was not compiled in float64")
    if _scs_gpu.sizeof_int() != 4:
        raise RuntimeError("SCS CUDA was not compiled with required 32-bit indices")

    problem = generate_elastic_net_problem(
        ElasticNetSpec(
            n=64,
            p=8,
            feature_seed=20260902,
            target_seed=20260903,
            teacher_density=0.25,
            noise_ratio=0.02,
            teacher_intercept=0.4,
            regularization_fraction=0.05,
        ),
        bounded=True,
        device="cuda",
    )
    result = solve_scs_cuda(
        problem,
        native_tolerance=1e-8,
        max_iterations=10_000,
    )
    stationarity = float(
        problem.kkt_residual(
            result.weights,
            result.intercept,
            activity_tolerance=COMMON_ACCURACY_TOLERANCE,
        )
    )
    feasibility = float(problem.constraint_violation(result.weights))
    if result.native_status != "converged":
        raise RuntimeError(f"SCS CUDA returned {result.native_status}")
    if stationarity > COMMON_ACCURACY_TOLERANCE or feasibility > COMMON_ACCURACY_TOLERANCE:
        raise RuntimeError(
            f"SCS CUDA accuracy failure: stationarity={stationarity:.3e}, "
            f"feasibility={feasibility:.3e}"
        )
    print(
        json.dumps(
            {
                "constraint_violation": feasibility,
                "iterations": result.iterations,
                "linear_solver": result.metadata["linear_solver"],
                "runtime_seconds": result.runtime_seconds,
                "stationarity": stationarity,
                "status": result.native_status,
            },
            sort_keys=True,
        )
    )
    print("SCS_CUDA_INTERFACE_OK=true")


if __name__ == "__main__":
    main()

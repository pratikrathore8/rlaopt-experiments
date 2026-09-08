"""Check both SCS linear solvers on one device at the frozen production tolerance."""
from __future__ import annotations

import argparse
from importlib import import_module
import json

import scs
import torch

from rlaopt_experiments.problems.synthetic_erm import ElasticNetSpec, generate_elastic_net_problem
from rlaopt_experiments.suites.synthetic_erm.bounded_elastic_net_solvers import (
    solve_scs, solve_scs_cpu_indirect, solve_scs_cuda, solve_scs_cuda_direct,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    if args.backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend checks require a visible GPU")
    adapters = (
        [("scs", "_scs_direct", solve_scs),
         ("scs_cpu_indirect", "_scs_indirect", solve_scs_cpu_indirect)]
        if args.backend == "cpu" else
        [("scs_cuda", "_scs_gpu", solve_scs_cuda),
         ("scs_cuda_direct", "_scs_cudss", solve_scs_cuda_direct)]
    )
    problem = generate_elastic_net_problem(
        ElasticNetSpec(
            n=64, p=8, feature_seed=20260902, target_seed=20260903,
            feature_generator="standardized_gaussian", feature_decay_exponent=None,
            teacher_density=0.25, noise_ratio=0.02, teacher_intercept=0.4,
            regularization_fraction=0.1,
        ), bounded=True, device=args.backend,
    )
    for name, extension_name, adapter in adapters:
        extension = import_module(f"scs.{extension_name}")
        if extension.sizeof_float() != 8:
            raise RuntimeError(f"{name} requires float64")
        if args.backend == "cuda" and extension.sizeof_int() != 4:
            raise RuntimeError(f"{name} requires 32-bit indices")
        # Verify the pinned Python interface selects the expected compiled module.
        settings = {"gpu": args.backend == "cuda", "use_indirect": "indirect" in name
                    or name == "scs_cuda"}
        if scs._select_scs_module(settings.copy()) is not extension:
            raise RuntimeError(f"{name} selected an unexpected compiled backend")
        result = adapter(problem, native_tolerance=1e-7, max_iterations=100_000)
        stationarity = float(problem.kkt_residual(
            result.weights, result.intercept, activity_tolerance=1e-6,
        ))
        feasibility = float(problem.constraint_violation(result.weights))
        if result.native_status != "converged" or not (
            stationarity <= 1e-4 and feasibility <= 1e-6
        ):
            raise RuntimeError(
                f"{name}: status={result.native_status}, "
                f"stationarity={stationarity}, feasibility={feasibility}"
            )
        print(json.dumps({
            "solver": name, "scs_version": scs.__version__,
            "extension": extension_name, "float_bytes": extension.sizeof_float(),
            "index_bytes": extension.sizeof_int(), "stationarity": stationarity,
            "feasibility": feasibility, "status": result.native_status,
            "iterations": result.iterations, **result.metadata,
        }, sort_keys=True), flush=True)
    print(f"SCS_{args.backend.upper()}_BACKENDS_OK=true", flush=True)


if __name__ == "__main__":
    main()

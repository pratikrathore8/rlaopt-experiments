"""Smoke-test CuClarabel's direct Julia interface on one bounded elastic-net QP."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rlaopt_experiments.problems.synthetic_erm import ElasticNetProblem


_JULIA_HELPERS = r"""
using Clarabel, PythonCall, SparseArrays

function rlaopt_solve_cpu(P, q, A, b, equality_dim, inequality_dim)
    settings = Clarabel.Settings(
        direct_solve_method = :qdldl,
        verbose = false,
        tol_gap_abs = 1e-10,
        tol_gap_rel = 1e-10,
        tol_feas = 1e-10,
    )
    cones = [Clarabel.ZeroConeT(equality_dim), Clarabel.NonnegativeConeT(inequality_dim)]
    solver = Clarabel.Solver(P, q, A, b, cones, settings)
    Clarabel.solve!(solver)
    return solver
end

function rlaopt_solve_gpu(P, q, A, b, equality_dim, inequality_dim)
    settings = Clarabel.Settings(
        direct_solve_method = :cudss,
        verbose = false,
        tol_gap_abs = 1e-10,
        tol_gap_rel = 1e-10,
        tol_feas = 1e-10,
    )
    cones = [Clarabel.ZeroConeT(equality_dim), Clarabel.NonnegativeConeT(inequality_dim)]
    solver = Clarabel.Solver(P, q, A, b, cones, settings)
    Clarabel.solve!(solver)
    CUDA.synchronize()
    return solver
end
"""


@dataclass(frozen=True)
class QuadraticProgram:
    canonical: ElasticNetProblem
    p_matrix: np.ndarray
    q_vector: np.ndarray
    a_matrix: np.ndarray
    b_vector: np.ndarray

    @property
    def equality_dim(self) -> int:
        return self.canonical.spec.n

    @property
    def inequality_dim(self) -> int:
        return 2 * self.canonical.spec.p


@dataclass(frozen=True)
class SolveResult:
    backend: str
    solution: np.ndarray
    status: str
    iterations: int
    elapsed_seconds: float
    native_setup_seconds: float
    native_solve_seconds: float
    objective: float
    stationarity: float
    constraint_violation: float


def make_problem(*, n_samples: int = 32, n_features: int = 8) -> QuadraticProgram:
    """Construct a deterministic bounded elastic-net problem in Clarabel form."""
    from rlaopt_experiments.problems.synthetic_erm import (
        ElasticNetSpec,
        generate_elastic_net_problem,
    )

    canonical = generate_elastic_net_problem(
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
        device="cpu",
    )
    x = canonical.X.numpy()
    y = canonical.y.numpy()

    # z = [r, w, intercept], with r = Xw + intercept - y.
    n_variables = n_samples + n_features + 1
    p_matrix = np.zeros((n_variables, n_variables), dtype=np.float64)
    p_matrix[:n_samples, :n_samples] = np.eye(n_samples) / n_samples
    p_matrix[n_samples : n_samples + n_features, n_samples : n_samples + n_features] = (
        canonical.lambda_l2 * np.eye(n_features)
    )
    q_vector = np.zeros(n_variables, dtype=np.float64)
    q_vector[n_samples : n_samples + n_features] = canonical.lambda_l1

    equality = np.column_stack(
        (
            np.eye(n_samples),
            -x,
            -np.ones(n_samples, dtype=np.float64),
        )
    )
    lower = np.zeros((n_features, n_variables), dtype=np.float64)
    upper = np.zeros((n_features, n_variables), dtype=np.float64)
    feature_columns = np.arange(n_samples, n_samples + n_features)
    lower[np.arange(n_features), feature_columns] = -1.0
    upper[np.arange(n_features), feature_columns] = 1.0
    a_matrix = np.vstack((equality, lower, upper))
    b_vector = np.concatenate((-y, np.zeros(n_features), np.ones(n_features)))

    return QuadraticProgram(
        canonical=canonical,
        p_matrix=p_matrix,
        q_vector=q_vector,
        a_matrix=a_matrix,
        b_vector=b_vector,
    )


def initialize_julia() -> Any:
    from juliacall import Main as jl

    jl.seval("using Clarabel, PythonCall, SparseArrays, CUDA, CUDA.CUSPARSE")
    jl.seval(_JULIA_HELPERS)
    return jl


def _external_metrics(
    problem: QuadraticProgram, solution: np.ndarray
) -> tuple[float, float, float]:
    import torch

    n_samples = problem.canonical.spec.n
    n_features = problem.canonical.spec.p
    w = solution[n_samples : n_samples + n_features]
    intercept = solution[-1]
    weights = torch.from_numpy(w)
    objective = float(problem.canonical.objective(weights, intercept))
    stationarity = float(
        problem.canonical.kkt_residual(
            weights,
            intercept,
            activity_tolerance=1e-8,
        )
    )
    constraint_violation = float(problem.canonical.constraint_violation(weights))
    return objective, stationarity, constraint_violation


def _extract_result(
    problem: QuadraticProgram,
    solver: Any,
    *,
    backend: str,
    elapsed_seconds: float,
) -> SolveResult:
    solution = np.asarray(solver.solution.x, dtype=np.float64).copy()
    objective, stationarity, constraint_violation = _external_metrics(problem, solution)
    return SolveResult(
        backend=backend,
        solution=solution,
        status=str(solver.solution.status),
        iterations=int(solver.solution.iterations),
        elapsed_seconds=elapsed_seconds,
        native_setup_seconds=float(solver.solution.setup_phase_time),
        native_solve_seconds=float(solver.solution.solve_phase_time),
        objective=objective,
        stationarity=stationarity,
        constraint_violation=constraint_violation,
    )


def solve_cpu(jl: Any, problem: QuadraticProgram) -> SolveResult:
    # Conversion is intentionally outside the solver-timing region.
    p_matrix = jl.sparse(jl.Matrix[jl.Float64](problem.p_matrix))
    q_vector = jl.Vector[jl.Float64](problem.q_vector)
    a_matrix = jl.sparse(jl.Matrix[jl.Float64](problem.a_matrix))
    b_vector = jl.Vector[jl.Float64](problem.b_vector)
    started = time.perf_counter()
    solver = jl.rlaopt_solve_cpu(
        p_matrix,
        q_vector,
        a_matrix,
        b_vector,
        problem.equality_dim,
        problem.inequality_dim,
    )
    elapsed_seconds = time.perf_counter() - started
    return _extract_result(problem, solver, backend="cpu-qdldl", elapsed_seconds=elapsed_seconds)


def solve_gpu(jl: Any, problem: QuadraticProgram) -> SolveResult:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix

    pyext = jl.Base.get_extension(jl.Clarabel, jl.Symbol("PythonExt"))

    # CuClarabel's bridge adds one to CSR indices and row pointers in place. These
    # dedicated arrays must never be reused, and their owners must outlive solve!.
    p_owner = csr_matrix(cp.asarray(problem.p_matrix, dtype=cp.float64))
    q_owner = cp.asarray(problem.q_vector, dtype=cp.float64)
    a_owner = csr_matrix(cp.asarray(problem.a_matrix, dtype=cp.float64))
    b_owner = cp.asarray(problem.b_vector, dtype=cp.float64)
    owners = (p_owner, q_owner, a_owner, b_owner)

    def wrap_vector(array: Any) -> Any:
        return pyext.cupy_to_cuvector(jl.Float64, int(array.data.ptr), array.size)

    def wrap_csr(matrix: Any) -> Any:
        return pyext.cupy_to_cucsrmat(
            jl.Float64,
            int(matrix.data.data.ptr),
            int(matrix.indices.data.ptr),
            int(matrix.indptr.data.ptr),
            matrix.shape[0],
            matrix.shape[1],
            matrix.nnz,
        )

    # Pointer wrapping and the bridge's index conversion are not solver time.
    p_matrix = wrap_csr(p_owner)
    q_vector = wrap_vector(q_owner)
    a_matrix = wrap_csr(a_owner)
    b_vector = wrap_vector(b_owner)
    started = time.perf_counter()
    solver = jl.rlaopt_solve_gpu(
        p_matrix,
        q_vector,
        a_matrix,
        b_vector,
        problem.equality_dim,
        problem.inequality_dim,
    )
    cp.cuda.get_current_stream().synchronize()
    elapsed_seconds = time.perf_counter() - started
    result = _extract_result(problem, solver, backend="gpu-cudss", elapsed_seconds=elapsed_seconds)
    del owners
    return result


def _validate(result: SolveResult) -> None:
    if result.status != "SOLVED":
        raise RuntimeError(f"{result.backend} returned status {result.status}")
    if result.stationarity > 1e-7:
        raise RuntimeError(f"{result.backend} stationarity is {result.stationarity:.3e}")
    if result.constraint_violation > 1e-8:
        raise RuntimeError(
            f"{result.backend} constraint violation is {result.constraint_violation:.3e}"
        )


def _summary(result: SolveResult) -> dict[str, Any]:
    return {
        "backend": result.backend,
        "status": result.status,
        "iterations": result.iterations,
        "elapsed_seconds": result.elapsed_seconds,
        "native_setup_seconds": result.native_setup_seconds,
        "native_solve_seconds": result.native_solve_seconds,
        "objective": result.objective,
        "stationarity": result.stationarity,
        "constraint_violation": result.constraint_violation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "cuda", "both"), default="both")
    args = parser.parse_args()

    jl = initialize_julia()
    # Importing the canonical problem module imports PyTorch. JuliaCall must be
    # initialized first to avoid a known shared-library conflict.
    problem = make_problem()
    methods = []
    if args.backend in {"cpu", "both"}:
        methods.append(solve_cpu)
    if args.backend in {"cuda", "both"}:
        methods.append(solve_gpu)

    results: list[SolveResult] = []
    for method in methods:
        # The first solve is an untimed JIT warmup. The measured solve uses fresh
        # solver state and, on GPU, fresh CSR buffers.
        _validate(method(jl, problem))
        result = method(jl, problem)
        _validate(result)
        results.append(result)
        print(json.dumps(_summary(result), sort_keys=True))

    if len(results) == 2:
        objective_difference = abs(results[0].objective - results[1].objective)
        coefficient_difference = float(
            np.linalg.norm(results[0].solution - results[1].solution, ord=np.inf)
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

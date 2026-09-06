"""Synthetic ERM suite adapter for isolated benchmark workers."""

from __future__ import annotations

from typing import Any

import torch

from rlaopt_experiments.problems.synthetic_erm import (
    GENERATOR_VERSION,
    ElasticNetProblem,
    ElasticNetSpec,
    MultinomialProblem,
    MultinomialSpec,
    generate_elastic_net_problem,
    generate_multinomial_problem,
)
from rlaopt_experiments.suites.synthetic_erm.bounded_elastic_net_solvers import (
    BoundedElasticNetSolverResult,
    solve_clarabel_qdldl,
    solve_cuclarabel_cudss,
    solve_rlaopt_admm,
    solve_scs,
    solve_scs_cuda,
    solve_scs_cpu_indirect,
    solve_scs_cuda_direct,
)
from rlaopt_experiments.suites.synthetic_erm.elastic_net_solvers import (
    ElasticNetSolverResult,
    solve_cuml_coordinate_descent,
    solve_jaxopt_proximal_gradient,
    solve_rlaopt_sapphire as solve_elastic_net_sapphire,
    solve_sklearn_coordinate_descent,
)
from rlaopt_experiments.suites.synthetic_erm.multinomial_solvers import (
    MultinomialSolverResult,
    solve_jaxopt_lbfgsb,
    solve_jaxopt_projected_gradient,
    solve_rlaopt_sapphire as solve_multinomial_sapphire,
)


class SyntheticErmSuite:
    """Generate and execute synthetic ERM problems behind the suite protocol."""

    name = "synthetic_erm"

    def generate(
        self,
        specification: dict[str, Any],
        device: torch.device,
    ) -> MultinomialProblem | ElasticNetProblem:
        problem_type = specification.get("problem_type")
        if problem_type == "multinomial":
            spec = MultinomialSpec(**specification["problem_spec"])
            return generate_multinomial_problem(spec, device=device)
        if problem_type in {"vanilla_elastic_net", "bounded_elastic_net"}:
            spec = ElasticNetSpec(**specification["problem_spec"])
            return generate_elastic_net_problem(
                spec,
                bounded=problem_type == "bounded_elastic_net",
                device=device,
            )
        raise ValueError(f"unsupported synthetic ERM problem type: {problem_type}")

    def problem_metadata(self, problem: MultinomialProblem | ElasticNetProblem) -> dict[str, Any]:
        problem_type = (
            "multinomial"
            if isinstance(problem, MultinomialProblem)
            else "bounded_elastic_net"
            if problem.bounded
            else "vanilla_elastic_net"
        )
        problem_id = (
            problem.spec.problem_id
            if isinstance(problem, MultinomialProblem)
            else problem.problem_id
        )
        return {
            "problem_generator": GENERATOR_VERSION,
            "problem_type": problem_type,
            "problem_id": problem_id,
            "matrix_representation": "materialized_dense",
            "dtype": str(problem.X.dtype),
        }

    def execute(
        self,
        problem: MultinomialProblem | ElasticNetProblem,
        command: dict[str, Any],
        backend: str,
    ) -> dict[str, Any]:
        expected_device = "cuda" if backend == "cuda" else "cpu"
        if problem.X.device.type != expected_device:
            raise ValueError(f"problem is on {problem.X.device.type}, but backend is {backend}")
        if isinstance(problem, MultinomialProblem):
            return self._execute_multinomial(problem, command)
        if problem.bounded:
            return self._execute_bounded_elastic_net(problem, command)
        return self._execute_vanilla_elastic_net(problem, command)

    def _execute_multinomial(
        self,
        problem: MultinomialProblem,
        command: dict[str, Any],
    ) -> dict[str, Any]:
        common = {
            "native_tolerance": command["native_tolerance"],
            "max_iterations": command["max_iters"],
        }
        solver_name = command["solver"]
        if solver_name == "rlaopt_sapphire":
            result = solve_multinomial_sapphire(
                problem,
                **common,
                batch_size=command["batch_size"],
                seed=command["solver_seed"],
            )
        elif solver_name in {"projected_gradient", "projected_gradient_no_jit"}:
            result = solve_jaxopt_projected_gradient(
                problem,
                **common,
                jit=solver_name == "projected_gradient",
            )
        elif solver_name in {"jaxopt_lbfgsb", "jaxopt_lbfgsb_no_jit"}:
            result = solve_jaxopt_lbfgsb(
                problem,
                **common,
                jit=solver_name == "jaxopt_lbfgsb",
            )
        else:
            raise ValueError(f"unknown multinomial solver: {solver_name}")
        return self._multinomial_outcome(problem, result, command)

    def _execute_vanilla_elastic_net(
        self,
        problem: ElasticNetProblem,
        command: dict[str, Any],
    ) -> dict[str, Any]:
        common = {
            "native_tolerance": command["native_tolerance"],
            "max_iterations": command["max_iters"],
        }
        solver_name = command["solver"]
        if solver_name == "rlaopt_sapphire":
            result = solve_elastic_net_sapphire(
                problem,
                **common,
                batch_size=command["batch_size"],
                seed=command["solver_seed"],
            )
        elif solver_name == "sklearn_coordinate_descent":
            result = solve_sklearn_coordinate_descent(problem, **common)
        elif solver_name == "cuml_coordinate_descent":
            result = solve_cuml_coordinate_descent(problem, **common)
        elif solver_name in {"jaxopt_proximal_gradient", "jaxopt_proximal_gradient_no_jit"}:
            result = solve_jaxopt_proximal_gradient(
                problem,
                **common,
                jit=solver_name == "jaxopt_proximal_gradient",
            )
        else:
            raise ValueError(f"unknown vanilla elastic-net solver: {solver_name}")
        return self._vanilla_elastic_net_outcome(problem, result, command)

    def _execute_bounded_elastic_net(
        self,
        problem: ElasticNetProblem,
        command: dict[str, Any],
    ) -> dict[str, Any]:
        common = {
            "native_tolerance": command["native_tolerance"],
            "max_iterations": command["max_iters"],
        }
        solver_name = command["solver"]
        if solver_name == "rlaopt_admm":
            result = solve_rlaopt_admm(
                problem,
                **common,
                batch_size=command["batch_size"],
                seed=command["solver_seed"],
            )
        elif solver_name == "scs":
            result = solve_scs(problem, **common)
        elif solver_name == "scs_cuda":
            result = solve_scs_cuda(problem, **common)
        elif solver_name == "scs_cpu_indirect":
            result = solve_scs_cpu_indirect(problem, **common)
        elif solver_name == "scs_cuda_direct":
            result = solve_scs_cuda_direct(problem, **common)
        elif solver_name in {"clarabel_qdldl", "cuclarabel_cudss"}:
            runtime = command.get("clarabel_runtime")
            if runtime is None:
                raise RuntimeError("Clarabel solver requires a preinitialized Julia runtime")
            adapter = (
                solve_clarabel_qdldl if solver_name == "clarabel_qdldl" else solve_cuclarabel_cudss
            )
            result = adapter(problem, runtime=runtime, **common)
        else:
            raise ValueError(f"unknown bounded elastic-net solver: {solver_name}")
        return self._bounded_elastic_net_outcome(problem, result, command)

    @staticmethod
    def _bounded_elastic_net_outcome(
        problem: ElasticNetProblem,
        result: BoundedElasticNetSolverResult,
        command: dict[str, Any],
    ) -> dict[str, Any]:
        stationarity = float(
            problem.kkt_residual(
                result.weights,
                result.intercept,
                activity_tolerance=command["feasibility_tolerance"],
            )
        )
        feasibility = float(problem.constraint_violation(result.weights))
        native_success = result.native_status == "converged"
        return {
            "runtime_seconds": result.runtime_seconds,
            "iterations": result.iterations,
            "native_status": result.native_status,
            "native_success": native_success,
            "runtime_eligible": native_success,
            "trace": [],
            "solver_metadata": result.metadata
            | {
                "native_tolerance": command["native_tolerance"],
                "native_error": result.native_error,
            },
            "accuracy": {
                "stationarity": stationarity,
                "feasibility": feasibility,
                "objective": float(problem.objective(result.weights, result.intercept)),
                "external_success": (
                    stationarity <= command["stationarity_tolerance"]
                    and feasibility <= command["feasibility_tolerance"]
                ),
            },
            "diagnostics": problem.diagnostics(),
        }

    @staticmethod
    def _vanilla_elastic_net_outcome(
        problem: ElasticNetProblem,
        result: ElasticNetSolverResult,
        command: dict[str, Any],
    ) -> dict[str, Any]:
        relative_duality_gap = float(problem.relative_duality_gap(result.weights, result.intercept))
        stationarity = float(
            problem.kkt_residual(
                result.weights,
                result.intercept,
                activity_tolerance=command["feasibility_tolerance"],
            )
        )
        feasibility = float(problem.constraint_violation(result.weights))
        native_success = result.native_status == "converged"
        return {
            "runtime_seconds": result.runtime_seconds,
            "iterations": result.iterations,
            "native_status": result.native_status,
            "native_success": native_success,
            "runtime_eligible": native_success,
            "trace": [],
            "solver_metadata": result.metadata
            | {
                "native_tolerance": command["native_tolerance"],
                "native_error": result.native_error,
            },
            "accuracy": {
                "relative_duality_gap": relative_duality_gap,
                "stationarity": stationarity,
                "feasibility": feasibility,
                "objective": float(problem.objective(result.weights, result.intercept)),
                "external_success": (
                    relative_duality_gap <= command["relative_duality_gap_tolerance"]
                ),
            },
            "diagnostics": problem.diagnostics(),
        }

    @staticmethod
    def _multinomial_outcome(
        problem: MultinomialProblem,
        result: MultinomialSolverResult,
        command: dict[str, Any],
    ) -> dict[str, Any]:
        stationarity = float(
            problem.kkt_residual(
                result.coefficients,
                activity_tolerance=command["feasibility_tolerance"],
            )
        )
        feasibility = float(problem.constraint_violation(result.coefficients))
        objective = float(problem.objective(result.coefficients))
        external_success = (
            stationarity <= command["stationarity_tolerance"]
            and feasibility <= command["feasibility_tolerance"]
        )
        native_success = result.native_status == "converged"
        return {
            "runtime_seconds": result.runtime_seconds,
            "iterations": result.iterations,
            "native_status": result.native_status,
            "native_success": native_success,
            "runtime_eligible": native_success,
            "trace": [],
            "solver_metadata": result.metadata
            | {
                "native_tolerance": command["native_tolerance"],
                "native_error": result.native_error,
            },
            "accuracy": {
                "stationarity": stationarity,
                "feasibility": feasibility,
                "objective": objective,
                "external_success": external_success,
            },
            "diagnostics": problem.diagnostics(),
        }

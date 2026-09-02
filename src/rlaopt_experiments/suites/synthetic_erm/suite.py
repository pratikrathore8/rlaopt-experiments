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
        if problem_type == "vanilla_elastic_net":
            spec = ElasticNetSpec(**specification["problem_spec"])
            return generate_elastic_net_problem(spec, bounded=False, device=device)
        raise ValueError(f"unsupported synthetic ERM problem type: {problem_type}")

    def problem_metadata(self, problem: MultinomialProblem | ElasticNetProblem) -> dict[str, Any]:
        problem_type = (
            "multinomial" if isinstance(problem, MultinomialProblem) else "vanilla_elastic_net"
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
        elif solver_name == "projected_gradient":
            result = solve_jaxopt_projected_gradient(problem, **common)
        elif solver_name == "jaxopt_lbfgsb":
            result = solve_jaxopt_lbfgsb(problem, **common)
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
        elif solver_name == "jaxopt_proximal_gradient":
            result = solve_jaxopt_proximal_gradient(problem, **common)
        else:
            raise ValueError(f"unknown vanilla elastic-net solver: {solver_name}")
        return self._vanilla_elastic_net_outcome(problem, result, command)

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

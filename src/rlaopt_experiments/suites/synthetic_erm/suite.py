"""Synthetic ERM suite adapter for isolated benchmark workers."""

from __future__ import annotations

from typing import Any

import torch

from rlaopt_experiments.problems.synthetic_erm import (
    GENERATOR_VERSION,
    MultinomialProblem,
    MultinomialSpec,
    generate_multinomial_problem,
)
from rlaopt_experiments.suites.synthetic_erm.multinomial_solvers import (
    MultinomialSolverResult,
    solve_jaxopt_lbfgsb,
    solve_jaxopt_projected_gradient,
    solve_rlaopt_sapphire,
)


class SyntheticErmSuite:
    """Generate and execute synthetic ERM problems behind the suite protocol."""

    name = "synthetic_erm"

    def generate(
        self,
        specification: dict[str, Any],
        device: torch.device,
    ) -> MultinomialProblem:
        problem_type = specification.get("problem_type")
        if problem_type != "multinomial":
            raise ValueError(f"unsupported synthetic ERM problem type: {problem_type}")
        spec = MultinomialSpec(**specification["problem_spec"])
        return generate_multinomial_problem(spec, device=device)

    def problem_metadata(self, problem: MultinomialProblem) -> dict[str, Any]:
        return {
            "problem_generator": GENERATOR_VERSION,
            "problem_type": "multinomial",
            "problem_id": problem.spec.problem_id,
            "matrix_representation": "materialized_dense",
            "dtype": str(problem.X.dtype),
        }

    def execute(
        self,
        problem: MultinomialProblem,
        command: dict[str, Any],
        backend: str,
    ) -> dict[str, Any]:
        expected_device = "cuda" if backend == "cuda" else "cpu"
        if problem.X.device.type != expected_device:
            raise ValueError(f"problem is on {problem.X.device.type}, but backend is {backend}")

        common = {
            "native_tolerance": command["native_tolerance"],
            "max_iterations": command["max_iters"],
        }
        solver_name = command["solver"]
        if solver_name == "rlaopt_sapphire":
            result = solve_rlaopt_sapphire(
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

"""Synthetic ridge suite adapter for the suite-neutral worker."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from rlaopt_experiments.diagnostics import adjudicate
from rlaopt_experiments.problem import (
    GENERATOR_VERSION,
    ProblemSpec,
    RidgeProblem,
    generate_problem,
)
from rlaopt_experiments.solvers import solve


class RidgeSuite:
    """Preserve the original synthetic-ridge implementation behind a suite API."""

    name = "synthetic_ridge"

    def generate(self, specification: dict[str, Any], device: torch.device) -> RidgeProblem:
        return generate_problem(ProblemSpec(**specification), device=device)

    def problem_metadata(self, problem: RidgeProblem) -> dict[str, Any]:
        del problem
        return {
            "problem_generator": GENERATOR_VERSION,
            "matrix_representation": "materialized_dense",
        }

    def execute(
        self,
        problem: RidgeProblem,
        command: dict[str, Any],
        backend: str,
    ) -> dict[str, Any]:
        torch.manual_seed(command["nystrom_seed"])
        if backend == "cuda":
            torch.cuda.manual_seed_all(command["nystrom_seed"])
        ridge = command["ridge"]
        result = solve(
            command["solver"],
            problem,
            ridge,
            command["native_tolerance"],
            command["max_iters"],
            command["timeout_seconds"],
            command["rank"],
        )
        accuracy = adjudicate(problem, ridge, result, command["kkt_tolerance"])
        accuracy_values = asdict(accuracy)
        external_success = accuracy_values.pop("success")
        native_success = result.native_status in {
            "direct",
            "istop_1",
            "istop_2",
            "native_complete",
            "native_converged",
        }
        return {
            "runtime_seconds": result.runtime_seconds,
            "iterations": result.iterations,
            "native_status": result.native_status,
            "native_success": native_success,
            "runtime_eligible": native_success,
            "trace": result.trace,
            "solver_metadata": result.metadata | {"native_tolerance": command["native_tolerance"]},
            "accuracy": accuracy_values | {"external_success": external_success},
            "diagnostics": problem.diagnostics(ridge),
        }

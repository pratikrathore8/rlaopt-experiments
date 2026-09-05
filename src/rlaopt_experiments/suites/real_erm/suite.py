"""Real-data ERM suite adapter for isolated benchmark workers."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from rlaopt_experiments.problems.real_erm import (
    PROBLEM_VERSION,
    RealElasticNetSpec,
    RealMultinomialSpec,
    build_real_elastic_net_problem,
    build_real_multinomial_problem,
)
from rlaopt_experiments.problems.synthetic_erm import ElasticNetProblem, MultinomialProblem
from rlaopt_experiments.suites.synthetic_erm.suite import SyntheticErmSuite


class RealErmSuite(SyntheticErmSuite):
    """Load real problems while reusing the common ERM solver and metric adapters."""

    name = "real_erm"

    def generate(
        self,
        specification: dict[str, Any],
        device: torch.device,
    ) -> MultinomialProblem | ElasticNetProblem:
        problem_type = specification.get("problem_type")
        if problem_type == "multinomial":
            spec = RealMultinomialSpec(**specification["problem_spec"])
            return build_real_multinomial_problem(spec, device=device)
        if problem_type in {"vanilla_elastic_net", "bounded_elastic_net"}:
            spec = RealElasticNetSpec(**specification["problem_spec"])
            return build_real_elastic_net_problem(
                spec,
                bounded=problem_type == "bounded_elastic_net",
                device=device,
            )
        raise ValueError(f"unsupported real ERM problem type: {problem_type}")

    def problem_metadata(self, problem: MultinomialProblem | ElasticNetProblem) -> dict[str, Any]:
        problem_type = (
            "multinomial"
            if isinstance(problem, MultinomialProblem)
            else "bounded_elastic_net"
            if problem.bounded
            else "vanilla_elastic_net"
        )
        dataset = problem.spec.dataset_spec
        problem_id = (
            problem.spec.problem_id
            if isinstance(problem, MultinomialProblem)
            else problem.problem_id
        )
        return {
            "problem_generator": PROBLEM_VERSION,
            "problem_type": problem_type,
            "problem_id": problem_id,
            "dataset": dataset.name,
            "source_sha256": dataset.sha256,
            "base_shape": list(dataset.base_shape),
            "solver_shape": list(dataset.solver_shape),
            "random_features": (
                asdict(dataset.random_features) if dataset.random_features is not None else None
            ),
            "matrix_representation": "materialized_dense",
            "dtype": str(problem.X.dtype),
        }

from __future__ import annotations

import pytest
import torch

from rlaopt_experiments.suites import get_suite
from rlaopt_experiments.suites.synthetic_erm.bounded_elastic_net_solvers import (
    BoundedElasticNetSolverResult,
)


def _multinomial_specification(n: int = 64, p: int = 8) -> dict:
    return {
        "problem_type": "multinomial",
        "problem_spec": {
            "n": n,
            "p": p,
            "n_classes": 3,
            "feature_seed": 11,
            "target_seed": 12,
            "teacher_scale": 1.0,
            "box_lower": -1.0,
            "box_upper": 1.0,
        },
    }


def _vanilla_elastic_net_specification() -> dict:
    return {
        "problem_type": "vanilla_elastic_net",
        "problem_spec": {
            "n": 96,
            "p": 12,
            "feature_seed": 21,
            "target_seed": 22,
            "teacher_density": 0.25,
            "noise_ratio": 0.1,
            "teacher_intercept": 0.5,
            "regularization_fraction": 0.1,
        },
    }


def _bounded_elastic_net_specification() -> dict:
    specification = _vanilla_elastic_net_specification()
    return specification | {"problem_type": "bounded_elastic_net"}


def test_multinomial_suite_executes_and_adjudicates_jaxopt_lbfgsb() -> None:
    suite = get_suite("synthetic_erm")
    problem = suite.generate(
        _multinomial_specification(),
        device=torch.device("cpu"),
    )

    outcome = suite.execute(
        problem,
        {
            "solver": "jaxopt_lbfgsb",
            "native_tolerance": 1e-7,
            "max_iters": 100,
            "stationarity_tolerance": 1e-6,
            "feasibility_tolerance": 1e-8,
        },
        backend="cpu",
    )

    assert outcome["native_status"] == "converged"
    assert outcome["native_success"]
    assert outcome["runtime_eligible"]
    assert outcome["accuracy"]["external_success"]
    assert outcome["accuracy"]["stationarity"] <= 1e-6
    assert outcome["accuracy"]["feasibility"] <= 1e-8
    assert outcome["solver_metadata"]["native_tolerance"] == 1e-7
    assert outcome["solver_metadata"]["native_error"] <= 1e-7
    assert outcome["diagnostics"]["class_counts"]


def test_vanilla_elastic_net_suite_executes_and_uses_external_gap() -> None:
    suite = get_suite("synthetic_erm")
    problem = suite.generate(
        _vanilla_elastic_net_specification(),
        device=torch.device("cpu"),
    )

    outcome = suite.execute(
        problem,
        {
            "solver": "sklearn_coordinate_descent",
            "native_tolerance": 1e-10,
            "max_iters": 10_000,
            "relative_duality_gap_tolerance": 1e-6,
            "feasibility_tolerance": 1e-8,
        },
        backend="cpu",
    )

    assert outcome["native_success"]
    assert outcome["runtime_eligible"]
    assert outcome["accuracy"]["external_success"]
    assert outcome["accuracy"]["relative_duality_gap"] <= 1e-6
    assert outcome["accuracy"]["feasibility"] == 0.0
    assert outcome["accuracy"]["stationarity"] <= 1e-6
    assert outcome["diagnostics"]["bounded"] is False


def test_bounded_elastic_net_suite_executes_and_uses_external_kkt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = get_suite("synthetic_erm")
    problem = suite.generate(
        _bounded_elastic_net_specification(),
        device=torch.device("cpu"),
    )
    result = BoundedElasticNetSolverResult(
        weights=torch.zeros(problem.spec.p, dtype=torch.float64),
        intercept=problem.y.mean(),
        runtime_seconds=0.25,
        iterations=4,
        native_status="converged",
        native_error=None,
        metadata={"linear_solver": "fake"},
    )
    monkeypatch.setattr(
        "rlaopt_experiments.suites.synthetic_erm.suite.solve_scs",
        lambda *_args, **_kwargs: result,
    )

    outcome = suite.execute(
        problem,
        {
            "solver": "scs",
            "native_tolerance": 1e-6,
            "max_iters": 100,
            "stationarity_tolerance": 1e6,
            "feasibility_tolerance": 1e-8,
        },
        backend="cpu",
    )

    assert outcome["native_success"]
    assert outcome["runtime_eligible"]
    assert outcome["accuracy"]["external_success"]
    assert outcome["accuracy"]["feasibility"] == 0.0
    assert outcome["accuracy"]["stationarity"] <= 1e6
    assert outcome["diagnostics"]["bounded"] is True
    assert outcome["solver_metadata"]["native_error"] is None


def test_multinomial_suite_rejects_unknown_solver() -> None:
    suite = get_suite("synthetic_erm")
    problem = suite.generate(
        _multinomial_specification(n=8, p=2),
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="unknown multinomial solver: unknown"):
        suite.execute(
            problem,
            {
                "solver": "unknown",
                "native_tolerance": 1e-6,
                "max_iters": 10,
            },
            backend="cpu",
        )

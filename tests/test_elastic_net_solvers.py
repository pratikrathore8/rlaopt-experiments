from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rlaopt_experiments.problems.synthetic_erm import (
    ElasticNetSpec,
    generate_elastic_net_problem,
)
from rlaopt_experiments.suites.synthetic_erm.elastic_net_solvers import (
    _build_rlaopt_objective,
    solve_cuml_coordinate_descent,
    solve_jaxopt_proximal_gradient,
    solve_rlaopt_sapphire,
    solve_sklearn_coordinate_descent,
)


@pytest.fixture(scope="module")
def problem():
    return generate_elastic_net_problem(
        ElasticNetSpec(
            n=96,
            p=12,
            feature_seed=21,
            target_seed=22,
            teacher_density=0.25,
            noise_ratio=0.1,
            teacher_intercept=0.5,
            regularization_fraction=0.1,
        ),
        bounded=False,
        device="cpu",
    )


def test_sklearn_and_jaxopt_match_the_canonical_problem(problem) -> None:
    sklearn_result = solve_sklearn_coordinate_descent(
        problem,
        native_tolerance=1e-10,
        max_iterations=10_000,
    )
    jaxopt_result = solve_jaxopt_proximal_gradient(
        problem,
        native_tolerance=1e-8,
        max_iterations=2_000,
    )

    for result in (sklearn_result, jaxopt_result):
        assert result.weights.shape == (problem.spec.p,)
        assert result.weights.dtype == torch.float64
        assert result.intercept.shape == ()
        assert result.intercept.dtype == torch.float64
        assert result.native_status == "converged"
        assert result.runtime_seconds > 0
        assert problem.relative_duality_gap(result.weights, result.intercept) <= 1e-6

    torch.testing.assert_close(
        jaxopt_result.weights,
        sklearn_result.weights,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        jaxopt_result.intercept,
        sklearn_result.intercept,
        rtol=2e-5,
        atol=2e-6,
    )
    assert sklearn_result.metadata["alpha"] == pytest.approx(problem.lambda_l1 + problem.lambda_l2)
    assert sklearn_result.metadata["l1_ratio"] == pytest.approx(0.5)
    assert jaxopt_result.metadata["acceleration"] is True
    assert jaxopt_result.metadata["line_search"] == "backtracking"


def test_sapphire_uses_default_float64_nystrom_path(problem) -> None:
    previous_default_dtype = torch.get_default_dtype()
    result = solve_rlaopt_sapphire(
        problem,
        native_tolerance=1e-5,
        max_iterations=50,
        batch_size=24,
        seed=23,
    )

    assert torch.get_default_dtype() == previous_default_dtype
    assert result.weights.shape == (problem.spec.p,)
    assert result.weights.dtype == torch.float64
    assert result.intercept.shape == ()
    assert result.intercept.dtype == torch.float64
    assert result.iterations <= 50
    assert result.runtime_seconds > 0
    assert result.metadata["base_method"] == "saga"
    assert result.metadata["native_objective_scale"] == 2.0
    assert result.metadata["nystrom_rank"] == 10
    assert result.metadata["torch_default_dtype_workaround"] is True
    initial_weights = torch.zeros_like(result.weights)
    initial_intercept = torch.zeros_like(result.intercept)
    assert problem.objective(result.weights, result.intercept) < problem.objective(
        initial_weights,
        initial_intercept,
    )


def test_rlaopt_native_objective_is_exactly_twice_canonical(problem) -> None:
    from rlaopt.data import DataLoader, Dataset

    dataset = Dataset(problem.X, problem.y, device=problem.X.device, dtype=torch.float64)
    loader = DataLoader(dataset, batch_size=problem.spec.n, shuffle=False)
    native_objective, weights, intercept = _build_rlaopt_objective(problem, loader)
    candidate_weights = torch.linspace(-0.4, 0.7, problem.spec.p, dtype=torch.float64)
    candidate_intercept = torch.tensor(-0.25, dtype=torch.float64)
    weights.value = candidate_weights
    intercept.value = candidate_intercept

    torch.testing.assert_close(
        native_objective.forward(),
        2.0 * problem.objective(candidate_weights, candidate_intercept),
        rtol=1e-14,
        atol=1e-14,
    )


def test_vanilla_adapters_reject_bounded_problems(problem) -> None:
    bounded = replace(problem, bounded=True)
    with pytest.raises(ValueError, match="unbounded"):
        solve_jaxopt_proximal_gradient(
            bounded,
            native_tolerance=1e-6,
            max_iterations=10,
        )


def test_backend_specific_adapters_reject_hidden_transfers(problem) -> None:
    with pytest.raises(ValueError, match="cuda"):
        solve_cuml_coordinate_descent(
            problem,
            native_tolerance=1e-6,
            max_iterations=10,
        )


def test_adapters_reject_invalid_controls(problem) -> None:
    with pytest.raises(ValueError, match="native_tolerance"):
        solve_sklearn_coordinate_descent(
            problem,
            native_tolerance=float("nan"),
            max_iterations=10,
        )
    with pytest.raises(ValueError, match="batch_size"):
        solve_rlaopt_sapphire(
            problem,
            native_tolerance=1e-5,
            max_iterations=10,
            batch_size=0,
            seed=23,
        )

from __future__ import annotations

import pytest
import torch

from rlaopt_experiments.problems.synthetic_erm import (
    MultinomialSpec,
    generate_multinomial_problem,
)
from rlaopt_experiments.suites.synthetic_erm.multinomial_solvers import (
    _build_rlaopt_objective,
    _jax_problem,
    solve_jaxopt_lbfgsb,
    solve_jaxopt_projected_gradient,
    solve_rlaopt_sapphire,
)


@pytest.fixture(scope="module")
def problem():
    return generate_multinomial_problem(
        MultinomialSpec(
            n=64,
            p=8,
            n_classes=3,
            feature_seed=11,
            target_seed=12,
            teacher_scale=1.0,
            box_lower=-1.0,
            box_upper=1.0,
        ),
        device="cpu",
    )


def test_jaxopt_multinomial_data_are_not_compilation_constants(problem) -> None:
    import jax

    objective, initial, _, features, labels = _jax_problem(problem)
    traced = jax.make_jaxpr(objective)(initial, features, labels)

    assert not traced.consts


@pytest.mark.parametrize(
    "adapter",
    [solve_jaxopt_projected_gradient, solve_jaxopt_lbfgsb],
)
def test_jaxopt_multinomial_adapters_match_canonical_problem(problem, adapter) -> None:
    result = adapter(problem, native_tolerance=1e-7, max_iterations=500)

    assert result.coefficients.dtype == torch.float64
    assert result.coefficients.device == problem.X.device
    assert result.native_status == "converged"
    assert result.iterations <= 500
    assert result.runtime_seconds > 0
    assert result.metadata["data_arguments"] == "dynamic"
    assert result.metadata["first_jit_compilation_included"] is True
    assert result.metadata["jit_enabled"] is True
    assert problem.constraint_violation(result.coefficients) <= 1e-12
    assert problem.kkt_residual(result.coefficients, activity_tolerance=1e-8) <= 1e-6
    initial = torch.zeros_like(result.coefficients)
    assert problem.objective(result.coefficients) < problem.objective(initial)


def test_sapphire_adapter_uses_float64_nystrom_path(problem) -> None:
    previous_default_dtype = torch.get_default_dtype()
    result = solve_rlaopt_sapphire(
        problem,
        native_tolerance=1e-5,
        max_iterations=40,
        batch_size=16,
        seed=13,
    )

    assert torch.get_default_dtype() == previous_default_dtype
    assert result.coefficients.dtype == torch.float64
    assert result.coefficients.device == problem.X.device
    assert result.iterations <= 40
    assert result.runtime_seconds > 0
    assert result.metadata["base_method"] == "saga"
    assert result.metadata["nystrom_rank"] == 10
    assert result.metadata["torch_default_dtype_workaround"] is True
    assert result.metadata["variable_autograd_tracking"] is False
    assert problem.constraint_violation(result.coefficients) == 0
    initial = torch.zeros_like(result.coefficients)
    assert problem.objective(result.coefficients) < problem.objective(initial)


def test_sapphire_saga_state_does_not_retain_autograd_graph(problem) -> None:
    from rlaopt.data import DataLoader, Dataset
    from rlaopt.solvers import Sapphire, SapphireConfig

    loader = DataLoader(
        Dataset(problem.X, problem.y, device=problem.X.device, dtype=torch.float64),
        batch_size=16,
        shuffle=False,
    )
    previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        objective, coefficients = _build_rlaopt_objective(problem, loader)
        solver = Sapphire(objective, SapphireConfig())
        variable_values = solver._op_split.variable_values
        state = solver.init_state(variable_values)
        _, state = solver.step(variable_values, state)
    finally:
        torch.set_default_dtype(previous_default_dtype)

    assert coefficients.value.requires_grad is False
    assert state.table.requires_grad is False
    assert state.table.grad_fn is None
    assert all(value.requires_grad is False for value in state.grad_avg.values())
    assert all(value.grad_fn is None for value in state.grad_avg.values())


def test_multinomial_adapters_reject_nonpositive_controls(problem) -> None:
    with pytest.raises(ValueError, match="native_tolerance"):
        solve_jaxopt_projected_gradient(
            problem,
            native_tolerance=0.0,
            max_iterations=10,
        )
    with pytest.raises(ValueError, match="batch_size"):
        solve_rlaopt_sapphire(
            problem,
            native_tolerance=1e-5,
            max_iterations=10,
            batch_size=0,
            seed=13,
        )

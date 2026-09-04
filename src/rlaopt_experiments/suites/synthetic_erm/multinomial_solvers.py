"""Direct solver adapters for box-constrained multinomial regression."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from rlaopt_experiments.problems.synthetic_erm import MultinomialProblem


@dataclass(frozen=True)
class MultinomialSolverResult:
    """Common result returned by every multinomial solver adapter."""

    coefficients: torch.Tensor
    runtime_seconds: float
    iterations: int
    native_status: str
    native_error: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_inputs(
    problem: MultinomialProblem,
    native_tolerance: float,
    max_iterations: int,
) -> None:
    if problem.X.dtype != torch.float64:
        raise ValueError("multinomial adapters require float64 problem data")
    if not 0 < native_tolerance:
        raise ValueError("native_tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_rlaopt_objective(problem: MultinomialProblem, loader: Any) -> tuple[Any, Any]:
    """Build the rlaopt objective without retaining higher-order autograd graphs."""
    from rlaopt.atoms import Box, MultinomialRegression
    from rlaopt.expression import Variable

    coefficients = Variable(
        (problem.spec.p, problem.spec.n_classes),
        requires_grad=False,
        name="beta",
        dtype=torch.float64,
        device=problem.X.device,
    )
    model = MultinomialRegression(coefficients, loader, fit_intercept=False)
    objective = model + Box(
        coefficients,
        lower=problem.spec.box_lower,
        upper=problem.spec.box_upper,
    )
    return objective, coefficients


def solve_rlaopt_sapphire(
    problem: MultinomialProblem,
    *,
    native_tolerance: float,
    max_iterations: int,
    batch_size: int,
    seed: int,
) -> MultinomialSolverResult:
    """Solve with the default rlaopt SAPPHIRE algorithm configuration."""
    _validate_inputs(problem, native_tolerance, max_iterations)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    from rlaopt.data import DataLoader, Dataset
    from rlaopt.solvers import (
        GradSolverStoppingCriteria,
        Sapphire,
        SapphireConfig,
    )

    device = problem.X.device
    effective_batch_size = min(batch_size, problem.spec.n)
    previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        loader_generator = torch.Generator(device="cpu").manual_seed(seed)
        dataset = Dataset(problem.X, problem.y, device=device, dtype=torch.float64)
        loader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        objective, _ = _build_rlaopt_objective(problem, loader)
        config = SapphireConfig()
        solver = Sapphire(objective, config)
        stopping = GradSolverStoppingCriteria(
            max_iters=max_iterations,
            eps_abs=native_tolerance,
            eps_rel=native_tolerance,
        )
        _synchronize(device)
        started = time.perf_counter()
        native_result = solver.solve(stopping_criteria=stopping)
        _synchronize(device)
        runtime_seconds = time.perf_counter() - started
        solution = native_result.variable_values["beta"].detach().clone()
    finally:
        torch.set_default_dtype(previous_default_dtype)

    return MultinomialSolverResult(
        coefficients=solution,
        runtime_seconds=runtime_seconds,
        iterations=native_result.num_iters,
        native_status=native_result.convergence_status.value,
        native_error=float(native_result.err),
        metadata={
            "base_method": config.base_method,
            "batch_size": effective_batch_size,
            "variable_autograd_tracking": False,
            "nystrom_rank": config.precond_config.rank_init,
            "native_solver_time_seconds": native_result.solver_time,
            "torch_default_dtype_workaround": True,
        },
    )


def _jax_problem(problem: MultinomialProblem) -> tuple[Any, Any, Any, Any, Any]:
    """Create device-sharing JAX views and the canonical objective."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    jax.config.update("jax_enable_x64", True)
    if problem.X.device.type == "cpu":
        jax.config.update("jax_platforms", "cpu")
    elif problem.X.device.type == "cuda":
        jax.config.update("jax_platforms", "cuda")
    else:
        raise ValueError(f"unsupported device for JAXopt: {problem.X.device}")

    import jax.numpy as jnp
    from jax import dlpack as jax_dlpack

    features = jax_dlpack.from_dlpack(problem.X.detach())
    labels = jax_dlpack.from_dlpack(problem.y.detach())

    def objective(coefficients: Any, data: Any, targets: Any) -> Any:
        logits = data @ coefficients
        losses = jax.nn.logsumexp(logits, axis=1) - logits[jnp.arange(targets.shape[0]), targets]
        return jnp.mean(losses)

    initial = jnp.zeros(
        (problem.spec.p, problem.spec.n_classes),
        dtype=jnp.float64,
    )
    bounds = (
        jnp.full_like(initial, problem.spec.box_lower),
        jnp.full_like(initial, problem.spec.box_upper),
    )
    return objective, initial, bounds, features, labels


def _torch_from_jax(array: Any, device: torch.device) -> torch.Tensor:
    solution = torch.utils.dlpack.from_dlpack(array).detach().clone()
    if solution.dtype != torch.float64 or solution.device != device:
        raise RuntimeError(
            f"JAXopt returned dtype={solution.dtype}, device={solution.device}; "
            f"expected float64 on {device}"
        )
    return solution


def solve_jaxopt_projected_gradient(
    problem: MultinomialProblem,
    *,
    native_tolerance: float,
    max_iterations: int,
) -> MultinomialSolverResult:
    """Solve with JAXopt default projected-gradient configuration."""
    _validate_inputs(problem, native_tolerance, max_iterations)
    objective, initial, bounds, features, labels = _jax_problem(problem)
    import jaxopt

    solver = jaxopt.ProjectedGradient(
        fun=objective,
        projection=jaxopt.projection.projection_box,
        maxiter=max_iterations,
        tol=native_tolerance,
    )
    started = time.perf_counter()
    step = solver.run(initial, bounds, features, labels)
    step.params.block_until_ready()
    runtime_seconds = time.perf_counter() - started
    iterations = int(step.state.iter_num)
    native_error = float(step.state.error)
    native_status = "converged" if native_error <= native_tolerance else "max_iterations"
    return MultinomialSolverResult(
        coefficients=_torch_from_jax(step.params, problem.X.device),
        runtime_seconds=runtime_seconds,
        iterations=iterations,
        native_status=native_status,
        native_error=native_error,
        metadata={
            "acceleration": solver.acceleration,
            "data_arguments": "dynamic",
            "first_jit_compilation_included": True,
            "jit_enabled": solver.jit,
            "line_search": "backtracking",
        },
    )


def solve_jaxopt_lbfgsb(
    problem: MultinomialProblem,
    *,
    native_tolerance: float,
    max_iterations: int,
) -> MultinomialSolverResult:
    """Solve with JAXopt L-BFGS-B and its zoom line search."""
    _validate_inputs(problem, native_tolerance, max_iterations)
    objective, initial, bounds, features, labels = _jax_problem(problem)
    import jaxopt

    solver = jaxopt.LBFGSB(
        fun=objective,
        maxiter=max_iterations,
        tol=native_tolerance,
    )
    started = time.perf_counter()
    step = solver.run(initial, bounds, features, labels)
    step.params.block_until_ready()
    runtime_seconds = time.perf_counter() - started
    iterations = int(step.state.iter_num)
    native_error = float(step.state.error)
    line_search_failed = bool(step.state.failed_linesearch)
    if line_search_failed:
        native_status = "line_search_failed"
    elif native_error <= native_tolerance:
        native_status = "converged"
    else:
        native_status = "max_iterations"
    return MultinomialSolverResult(
        coefficients=_torch_from_jax(step.params, problem.X.device),
        runtime_seconds=runtime_seconds,
        iterations=iterations,
        native_status=native_status,
        native_error=native_error,
        metadata={
            "history_size": solver.history_size,
            "data_arguments": "dynamic",
            "first_jit_compilation_included": True,
            "jit_enabled": solver.jit,
            "line_search": solver.linesearch,
            "line_search_failed": line_search_failed,
            "num_function_evaluations": int(step.state.num_fun_eval),
            "num_gradient_evaluations": int(step.state.num_grad_eval),
        },
    )

"""Direct solver adapters for unconstrained elastic-net regression."""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from rlaopt_experiments.problems.synthetic_erm import ElasticNetProblem


@dataclass(frozen=True)
class ElasticNetSolverResult:
    """Common result returned by every vanilla elastic-net adapter."""

    weights: torch.Tensor
    intercept: torch.Tensor
    runtime_seconds: float
    iterations: int
    native_status: str
    native_error: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_inputs(
    problem: ElasticNetProblem,
    native_tolerance: float,
    max_iterations: int,
) -> None:
    if problem.bounded:
        raise ValueError("vanilla elastic-net adapters require an unbounded problem")
    if problem.X.dtype != torch.float64 or problem.y.dtype != torch.float64:
        raise ValueError("elastic-net adapters require float64 problem data")
    if not math.isfinite(native_tolerance) or native_tolerance <= 0:
        raise ValueError("native_tolerance must be finite and positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")


def _require_device(problem: ElasticNetProblem, expected: str, solver: str) -> None:
    if problem.X.device.type != expected:
        raise ValueError(f"{solver} requires a {expected} problem")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _penalty_parameters(problem: ElasticNetProblem) -> tuple[float, float]:
    alpha = problem.lambda_l1 + problem.lambda_l2
    return alpha, problem.lambda_l1 / alpha


def _build_rlaopt_objective(problem: ElasticNetProblem, loader: Any) -> tuple[Any, Any, Any]:
    """Build an rlaopt 0.1.0 expression equal to twice the canonical objective."""
    from rlaopt.atoms import ElasticNet, LinearRegression
    from rlaopt.expression import Variable

    weights = Variable(
        (problem.spec.p,),
        name="beta",
        dtype=torch.float64,
        device=problem.X.device,
    )
    model = LinearRegression(weights, loader, fit_intercept=True)
    # LinearRegression uses MSE rather than 0.5*MSE. Doubling both penalties
    # makes the complete native objective exactly 2 * the canonical objective.
    regularizer = ElasticNet(
        weights,
        l1_scaling=torch.tensor(
            2.0 * problem.lambda_l1, dtype=torch.float64, device=problem.X.device
        ),
        l2_scaling=torch.tensor(
            2.0 * problem.lambda_l2, dtype=torch.float64, device=problem.X.device
        ),
    )
    return model + regularizer, weights, model.get_input("intercept")


def solve_rlaopt_sapphire(
    problem: ElasticNetProblem,
    *,
    native_tolerance: float,
    max_iterations: int,
    batch_size: int,
    seed: int,
) -> ElasticNetSolverResult:
    """Solve with rlaopt SAPPHIRE and its default algorithm configuration."""
    _validate_inputs(problem, native_tolerance, max_iterations)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    from rlaopt.data import DataLoader, Dataset
    from rlaopt.solvers import GradSolverStoppingCriteria, Sapphire, SapphireConfig

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
        objective, _, _ = _build_rlaopt_objective(problem, loader)
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
        solution_weights = native_result.variable_values["beta"].detach().clone()
        solution_intercept = native_result.variable_values["intercept"].detach().clone().reshape(())
    finally:
        torch.set_default_dtype(previous_default_dtype)

    return ElasticNetSolverResult(
        weights=solution_weights,
        intercept=solution_intercept,
        runtime_seconds=runtime_seconds,
        iterations=native_result.num_iters,
        native_status=native_result.convergence_status.value,
        native_error=float(native_result.err),
        metadata={
            "base_method": config.base_method,
            "batch_size": effective_batch_size,
            "native_objective_scale": 2.0,
            "nystrom_rank": config.precond_config.rank_init,
            "native_solver_time_seconds": native_result.solver_time,
            "torch_default_dtype_workaround": True,
        },
    )


def solve_sklearn_coordinate_descent(
    problem: ElasticNetProblem,
    *,
    native_tolerance: float,
    max_iterations: int,
) -> ElasticNetSolverResult:
    """Solve with scikit-learn's cyclic coordinate-descent ElasticNet."""
    _validate_inputs(problem, native_tolerance, max_iterations)
    _require_device(problem, "cpu", "scikit-learn ElasticNet")

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNet

    alpha, l1_ratio = _penalty_parameters(problem)
    # Keep solver input conversion outside the measured native fit call.
    features = np.asfortranarray(problem.X.detach().numpy())
    targets = np.asarray(problem.y.detach().numpy())
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=max_iterations,
        tol=native_tolerance,
        copy_X=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        started = time.perf_counter()
        model.fit(features, targets)
        runtime_seconds = time.perf_counter() - started
    convergence_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    iterations = int(model.n_iter_)
    native_status = "max_iterations" if convergence_warning else "converged"
    return ElasticNetSolverResult(
        weights=torch.from_numpy(np.asarray(model.coef_)).detach().clone(),
        intercept=torch.as_tensor(model.intercept_, dtype=torch.float64).reshape(()),
        runtime_seconds=runtime_seconds,
        iterations=iterations,
        native_status=native_status,
        native_error=float(abs(model.dual_gap_)),
        metadata={
            "alpha": alpha,
            "copy_X": False,
            "input_order": "F",
            "l1_ratio": l1_ratio,
            "selection": model.selection,
        },
    )


def _jax_problem(problem: ElasticNetProblem) -> tuple[Any, Any, Any]:
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
    targets = jax_dlpack.from_dlpack(problem.y.detach())

    def smooth_objective(params: dict[str, Any]) -> Any:
        residual = features @ params["weights"] + params["intercept"] - targets
        return 0.5 * jnp.mean(residual**2) + 0.5 * problem.lambda_l2 * jnp.sum(
            params["weights"] ** 2
        )

    initial = {
        "weights": jnp.zeros(problem.spec.p, dtype=jnp.float64),
        "intercept": jnp.zeros((), dtype=jnp.float64),
    }
    return smooth_objective, initial, jax


def _torch_from_jax(array: Any, device: torch.device) -> torch.Tensor:
    result = torch.utils.dlpack.from_dlpack(array).detach().clone()
    if result.dtype != torch.float64 or result.device != device:
        raise RuntimeError(
            f"JAXopt returned dtype={result.dtype}, device={result.device}; "
            f"expected float64 on {device}"
        )
    return result


def solve_jaxopt_proximal_gradient(
    problem: ElasticNetProblem,
    *,
    native_tolerance: float,
    max_iterations: int,
) -> ElasticNetSolverResult:
    """Solve with JAXopt's default accelerated proximal-gradient method."""
    _validate_inputs(problem, native_tolerance, max_iterations)
    smooth_objective, initial, jax = _jax_problem(problem)
    import jaxopt

    def prox(params: dict[str, Any], l1_strength: float, scaling: float) -> dict[str, Any]:
        return {
            "weights": jaxopt.prox.prox_lasso(
                params["weights"],
                l1reg=l1_strength,
                scaling=scaling,
            ),
            "intercept": params["intercept"],
        }

    solver = jaxopt.ProximalGradient(
        fun=smooth_objective,
        prox=prox,
        maxiter=max_iterations,
        tol=native_tolerance,
    )
    # Compile outside the timed region; the measured run starts from zero again.
    warmup = solver.run(initial, hyperparams_prox=problem.lambda_l1)
    jax.block_until_ready(warmup.params)
    started = time.perf_counter()
    step = solver.run(initial, hyperparams_prox=problem.lambda_l1)
    jax.block_until_ready(step.params)
    runtime_seconds = time.perf_counter() - started
    iterations = int(step.state.iter_num)
    native_error = float(step.state.error)
    native_status = "converged" if native_error <= native_tolerance else "max_iterations"
    return ElasticNetSolverResult(
        weights=_torch_from_jax(step.params["weights"], problem.X.device),
        intercept=_torch_from_jax(step.params["intercept"], problem.X.device).reshape(()),
        runtime_seconds=runtime_seconds,
        iterations=iterations,
        native_status=native_status,
        native_error=native_error,
        metadata={
            "acceleration": solver.acceleration,
            "jit_warmup_excluded": True,
            "line_search": "backtracking",
        },
    )


def solve_cuml_coordinate_descent(
    problem: ElasticNetProblem,
    *,
    native_tolerance: float,
    max_iterations: int,
) -> ElasticNetSolverResult:
    """Solve with cuML's cyclic coordinate-descent ElasticNet."""
    _validate_inputs(problem, native_tolerance, max_iterations)
    _require_device(problem, "cuda", "cuML ElasticNet")
    try:
        import cupy as cp
        from cuml.linear_model import ElasticNet
    except ImportError as error:
        raise RuntimeError("cuML ElasticNet requires the pinned CUDA image") from error

    alpha, l1_ratio = _penalty_parameters(problem)
    features = cp.asfortranarray(cp.from_dlpack(problem.X.detach()))
    targets = cp.asarray(cp.from_dlpack(problem.y.detach()))
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=max_iterations,
        tol=native_tolerance,
        solver="cd",
        output_type="cupy",
    )
    _synchronize(problem.X.device)
    started = time.perf_counter()
    model.fit(features, targets)
    _synchronize(problem.X.device)
    runtime_seconds = time.perf_counter() - started
    iterations = int(cp.asnumpy(cp.asarray(model.n_iter_)).max())
    native_status = "converged" if iterations < max_iterations else "max_iterations"
    return ElasticNetSolverResult(
        weights=torch.from_dlpack(model.coef_).detach().clone(),
        intercept=torch.from_dlpack(cp.asarray(model.intercept_)).detach().clone().reshape(()),
        runtime_seconds=runtime_seconds,
        iterations=iterations,
        native_status=native_status,
        native_error=None,
        metadata={
            "alpha": alpha,
            "input_order": "F",
            "l1_ratio": l1_ratio,
            "selection": model.selection,
            "solver": model.solver,
        },
    )

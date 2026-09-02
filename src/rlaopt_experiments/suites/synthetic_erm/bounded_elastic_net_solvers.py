"""Direct solver adapters for box-constrained elastic-net regression."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy import sparse

from rlaopt_experiments.problems.synthetic_erm import ElasticNetProblem


@dataclass(frozen=True)
class BoundedElasticNetSolverResult:
    """Common result returned by every bounded elastic-net adapter."""

    weights: torch.Tensor
    intercept: torch.Tensor
    runtime_seconds: float
    iterations: int
    native_status: str
    native_error: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundedElasticNetConicForm:
    """SCS/Clarabel form for the canonical problem with variables ``(r, w, b)``."""

    quadratic: sparse.csc_matrix
    linear: np.ndarray
    constraints: sparse.csc_matrix
    rhs: np.ndarray
    n_samples: int
    n_features: int

    @property
    def zero_cone_dim(self) -> int:
        return self.n_samples

    @property
    def nonnegative_cone_dim(self) -> int:
        return 2 * self.n_features

    def extract_primal(self, solution: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.n_samples + self.n_features + 1
        if solution.shape != (expected,):
            raise ValueError(f"conic solution must have shape {(expected,)}")
        weights = torch.from_numpy(
            solution[self.n_samples : self.n_samples + self.n_features].copy()
        )
        intercept = torch.as_tensor(float(solution[-1]), dtype=torch.float64)
        return weights, intercept


def _validate_problem(problem: ElasticNetProblem) -> None:
    if not problem.bounded:
        raise ValueError("bounded elastic-net adapters require a bounded problem")
    if problem.X.dtype != torch.float64 or problem.y.dtype != torch.float64:
        raise ValueError("bounded elastic-net adapters require float64 problem data")


def _validate_controls(native_tolerance: float, max_iterations: int) -> None:
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


def build_bounded_elastic_net_conic_form(
    problem: ElasticNetProblem,
) -> BoundedElasticNetConicForm:
    """Build ``min 0.5 z'Pz + c'z`` subject to ``Az + s = b``.

    The first ``n`` cone rows are equalities enforcing
    ``r = Xw + intercept - y``. The remaining rows use a nonnegative
    slack to encode ``0 <= w <= 1``. Since bounded weights are nonnegative,
    the l1 penalty is the linear term ``lambda_l1 * sum(w)``.
    """
    _validate_problem(problem)
    _require_device(problem, "cpu", "conic-form construction")
    n = problem.spec.n
    p = problem.spec.p
    variable_count = n + p + 1

    diagonal = np.concatenate(
        (
            np.full(n, 1.0 / n, dtype=np.float64),
            np.full(p, problem.lambda_l2, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
        )
    )
    quadratic = sparse.diags(diagonal, format="csc")
    linear = np.zeros(variable_count, dtype=np.float64)
    linear[n : n + p] = problem.lambda_l1

    residual_equalities = sparse.hstack(
        (
            sparse.eye(n, format="csc"),
            -sparse.csc_matrix(problem.X.detach().numpy()),
            -sparse.csc_matrix(np.ones((n, 1), dtype=np.float64)),
        ),
        format="csc",
    )
    weight_columns = sparse.csc_matrix((p, n))
    intercept_column = sparse.csc_matrix((p, 1))
    lower = sparse.hstack(
        (weight_columns, -sparse.eye(p, format="csc"), intercept_column),
        format="csc",
    )
    upper = sparse.hstack(
        (weight_columns, sparse.eye(p, format="csc"), intercept_column),
        format="csc",
    )
    constraints = sparse.vstack((residual_equalities, lower, upper), format="csc")
    rhs = np.concatenate(
        (
            -problem.y.detach().numpy(),
            np.zeros(p, dtype=np.float64),
            np.ones(p, dtype=np.float64),
        )
    )
    return BoundedElasticNetConicForm(
        quadratic=quadratic,
        linear=linear,
        constraints=constraints,
        rhs=rhs,
        n_samples=n,
        n_features=p,
    )


def _build_rlaopt_objective(problem: ElasticNetProblem, loader: Any) -> tuple[Any, Any, Any]:
    """Build the rlaopt expression for the canonical bounded objective."""
    from rlaopt.atoms import Box, ElasticNet, LinearRegression
    from rlaopt.expression import Variable

    device = problem.X.device
    weights = Variable((problem.spec.p,), name="beta", dtype=torch.float64, device=device)
    model = LinearRegression(weights, loader, fit_intercept=True)
    scalar = {"dtype": torch.float64, "device": device}
    regularizer = ElasticNet(
        weights,
        l1_scaling=torch.tensor(problem.lambda_l1, **scalar),
        l2_scaling=torch.tensor(problem.lambda_l2, **scalar),
    )
    constraint = Box(
        weights,
        lower=torch.tensor(0.0, **scalar),
        upper=torch.tensor(1.0, **scalar),
    )
    return 0.5 * model + regularizer + constraint, weights, model.get_input("intercept")


def solve_rlaopt_admm(
    problem: ElasticNetProblem,
    *,
    native_tolerance: float,
    max_iterations: int,
    batch_size: int,
    seed: int,
) -> BoundedElasticNetSolverResult:
    """Solve with rlaopt ADMM using its default algorithm configuration."""
    _validate_problem(problem)
    _validate_controls(native_tolerance, max_iterations)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    from rlaopt.data import DataLoader, Dataset
    from rlaopt.solvers import ADMM, ADMMConfig, ADMMStoppingCriteria

    device = problem.X.device
    effective_batch_size = min(batch_size, problem.spec.n)
    previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        loader = DataLoader(
            Dataset(problem.X, problem.y, device=device, dtype=torch.float64),
            batch_size=effective_batch_size,
            shuffle=False,
        )
        objective, _, _ = _build_rlaopt_objective(problem, loader)
        config = ADMMConfig()
        solver = ADMM(objective, config)
        stopping = ADMMStoppingCriteria(
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

    primal_residual = float(native_result.primal_residual_norm)
    dual_residual = float(native_result.dual_residual_norm)
    return BoundedElasticNetSolverResult(
        weights=solution_weights,
        intercept=solution_intercept,
        runtime_seconds=runtime_seconds,
        iterations=native_result.num_iters,
        native_status=native_result.convergence_status.value,
        native_error=max(primal_residual, dual_residual),
        metadata={
            "batch_size": effective_batch_size,
            "dual_residual_norm": dual_residual,
            "native_objective_scale": 1.0,
            "native_solver_time_seconds": native_result.solver_time,
            "nystrom_rank": config.preconditioner_config.rank_init,
            "primal_residual_norm": primal_residual,
            "torch_default_dtype_workaround": True,
        },
    )


def solve_scs(
    problem: ElasticNetProblem,
    *,
    native_tolerance: float,
    max_iterations: int,
) -> BoundedElasticNetSolverResult:
    """Solve through SCS's direct Python interface using its CPU direct solver."""
    _validate_problem(problem)
    _validate_controls(native_tolerance, max_iterations)
    _require_device(problem, "cpu", "SCS")
    import scs

    preparation_started = time.perf_counter()
    conic = build_bounded_elastic_net_conic_form(problem)
    data = {
        "P": conic.quadratic,
        "A": conic.constraints,
        "b": conic.rhs,
        "c": conic.linear,
    }
    cone = {"z": conic.zero_cone_dim, "l": conic.nonnegative_cone_dim}
    preparation_seconds = time.perf_counter() - preparation_started
    started = time.perf_counter()
    native_result = scs.solve(
        data,
        cone,
        eps_abs=native_tolerance,
        eps_rel=native_tolerance,
        max_iters=max_iterations,
        verbose=False,
    )
    runtime_seconds = time.perf_counter() - started
    info = native_result["info"]
    raw_status = str(info["status"])
    native_status = "converged" if raw_status.lower() == "solved" else raw_status
    weights, intercept = conic.extract_primal(np.asarray(native_result["x"], dtype=np.float64))
    residuals = [abs(float(info[name])) for name in ("res_pri", "res_dual", "gap")]
    return BoundedElasticNetSolverResult(
        weights=weights,
        intercept=intercept,
        runtime_seconds=runtime_seconds,
        iterations=int(info["iter"]),
        native_status=native_status,
        native_error=max(residuals),
        metadata={
            "conic_preparation_seconds": preparation_seconds,
            "linear_solver": "cpu_direct",
            "native_duality_gap": float(info["gap"]),
            "native_primal_residual": float(info["res_pri"]),
            "native_dual_residual": float(info["res_dual"]),
            "native_setup_time_milliseconds": float(info["setup_time"]),
            "native_solve_time_milliseconds": float(info["solve_time"]),
            "raw_status": raw_status,
        },
    )

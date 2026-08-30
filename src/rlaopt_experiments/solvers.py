"""Unified, double-precision solver adapters."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from rlaopt_experiments.problem import RidgeProblem


@dataclass
class SolveResult:
    solution: torch.Tensor
    runtime_seconds: float
    iterations: int | None
    native_status: str
    trace: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, function: Callable[[], tuple]) -> tuple[float, tuple]:
    synchronize(device)
    started = time.perf_counter()
    value = function()
    synchronize(device)
    return time.perf_counter() - started, value


def _rlaopt(problem: RidgeProblem, ridge: float, tolerance: float, max_iters: int,
             nystrom_rank: int | None, timeout_seconds: float) -> SolveResult:
    from linops import aslinearoperator
    from rlaopt.linalg import IdentityConfig, LinSys, NystromConfig
    from rlaopt.solvers import PCG, PCGConfig

    x = problem.X
    x_op = aslinearoperator(x)
    normal_op = x_op.T @ x_op
    # torch-linops 0.2.0 does not propagate a wrapped tensor's device to
    # MatrixOperator or composed operators. rlaopt uses A.device to allocate
    # the Nyström sketch, so attach the known device explicitly.
    x_op.device = x.device
    x_op.T.device = x.device
    normal_op.device = x.device
    rhs = (x.mT @ problem.y).unsqueeze(-1)
    if nystrom_rank is None:
        preconditioner = IdentityConfig()
    else:
        rank = min(nystrom_rank, problem.spec.p)
        preconditioner = NystromConfig(
            rank_init=rank, rank_max=rank, base_damping=ridge, damping_mode="adaptive"
        )

    started = time.perf_counter()
    # rlaopt 0.1.0's randomized error estimator creates one vector using the
    # process default dtype. Keep that internal vector in benchmark float64.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(x.dtype)
    try:
        solver = PCG(LinSys(normal_op, rhs, reg=float(ridge)), PCGConfig(
            preconditioner_config=preconditioner
        ))
    finally:
        torch.set_default_dtype(previous_dtype)
    estimate = torch.zeros_like(rhs)
    state = solver.init_state(estimate)
    rhs_norm = torch.linalg.vector_norm(rhs)
    trace: list[dict[str, float]] = []
    status = "iteration_limit"
    for iteration in range(max_iters):
        estimate, state = solver.step(estimate, state)
        relative = float(torch.max(state.res_norm) / rhs_norm)
        trace.append({"iteration": iteration + 1, "native_relative_residual": relative,
                      "elapsed_seconds": time.perf_counter() - started})
        if relative <= tolerance:
            status = "native_converged"
            break
        if time.perf_counter() - started >= timeout_seconds:
            status = "timeout"
            break
    synchronize(x.device)
    return SolveResult(estimate.squeeze(-1), time.perf_counter() - started,
                       len(trace), status, trace)


def scipy_lsqr(problem: RidgeProblem, ridge: float, tolerance: float,
               max_iters: int, timeout_seconds: float) -> SolveResult:
    del timeout_seconds  # The job-level process timeout is the hard enforcement mechanism.
    from scipy.sparse.linalg import LinearOperator, lsqr

    x = problem.X.detach().cpu().numpy()
    y = problem.y.detach().cpu().numpy()
    operator = LinearOperator(x.shape, matvec=lambda v: x @ v, rmatvec=lambda v: x.T @ v,
                              dtype=np.float64)
    runtime, output = _timed(torch.device("cpu"), lambda: lsqr(
        operator, y, damp=np.sqrt(ridge), atol=tolerance, btol=tolerance,
        iter_lim=max_iters, show=False
    ))
    solution, istop, iterations, r1norm, r2norm, anorm, acond, arnorm, xnorm = output[:9]
    return SolveResult(torch.from_numpy(solution), runtime, int(iterations), f"istop_{istop}",
                       metadata={"r1norm": r1norm, "r2norm": r2norm, "anorm": anorm,
                                 "acond": acond, "arnorm": arnorm, "xnorm": xnorm})


def torch_lstsq_qr(problem: RidgeProblem, ridge: float, **_: Any) -> SolveResult:
    """Solve the augmented ridge problem with PyTorch's QR-based lstsq driver."""
    x, y = problem.X, problem.y
    def factor_and_solve() -> tuple:
        augmented_x = torch.cat((
            x,
            np.sqrt(ridge) * torch.eye(problem.spec.p, dtype=x.dtype, device=x.device),
        ))
        augmented_y = torch.cat((
            y,
            torch.zeros(problem.spec.p, dtype=y.dtype, device=y.device),
        ))
        return torch.linalg.lstsq(
            augmented_x,
            augmented_y,
            driver="gels" if x.device.type == "cuda" else "gelsy",
        )

    runtime, output = _timed(x.device, factor_and_solve)
    return SolveResult(output[0], runtime, None, "direct")


def cuml_lsmr(problem: RidgeProblem, ridge: float, tolerance: float,
              max_iters: int, **_: Any) -> SolveResult:
    try:
        import cupy as cp
        from cuml.linear_model import Ridge
    except ImportError as error:
        raise RuntimeError("cuml_lsmr requires the pinned CUDA image") from error
    x_cp = cp.from_dlpack(problem.X)
    y_cp = cp.from_dlpack(problem.y)
    model = Ridge(alpha=ridge, fit_intercept=False, solver="lsmr", tol=tolerance,
                  max_iter=max_iters, output_type="cupy")
    runtime, _ = _timed(problem.X.device, lambda: (model.fit(x_cp, y_cp),))
    solution = torch.from_dlpack(model.coef_)
    native_iterations = getattr(model, "n_iter_", None)
    iterations = int(np.asarray(native_iterations).max()) if native_iterations is not None else None
    return SolveResult(solution, runtime, iterations, "native_complete")


def solve(name: str, problem: RidgeProblem, ridge: float, tolerance: float,
          max_iters: int, timeout_seconds: float, nystrom_rank: int = 128) -> SolveResult:
    kwargs = dict(problem=problem, ridge=ridge, tolerance=tolerance, max_iters=max_iters,
                  timeout_seconds=timeout_seconds)
    if name == "rlaopt_nystrom_pcg":
        return _rlaopt(**kwargs, nystrom_rank=nystrom_rank)
    if name == "rlaopt_cg":
        return _rlaopt(**kwargs, nystrom_rank=None)
    adapters = {
        "scipy_lsqr": scipy_lsqr,
        "torch_qr": torch_lstsq_qr,
        "cuml_lsmr": cuml_lsmr,
    }
    try:
        return adapters[name](**kwargs)
    except KeyError as error:
        raise ValueError(f"unknown solver: {name}") from error

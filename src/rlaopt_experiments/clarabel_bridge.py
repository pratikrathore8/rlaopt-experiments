"""JuliaCall bridge for the direct Clarabel and CuClarabel interfaces.

This module intentionally does not import PyTorch. Clarabel workers initialize
Julia through this module before importing the benchmark problem stack, avoiding
the Julia/PyTorch shared-library conflict observed in the CUDA container.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any


_JULIA_HELPERS = r"""
using Clarabel, PythonCall, SparseArrays

function rlaopt_clarabel_solve_cpu(
    P, q, A, b, equality_dim, inequality_dim, tolerance, max_iterations
)
    settings = Clarabel.Settings(
        direct_solve_method = :qdldl,
        verbose = false,
        tol_gap_abs = tolerance,
        tol_gap_rel = tolerance,
        tol_feas = tolerance,
        max_iter = max_iterations,
    )
    cones = [Clarabel.ZeroConeT(equality_dim), Clarabel.NonnegativeConeT(inequality_dim)]
    solver = Clarabel.Solver(P, q, A, b, cones, settings)
    Clarabel.solve!(solver)
    return solver
end

function rlaopt_clarabel_solve_gpu(
    P, q, A, b, equality_dim, inequality_dim, tolerance, max_iterations
)
    settings = Clarabel.Settings(
        direct_solve_method = :cudss,
        verbose = false,
        tol_gap_abs = tolerance,
        tol_gap_rel = tolerance,
        tol_feas = tolerance,
        max_iter = max_iterations,
    )
    cones = [Clarabel.ZeroConeT(equality_dim), Clarabel.NonnegativeConeT(inequality_dim)]
    # CuClarabel's convenience constructor requires q::Vector even though
    # setup! correctly accepts GPU AbstractVector inputs.
    solver = Clarabel.Solver{Float64}()
    Clarabel.setup!(solver, P, q, A, b, cones, settings)
    Clarabel.solve!(solver)
    CUDA.synchronize()
    return solver
end

# CuClarabel's Python extension aliases CuPy-owned device buffers with
# unsafe_wrap(..., own=false). Copy every wrapped allocation before it enters
# solver state so Julia, rather than CuPy's memory pool, owns its lifetime.
function rlaopt_copy_cupy_vector(T::Type, ptr, rows)
    device_pointer = CUDA.CuPtr{T}(UInt(ptr))
    aliased = CUDA.unsafe_wrap(CUDA.CuVector{T}, device_pointer, rows)
    return copy(aliased)
end

function rlaopt_copy_cupy_csr(
    T::Type, data_ptr, indices_ptr, indptr_ptr, n_rows, n_cols, nnz
)
    n_rows = Int32(n_rows)
    n_cols = Int32(n_cols)
    data = copy(CUDA.unsafe_wrap(
        CUDA.CuVector{T}, CUDA.CuPtr{T}(UInt(data_ptr)), nnz
    ))
    indices = copy(CUDA.unsafe_wrap(
        CUDA.CuVector{Int32}, CUDA.CuPtr{Int32}(UInt(indices_ptr)), nnz
    ))
    indptr = copy(CUDA.unsafe_wrap(
        CUDA.CuVector{Int32}, CUDA.CuPtr{Int32}(UInt(indptr_ptr)), n_rows + 1
    ))
    @. indices += 1
    @. indptr += 1
    return CUDA.CUSPARSE.CuSparseMatrixCSR(indptr, indices, data, (n_rows, n_cols))
end
"""


@dataclass(frozen=True)
class PreparedClarabelData:
    """Backend-native conic arrays whose owners remain live through extraction."""

    quadratic: Any
    linear: Any
    constraints: Any
    rhs: Any
    owners: tuple[Any, ...]


class ClarabelRuntime:
    """Initialized Julia runtime and backend-specific array bridge."""

    def __init__(self, main: Any, backend: str):
        if backend not in {"cpu", "cuda"}:
            raise ValueError("Clarabel backend must be cpu or cuda")
        self.main = main
        self.backend = backend

    def prepare(
        self,
        quadratic: Any,
        linear: Any,
        constraints: Any,
        rhs: Any,
    ) -> PreparedClarabelData:
        """Convert immutable SciPy/NumPy arrays to fresh backend-native arrays."""
        if self.backend == "cpu":
            return self._prepare_cpu(quadratic, linear, constraints, rhs)
        return self._prepare_cuda(quadratic, linear, constraints, rhs)

    def solve(
        self,
        data: PreparedClarabelData,
        *,
        equality_dim: int,
        inequality_dim: int,
        native_tolerance: float,
        max_iterations: int,
    ) -> Any:
        """Construct and solve a fresh native Clarabel solver."""
        arguments = (
            data.quadratic,
            data.linear,
            data.constraints,
            data.rhs,
            equality_dim,
            inequality_dim,
            native_tolerance,
            max_iterations,
        )
        if self.backend == "cpu":
            return self.main.rlaopt_clarabel_solve_cpu(*arguments)

        solver = self.main.rlaopt_clarabel_solve_gpu(*arguments)
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()
        return solver

    def _prepare_cpu(
        self,
        quadratic: Any,
        linear: Any,
        constraints: Any,
        rhs: Any,
    ) -> PreparedClarabelData:
        import numpy as np

        def sparse_matrix(matrix: Any) -> Any:
            csc = matrix.tocsc(copy=True)
            column_pointers = np.asarray(csc.indptr, dtype=np.int64) + 1
            row_indices = np.asarray(csc.indices, dtype=np.int64) + 1
            values = np.asarray(csc.data, dtype=np.float64)
            return self.main.SparseMatrixCSC(
                csc.shape[0],
                csc.shape[1],
                self.main.Vector[self.main.Int64](column_pointers),
                self.main.Vector[self.main.Int64](row_indices),
                self.main.Vector[self.main.Float64](values),
            )

        return PreparedClarabelData(
            quadratic=sparse_matrix(quadratic),
            linear=self.main.Vector[self.main.Float64](np.asarray(linear, dtype=np.float64)),
            constraints=sparse_matrix(constraints),
            rhs=self.main.Vector[self.main.Float64](np.asarray(rhs, dtype=np.float64)),
            owners=(),
        )

    def _prepare_cuda(
        self,
        quadratic: Any,
        linear: Any,
        constraints: Any,
        rhs: Any,
    ) -> PreparedClarabelData:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix

        # Use dedicated CuPy inputs because the bridge first aliases their raw
        # pointers. The Julia helpers immediately copy those aliases into
        # Julia-owned device allocations before shifting CSR indices in place.
        p_owner = csr_matrix(quadratic, dtype=cp.float64)
        q_owner = cp.asarray(linear, dtype=cp.float64)
        a_owner = csr_matrix(constraints, dtype=cp.float64)
        b_owner = cp.asarray(rhs, dtype=cp.float64)

        def vector(array: Any) -> Any:
            return self.main.rlaopt_copy_cupy_vector(
                self.main.Float64,
                int(array.data.ptr),
                array.size,
            )

        def sparse_matrix(matrix: Any) -> Any:
            return self.main.rlaopt_copy_cupy_csr(
                self.main.Float64,
                int(matrix.data.data.ptr),
                int(matrix.indices.data.ptr),
                int(matrix.indptr.data.ptr),
                matrix.shape[0],
                matrix.shape[1],
                matrix.nnz,
            )

        return PreparedClarabelData(
            quadratic=sparse_matrix(p_owner),
            linear=vector(q_owner),
            constraints=sparse_matrix(a_owner),
            rhs=vector(b_owner),
            owners=(p_owner, q_owner, a_owner, b_owner),
        )


def initialize_clarabel_runtime(backend: str) -> ClarabelRuntime:
    """Initialize Julia before PyTorch and load the requested direct backend."""
    if backend not in {"cpu", "cuda"}:
        raise ValueError("Clarabel backend must be cpu or cuda")
    if "torch" in sys.modules:
        raise RuntimeError("Clarabel runtime must be initialized before importing PyTorch")

    from juliacall import Main as jl

    if backend == "cuda":
        jl.seval("using Clarabel, PythonCall, SparseArrays, CUDA, CUDA.CUSPARSE, CUDSS")
    else:
        jl.seval("using Clarabel, PythonCall, SparseArrays")
    jl.seval(_JULIA_HELPERS)
    return ClarabelRuntime(jl, backend)

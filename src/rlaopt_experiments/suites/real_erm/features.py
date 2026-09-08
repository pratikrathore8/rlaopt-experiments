"""Deterministically materialize the PROMISE random-feature maps at runtime."""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sparse
import torch

from rlaopt_experiments.suites.real_erm.data import RandomFeatureSpec


CpuMatrix = np.ndarray | sparse.spmatrix
InputMatrix = CpuMatrix | torch.Tensor


def _promise_parameters(
    input_features: int,
    spec: RandomFeatureSpec,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Draw the same parameters as PROMISE without changing NumPy's global RNG."""
    random_state = np.random.RandomState(spec.seed)
    weights = random_state.randn(spec.dimension, input_features)
    weights /= np.sqrt(spec.dimension)
    if spec.kind == "gaussian":
        weights /= spec.bandwidth
        phase = random_state.uniform(0.0, 2.0 * np.pi, size=spec.dimension)
        return weights, phase
    return weights, None


def _validate_matrix(matrix: InputMatrix) -> tuple[int, int]:
    if matrix.ndim != 2:
        raise ValueError("random-feature input must be a matrix")
    rows, columns = matrix.shape
    if rows < 1 or columns < 1:
        raise ValueError("random-feature input dimensions must be positive")
    return int(rows), int(columns)


def _materialize_cpu(matrix: InputMatrix, spec: RandomFeatureSpec) -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        if matrix.device.type != "cpu" or matrix.layout != torch.strided:
            raise ValueError("CPU random features require a dense CPU tensor or NumPy/SciPy input")
        matrix = matrix.detach().numpy()
    _, input_features = _validate_matrix(matrix)
    weights, phase = _promise_parameters(input_features, spec)
    if sparse.issparse(matrix):
        base = sparse.csr_matrix(matrix, dtype=np.float64)
    else:
        base = np.asarray(matrix, dtype=np.float64)
    output = np.asarray(base @ weights.T, dtype=np.float64)
    if spec.kind == "gaussian":
        output += phase
        np.cos(output, out=output)
        output *= np.sqrt(2.0 / spec.dimension)
    else:
        np.maximum(output, 0.0, out=output)
    return output


def _torch_matrix(matrix: InputMatrix, device: torch.device) -> torch.Tensor:
    if isinstance(matrix, torch.Tensor):
        return matrix.detach().to(device=device, dtype=torch.float64)
    if sparse.issparse(matrix):
        base = sparse.csr_matrix(matrix, dtype=np.float64)
        return torch.sparse_csr_tensor(
            torch.as_tensor(base.indptr, dtype=torch.int64, device=device),
            torch.as_tensor(base.indices, dtype=torch.int64, device=device),
            torch.as_tensor(base.data, dtype=torch.float64, device=device),
            size=base.shape,
            dtype=torch.float64,
            device=device,
            check_invariants=True,
        )
    return torch.as_tensor(np.asarray(matrix), dtype=torch.float64, device=device)


def _materialize_cuda(matrix: InputMatrix, spec: RandomFeatureSpec) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA random features require an available CUDA device")
    _, input_features = _validate_matrix(matrix)
    weights, phase = _promise_parameters(input_features, spec)
    device = torch.device("cuda")
    base = _torch_matrix(matrix, device)
    torch_weights = torch.as_tensor(weights, dtype=torch.float64, device=device)
    output = torch.mm(base, torch_weights.T)
    if spec.kind == "gaussian":
        output.add_(torch.as_tensor(phase, dtype=torch.float64, device=device))
        output.cos_()
        output.mul_(np.sqrt(2.0 / spec.dimension))
    else:
        output.relu_()
    return output


def materialize_random_features(
    matrix: InputMatrix,
    spec: RandomFeatureSpec,
    *,
    backend: Literal["cpu", "cuda"],
) -> np.ndarray | torch.Tensor:
    """Create the full float64 feature matrix in host RAM or CUDA device memory.

    The returned matrix is materialized—not a linear operator—and therefore gives every
    solver the same explicit problem. The output is formed once and its nonlinearity is
    applied in place to avoid a second output-sized temporary.
    """
    if backend == "cpu":
        return _materialize_cpu(matrix, spec)
    if backend == "cuda":
        return _materialize_cuda(matrix, spec)
    raise ValueError("random-feature backend must be cpu or cuda")

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sparse
import torch

from rlaopt_experiments.suites.real_erm.data import DATASETS, RandomFeatureSpec
from rlaopt_experiments.suites.real_erm.features import materialize_random_features


def _expected_parameters(
    input_features: int, spec: RandomFeatureSpec
) -> tuple[np.ndarray, np.ndarray | None]:
    np.random.seed(spec.seed)
    weights = np.random.randn(spec.dimension, input_features) / np.sqrt(spec.dimension)
    if spec.kind == "gaussian":
        weights /= spec.bandwidth
        phase = np.random.uniform(0.0, 2.0 * np.pi, size=spec.dimension)
        return weights, phase
    return weights, None


def test_catalog_exposes_base_and_solver_shapes() -> None:
    assert DATASETS["acsincome"].base_shape == (1_664_500, 11)
    assert DATASETS["acsincome"].solver_shape == (1_664_500, 1_000)
    assert DATASETS["yolanda"].solver_shape == (400_000, 1_000)
    assert DATASETS["yearpredictionmsd"].solver_shape == (463_715, 4_367)
    assert DATASETS["e2006"].solver_shape == DATASETS["e2006"].base_shape


def test_gaussian_features_match_promise_and_are_deterministic() -> None:
    matrix = np.array([[1.0, 2.0, -1.0], [0.5, -3.0, 4.0]], dtype=np.float64)
    spec = RandomFeatureSpec("gaussian", 5, seed=2468, bandwidth=1.0)
    weights, phase = _expected_parameters(matrix.shape[1], spec)
    expected = np.sqrt(2.0 / spec.dimension) * np.cos(matrix @ weights.T + phase)

    first = materialize_random_features(matrix, spec, backend="cpu")
    second = materialize_random_features(matrix, spec, backend="cpu")

    assert first.shape == (2, 5)
    assert first.dtype == np.float64
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, expected, rtol=0.0, atol=1e-15)


def test_feature_generation_does_not_advance_numpy_global_rng() -> None:
    matrix = np.array([[1.0, 2.0]], dtype=np.float64)
    spec = RandomFeatureSpec("gaussian", 3, seed=2468, bandwidth=1.0)
    np.random.seed(123)
    expected = np.random.random(4)
    np.random.seed(123)

    materialize_random_features(matrix, spec, backend="cpu")

    np.testing.assert_array_equal(np.random.random(4), expected)


def test_relu_features_match_promise_for_sparse_input() -> None:
    matrix = sparse.csr_matrix([[1.0, 0.0, -2.0], [0.0, 3.0, 1.0]])
    spec = RandomFeatureSpec("relu", 4, seed=2468)
    weights, phase = _expected_parameters(matrix.shape[1], spec)
    assert phase is None
    expected = np.maximum(matrix.toarray() @ weights.T, 0.0)

    observed = materialize_random_features(matrix, spec, backend="cpu")

    assert observed.shape == (2, 4)
    assert observed.dtype == np.float64
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)


def test_cpu_tensor_input_and_invalid_backend() -> None:
    matrix = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    spec = RandomFeatureSpec("relu", 3)
    observed = materialize_random_features(matrix, spec, backend="cpu")
    assert isinstance(observed, np.ndarray)
    with pytest.raises(ValueError, match="backend must be cpu or cuda"):
        materialize_random_features(matrix, spec, backend="tpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kind", ["gaussian", "relu"])
def test_cuda_features_match_cpu(kind: str) -> None:
    matrix = np.array([[1.0, 2.0, -1.0], [0.5, -3.0, 4.0]], dtype=np.float64)
    spec = RandomFeatureSpec(
        kind,
        5,
        seed=2468,
        bandwidth=1.0 if kind == "gaussian" else None,
    )
    expected = materialize_random_features(matrix, spec, backend="cpu")
    observed = materialize_random_features(matrix, spec, backend="cuda")
    assert isinstance(observed, torch.Tensor)
    assert observed.device.type == "cuda"
    assert observed.dtype == torch.float64
    np.testing.assert_allclose(observed.cpu().numpy(), expected, rtol=1e-13, atol=1e-15)

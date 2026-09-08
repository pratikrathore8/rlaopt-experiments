"""Smoke-test real-data random features on the allocated CUDA device."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
import torch

from rlaopt_experiments.suites.real_erm.data import RandomFeatureSpec
from rlaopt_experiments.suites.real_erm.features import materialize_random_features


def main() -> None:
    dense = np.array([[1.0, 2.0, -1.0], [0.5, -3.0, 4.0]], dtype=np.float64)
    matrices = {
        "gaussian": dense,
        "relu": sparse.csr_matrix(dense),
    }
    for kind, matrix in matrices.items():
        spec = RandomFeatureSpec(
            kind,
            5,
            seed=2468,
            bandwidth=1.0 if kind == "gaussian" else None,
        )
        expected = materialize_random_features(matrix, spec, backend="cpu")
        observed = materialize_random_features(matrix, spec, backend="cuda")
        torch.testing.assert_close(
            observed.cpu(),
            torch.from_numpy(expected),
            rtol=1e-13,
            atol=1e-15,
        )
        print(f"{kind}: {tuple(observed.shape)}, {observed.dtype}, {observed.device}")


if __name__ == "__main__":
    main()

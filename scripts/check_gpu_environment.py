"""Exercise every GPU benchmark dependency in one Python interpreter."""

from __future__ import annotations

import platform

import cupy
import cuml
import numpy
import rlaopt
import torch
from cuml.linear_model import Ridge


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see CUDA")
    torch_value = torch.ones(4, device="cuda", dtype=torch.float64)
    if torch_value.dtype != torch.float64:
        raise RuntimeError("PyTorch float64 check failed")

    x = cupy.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=cupy.float64)
    y = cupy.asarray([1.0, 2.0, 3.0], dtype=cupy.float64)
    model = Ridge(
        alpha=1e-3,
        fit_intercept=False,
        solver="lsmr",
        tol=1e-10,
        max_iter=100,
        output_type="cupy",
    ).fit(x, y)
    if model.coef_.dtype != cupy.float64:
        raise RuntimeError(f"cuML returned {model.coef_.dtype}, expected float64")

    print("Python:", platform.python_version())
    print("NumPy:", numpy.__version__)
    print("rlaopt:", getattr(rlaopt, "__version__", "0.1.0"))
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("PyTorch GPU:", torch.cuda.get_device_name(0))
    print("cuML:", cuml.__version__)
    print("CuPy:", cupy.__version__)
    print("cuML coefficient dtype:", model.coef_.dtype)
    print("cuML iterations:", model.n_iter_)
    print("PROJECT_ENVIRONMENT_INTEROPERABLE=true")


if __name__ == "__main__":
    main()

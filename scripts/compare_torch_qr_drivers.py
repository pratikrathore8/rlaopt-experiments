"""Compare PyTorch's pivoted and unpivoted CPU QR least-squares drivers."""

from __future__ import annotations

import json
import os
import time

import torch

from rlaopt_experiments.problem import (
    ProblemSpec,
    generate_problem,
    relative_kkt,
    relative_solution_error,
)
from rlaopt_experiments.seeds import derive_seed


def solve(problem, ridge: float, driver: str) -> tuple[float, torch.Tensor]:
    started = time.perf_counter()
    augmented_x = torch.cat(
        (
            problem.X,
            ridge**0.5 * torch.eye(problem.spec.p, dtype=problem.X.dtype),
        )
    )
    augmented_y = torch.cat((problem.y, torch.zeros(problem.spec.p, dtype=problem.y.dtype)))
    solution = torch.linalg.lstsq(augmented_x, augmented_y, driver=driver).solution
    return time.perf_counter() - started, solution


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(int(os.environ.get("BENCHMARK_CPU_THREADS", "64")))
    seed = 0
    spec = ProblemSpec(
        n=16384,
        p=4096,
        alpha=0.5,
        factor_seed=derive_seed(seed, "factors"),
        response_seed=derive_seed(seed, "response"),
    )
    problem = generate_problem(spec)
    results = []
    for ridge in (1e-2, 1e-6):
        # Warm both implementations before alternating timed calls.
        for driver in ("gelsy", "gels"):
            solve(problem, ridge, driver)
        for repetition in range(5):
            drivers = ("gelsy", "gels") if repetition % 2 == 0 else ("gels", "gelsy")
            for driver in drivers:
                runtime, solution = solve(problem, ridge, driver)
                row = {
                    "ridge": ridge,
                    "driver": driver,
                    "repetition": repetition,
                    "runtime_seconds": runtime,
                    "relative_kkt": relative_kkt(problem, solution, ridge),
                    "relative_solution_error": relative_solution_error(problem, solution, ridge),
                }
                results.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    print(json.dumps({"results": results}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

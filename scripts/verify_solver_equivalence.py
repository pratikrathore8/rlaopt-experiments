"""Verify that every adapter solves the same ridge objective on one problem."""

from __future__ import annotations

import argparse
import itertools

import torch

from rlaopt_experiments.problem import (
    ProblemSpec,
    generate_problem,
    relative_kkt,
    relative_solution_error,
)
from rlaopt_experiments.solvers import solve


SOLVERS = {
    "cpu": ("rlaopt_nystrom_pcg", "rlaopt_cg", "scipy_lsqr", "torch_qr"),
    "cuda": ("rlaopt_nystrom_pcg", "rlaopt_cg", "cuml_lsmr", "torch_qr"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=tuple(SOLVERS), required=True)
    args = parser.parse_args()

    device = torch.device(args.backend)
    torch.set_default_dtype(torch.float64)
    problem = generate_problem(ProblemSpec(256, 128, 1.0, 11, 12), device=device)
    maximum_iterations = 2 * problem.spec.p

    for ridge in (1e-2, 1e-4, 1e-6):
        solutions: dict[str, torch.Tensor] = {}
        for solver_name in SOLVERS[args.backend]:
            torch.manual_seed(13)
            if args.backend == "cuda":
                torch.cuda.manual_seed_all(13)
            result = solve(
                solver_name,
                problem,
                ridge,
                tolerance=1e-10,
                max_iters=maximum_iterations,
                timeout_seconds=300,
                nystrom_rank=64,
            )
            kkt = relative_kkt(problem, result.solution, ridge)
            oracle_error = relative_solution_error(problem, result.solution, ridge)
            print(
                f"backend={args.backend} ridge={ridge:g} solver={solver_name} "
                f"kkt={kkt:.3e} oracle_error={oracle_error:.3e}",
                flush=True,
            )
            if kkt > 1e-8 or oracle_error > 1e-7:
                raise RuntimeError(f"{solver_name} does not match the ridge oracle")
            solutions[solver_name] = result.solution

        for left_name, right_name in itertools.combinations(solutions, 2):
            left = solutions[left_name]
            right = solutions[right_name]
            relative = float(
                torch.linalg.vector_norm(left - right) / torch.linalg.vector_norm(left)
            )
            print(
                f"backend={args.backend} ridge={ridge:g} pair={left_name}:{right_name} "
                f"relative_difference={relative:.3e}",
                flush=True,
            )
            if relative > 1e-7:
                raise RuntimeError(f"{left_name} and {right_name} disagree")

    print(f"SOLVER_EQUIVALENCE_{args.backend.upper()}=true")


if __name__ == "__main__":
    main()

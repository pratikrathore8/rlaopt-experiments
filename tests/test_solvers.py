import torch

from rlaopt_experiments.problem import ProblemSpec, generate_problem, relative_solution_error
from rlaopt_experiments.solvers import solve


def test_cpu_adapters_solve_the_same_ridge_problem():
    problem = generate_problem(ProblemSpec(32, 16, 1.0, 11, 12))
    solutions = []
    for solver_name in ("rlaopt_nystrom_pcg", "rlaopt_cg", "scipy_lsqr", "torch_qr"):
        torch.manual_seed(13)
        result = solve(
            solver_name,
            problem,
            ridge=1e-3,
            tolerance=1e-10,
            max_iters=2 * problem.spec.p,
            timeout_seconds=30,
            nystrom_rank=8,
        )
        assert relative_solution_error(problem, result.solution, 1e-3) < 1e-7
        solutions.append(result.solution)

    reference = solutions[-1]
    for solution in solutions[:-1]:
        relative = torch.linalg.vector_norm(solution - reference) / torch.linalg.vector_norm(reference)
        assert relative < 1e-7

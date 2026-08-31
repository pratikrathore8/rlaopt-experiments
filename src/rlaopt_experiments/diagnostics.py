"""External accuracy adjudication independent of native solver status."""

from __future__ import annotations

from dataclasses import dataclass

from rlaopt_experiments.problem import RidgeProblem, relative_kkt, relative_solution_error
from rlaopt_experiments.solvers import SolveResult


@dataclass(frozen=True)
class Accuracy:
    relative_kkt: float
    relative_solution_error: float
    success: bool


def adjudicate(
    problem: RidgeProblem, ridge: float, result: SolveResult, target: float = 1e-6
) -> Accuracy:
    kkt = relative_kkt(problem, result.solution, ridge)
    solution_error = relative_solution_error(problem, result.solution, ridge)
    return Accuracy(kkt, solution_error, kkt <= target)


def choose_native_tolerance(outcomes: dict[float, list[Accuracy]], target: float) -> float:
    """Choose the loosest candidate that meets the target for every calibration case."""
    valid = [
        tol
        for tol, values in outcomes.items()
        if values and all(item.relative_kkt <= target for item in values)
    ]
    if not valid:
        raise RuntimeError("no native tolerance passed every calibration case")
    return max(valid)

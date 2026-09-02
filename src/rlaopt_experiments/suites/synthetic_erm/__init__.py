"""Synthetic empirical-risk-minimization benchmark suite."""

from rlaopt_experiments.suites.synthetic_erm.config import (
    AccuracyThresholds,
    BackendTolerances,
    BackendSolvers,
    ElasticNetExperiment,
    ErmShape,
    MultinomialExecution,
    MultinomialExperiment,
    SyntheticErmConfig,
    load_synthetic_erm_config,
)
from rlaopt_experiments.suites.synthetic_erm.multinomial_solvers import (
    MultinomialSolverResult,
    solve_jaxopt_lbfgsb,
    solve_jaxopt_projected_gradient,
    solve_rlaopt_sapphire,
)

__all__ = [
    "AccuracyThresholds",
    "BackendTolerances",
    "BackendSolvers",
    "ElasticNetExperiment",
    "ErmShape",
    "MultinomialExecution",
    "MultinomialExperiment",
    "MultinomialSolverResult",
    "SyntheticErmConfig",
    "load_synthetic_erm_config",
    "solve_jaxopt_lbfgsb",
    "solve_jaxopt_projected_gradient",
    "solve_rlaopt_sapphire",
]

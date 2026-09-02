"""Synthetic empirical-risk-minimization benchmark suite."""

from rlaopt_experiments.suites.synthetic_erm.config import (
    AccuracyThresholds,
    BackendSolvers,
    ElasticNetExperiment,
    ErmShape,
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
    "BackendSolvers",
    "ElasticNetExperiment",
    "ErmShape",
    "MultinomialExperiment",
    "MultinomialSolverResult",
    "SyntheticErmConfig",
    "load_synthetic_erm_config",
    "solve_jaxopt_lbfgsb",
    "solve_jaxopt_projected_gradient",
    "solve_rlaopt_sapphire",
]

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

__all__ = [
    "AccuracyThresholds",
    "BackendSolvers",
    "ElasticNetExperiment",
    "ErmShape",
    "MultinomialExperiment",
    "SyntheticErmConfig",
    "load_synthetic_erm_config",
]

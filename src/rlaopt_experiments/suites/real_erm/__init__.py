"""Real-data empirical-risk-minimization benchmark support."""

from rlaopt_experiments.suites.real_erm.data import (
    DATASETS,
    load_prepared_dataset,
    prepare_datasets,
)

__all__ = ["DATASETS", "load_prepared_dataset", "prepare_datasets"]

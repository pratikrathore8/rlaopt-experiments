"""Suite-aware benchmark manifest generation."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from rlaopt_experiments.config import load_experiment, load_solvers
from rlaopt_experiments.suites.synthetic_erm.config import load_synthetic_erm_config
from rlaopt_experiments.suites.synthetic_erm.manifest import (
    build_multinomial_manifest,
)


def _configured_suite(path: Path) -> str:
    root = tomllib.loads(path.read_text())
    try:
        suite = root["experiment"]["suite"]
    except (KeyError, TypeError) as error:
        raise ValueError("configuration must define experiment.suite") from error
    if not isinstance(suite, str) or not suite:
        raise ValueError("experiment.suite must be a nonempty string")
    return suite


def build_manifest(path: Path, backend: str) -> list[dict[str, Any]]:
    """Build the selected suite's manifest from one strict configuration."""
    suite = _configured_suite(path)
    if suite == "synthetic_erm":
        return build_multinomial_manifest(load_synthetic_erm_config(path), backend)
    if suite == "synthetic_ridge":
        config = load_experiment(path)
        solvers = load_solvers(path, backend)
        return [
            {
                "suite": config.suite,
                "n": shape.n,
                "p": shape.p,
                "family": shape.family,
                "alpha": alpha,
                "seed": seed,
                "solver": solver,
                "backend": backend,
            }
            for shape in config.shapes
            for alpha in config.alphas
            for seed in config.seeds
            for solver in solvers
        ]
    raise ValueError(f"unknown benchmark suite: {suite}")

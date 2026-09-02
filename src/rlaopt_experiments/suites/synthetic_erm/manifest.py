"""Deterministic job manifests for the synthetic ERM suite."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rlaopt_experiments.problems.synthetic_erm import MultinomialSpec
from rlaopt_experiments.seeds import derive_seed
from rlaopt_experiments.suites.synthetic_erm.config import SyntheticErmConfig


def build_multinomial_manifest(
    config: SyntheticErmConfig,
    backend: str,
) -> list[dict[str, Any]]:
    """Expand the configured multinomial grid into self-contained jobs."""
    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")
    solvers = getattr(config.multinomial.solvers, backend)
    jobs: list[dict[str, Any]] = []
    for shape in config.multinomial.shapes:
        for seed in config.seeds:
            problem_spec = MultinomialSpec(
                n=shape.n,
                p=shape.p,
                n_classes=config.multinomial.n_classes,
                feature_seed=derive_seed(seed, "multinomial_features"),
                target_seed=derive_seed(seed, "multinomial_targets"),
                teacher_scale=config.multinomial.teacher_scale,
                box_lower=config.multinomial.box_lower,
                box_upper=config.multinomial.box_upper,
            )
            for solver in solvers:
                jobs.append(
                    {
                        "suite": config.suite,
                        "problem_type": "multinomial",
                        "problem_id": problem_spec.problem_id,
                        "problem_spec": asdict(problem_spec),
                        "seed": seed,
                        "solver_seed": derive_seed(seed, "multinomial_solver"),
                        "solver": solver,
                        "backend": backend,
                    }
                )
    return jobs

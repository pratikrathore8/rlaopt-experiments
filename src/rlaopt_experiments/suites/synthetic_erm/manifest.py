"""Deterministic job manifests for the synthetic ERM suite."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rlaopt_experiments.problems.synthetic_erm import ElasticNetSpec, MultinomialSpec
from rlaopt_experiments.seeds import derive_seed
from rlaopt_experiments.suites.synthetic_erm.config import SyntheticErmConfig


def _require_backend(backend: str) -> None:
    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be cpu or cuda")


def build_multinomial_manifest(
    config: SyntheticErmConfig,
    backend: str,
) -> list[dict[str, Any]]:
    _require_backend(backend)
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


def build_vanilla_elastic_net_manifest(
    config: SyntheticErmConfig,
    backend: str,
) -> list[dict[str, Any]]:
    """Expand the configured unbounded elastic-net grid into self-contained jobs."""
    _require_backend(backend)
    solvers = getattr(config.elastic_net.vanilla_solvers, backend)
    jobs: list[dict[str, Any]] = []
    for shape in config.elastic_net.shapes:
        for seed in config.seeds:
            for fraction in config.elastic_net.regularization_fractions:
                problem_spec = ElasticNetSpec(
                    n=shape.n,
                    p=shape.p,
                    feature_seed=derive_seed(seed, "elastic_net_features"),
                    target_seed=derive_seed(seed, "elastic_net_targets"),
                    teacher_density=config.elastic_net.teacher_density,
                    noise_ratio=config.elastic_net.noise_ratio,
                    teacher_intercept=config.elastic_net.teacher_intercept,
                    regularization_fraction=fraction,
                )
                for solver in solvers:
                    jobs.append(
                        {
                            "suite": config.suite,
                            "problem_type": "vanilla_elastic_net",
                            "problem_id": problem_spec.problem_id(bounded=False),
                            "problem_spec": asdict(problem_spec),
                            "seed": seed,
                            "solver_seed": derive_seed(seed, "vanilla_elastic_net_solver"),
                            "solver": solver,
                            "backend": backend,
                        }
                    )
    return jobs


def build_bounded_elastic_net_manifest(
    config: SyntheticErmConfig,
    backend: str,
) -> list[dict[str, Any]]:
    """Expand the configured bounded elastic-net grid into self-contained jobs."""
    _require_backend(backend)
    solvers = getattr(config.elastic_net.bounded_solvers, backend)
    jobs: list[dict[str, Any]] = []
    for shape in config.elastic_net.shapes:
        for seed in config.seeds:
            for fraction in config.elastic_net.regularization_fractions:
                problem_spec = ElasticNetSpec(
                    n=shape.n,
                    p=shape.p,
                    feature_seed=derive_seed(seed, "elastic_net_features"),
                    target_seed=derive_seed(seed, "elastic_net_targets"),
                    teacher_density=config.elastic_net.teacher_density,
                    noise_ratio=config.elastic_net.noise_ratio,
                    teacher_intercept=config.elastic_net.teacher_intercept,
                    regularization_fraction=fraction,
                )
                for solver in solvers:
                    jobs.append(
                        {
                            "suite": config.suite,
                            "problem_type": "bounded_elastic_net",
                            "problem_id": problem_spec.problem_id(bounded=True),
                            "problem_spec": asdict(problem_spec),
                            "seed": seed,
                            "solver_seed": derive_seed(seed, "bounded_elastic_net_solver"),
                            "solver": solver,
                            "backend": backend,
                        }
                    )
    return jobs


def build_synthetic_erm_manifest(
    config: SyntheticErmConfig,
    backend: str,
) -> list[dict[str, Any]]:
    """Return every executable synthetic-ERM job."""
    return (
        build_multinomial_manifest(config, backend)
        + build_vanilla_elastic_net_manifest(config, backend)
        + build_bounded_elastic_net_manifest(config, backend)
    )

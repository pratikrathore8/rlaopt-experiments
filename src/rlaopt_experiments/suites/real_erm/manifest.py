"""Deterministic job manifests for real-data ERM production."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rlaopt_experiments.problems.real_erm import RealElasticNetSpec, RealMultinomialSpec
from rlaopt_experiments.seeds import derive_seed
from rlaopt_experiments.suites.real_erm.config import RealErmConfig


def _require_backend(backend: str) -> None:
    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be cpu or cuda")


def build_real_multinomial_manifest(
    config: RealErmConfig,
    backend: str,
) -> list[dict[str, Any]]:
    _require_backend(backend)
    jobs: list[dict[str, Any]] = []
    for dataset in config.multinomial.datasets:
        spec = RealMultinomialSpec(
            dataset=dataset,
            data_root=config.data_root,
            box_lower=config.multinomial.box_lower,
            box_upper=config.multinomial.box_upper,
        )
        for seed in config.seeds:
            for solver in getattr(config.multinomial.solvers, backend):
                jobs.append(
                    {
                        "suite": config.suite,
                        "problem_type": "multinomial",
                        "problem_id": spec.problem_id,
                        "problem_spec": asdict(spec),
                        "seed": seed,
                        "solver_seed": derive_seed(seed, "multinomial_solver"),
                        "solver": solver,
                        "backend": backend,
                    }
                )
    return jobs


def _elastic_net_manifest(
    config: RealErmConfig,
    backend: str,
    *,
    bounded: bool,
) -> list[dict[str, Any]]:
    _require_backend(backend)
    problem_type = "bounded_elastic_net" if bounded else "vanilla_elastic_net"
    solvers = config.elastic_net.bounded_solvers if bounded else config.elastic_net.vanilla_solvers
    jobs: list[dict[str, Any]] = []
    for dataset in config.elastic_net.datasets:
        for fraction in config.elastic_net.regularization_fractions:
            spec = RealElasticNetSpec(
                dataset=dataset,
                data_root=config.data_root,
                regularization_fraction=fraction,
            )
            for seed in config.seeds:
                for solver in getattr(solvers, backend):
                    jobs.append(
                        {
                            "suite": config.suite,
                            "problem_type": problem_type,
                            "problem_id": spec.problem_id(bounded=bounded),
                            "problem_spec": asdict(spec),
                            "seed": seed,
                            "solver_seed": derive_seed(seed, f"{problem_type}_solver"),
                            "solver": solver,
                            "backend": backend,
                        }
                    )
    return jobs


def build_real_vanilla_elastic_net_manifest(
    config: RealErmConfig,
    backend: str,
) -> list[dict[str, Any]]:
    return _elastic_net_manifest(config, backend, bounded=False)


def build_real_bounded_elastic_net_manifest(
    config: RealErmConfig,
    backend: str,
) -> list[dict[str, Any]]:
    return _elastic_net_manifest(config, backend, bounded=True)


def build_real_erm_manifest(config: RealErmConfig, backend: str) -> list[dict[str, Any]]:
    """Return every executable real-data ERM job for one backend."""
    jobs = (
        build_real_multinomial_manifest(config, backend)
        + build_real_vanilla_elastic_net_manifest(config, backend)
        + build_real_bounded_elastic_net_manifest(config, backend)
    )
    return [
        job for job in jobs if config.solver_subset is None or job["solver"] in config.solver_subset
    ]

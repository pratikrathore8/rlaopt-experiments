from __future__ import annotations

from pathlib import Path

import pytest

from rlaopt_experiments.suites.synthetic_erm.config import load_synthetic_erm_config
from rlaopt_experiments.manifests import build_manifest
from rlaopt_experiments.suites.synthetic_erm.manifest import (
    build_bounded_elastic_net_manifest,
    build_multinomial_manifest,
    build_synthetic_erm_manifest,
    build_vanilla_elastic_net_manifest,
)


CONFIG = Path(__file__).parents[1] / "configs" / "synthetic_erm_smoke.toml"


def test_synthetic_erm_manifest_is_complete_deterministic_and_self_contained() -> None:
    config = load_synthetic_erm_config(CONFIG)

    first = build_manifest(CONFIG, "cpu")
    second = build_synthetic_erm_manifest(config, "cpu")
    multinomial_jobs = [job for job in first if job["problem_type"] == "multinomial"]
    vanilla_jobs = [job for job in first if job["problem_type"] == "vanilla_elastic_net"]
    bounded_jobs = [job for job in first if job["problem_type"] == "bounded_elastic_net"]

    assert first == second
    assert len(first) == 90
    assert len(multinomial_jobs) == 18
    assert len(vanilla_jobs) == 36
    assert len(bounded_jobs) == 36
    assert len({job["problem_id"] for job in first}) == 30
    assert len({(job["problem_id"], job["solver"], job["backend"]) for job in first}) == len(first)
    assert {job["solver"] for job in multinomial_jobs} == set(config.multinomial.solvers.cpu)
    assert {job["solver"] for job in vanilla_jobs} == set(config.elastic_net.vanilla_solvers.cpu)
    assert {job["solver"] for job in bounded_jobs} == set(config.elastic_net.bounded_solvers.cpu)
    assert {job["backend"] for job in first} == {"cpu"}
    assert all(job["suite"] == "synthetic_erm" for job in first)
    assert all(job["problem_spec"]["feature_seed"] >= 0 for job in first)
    assert all(job["problem_spec"]["target_seed"] >= 0 for job in first)


def test_vanilla_manifest_preserves_data_across_regularization_fractions() -> None:
    config = load_synthetic_erm_config(CONFIG)
    jobs = build_vanilla_elastic_net_manifest(config, "cpu")
    selected = [
        job
        for job in jobs
        if job["seed"] == 0
        and job["solver"] == "rlaopt_sapphire"
        and job["problem_spec"]["n"] == 1024
    ]

    assert len(selected) == 2
    assert {job["problem_spec"]["regularization_fraction"] for job in selected} == {0.1, 0.01}
    assert len({job["problem_spec"]["feature_seed"] for job in selected}) == 1
    assert len({job["problem_spec"]["target_seed"] for job in selected}) == 1
    assert len({job["problem_id"] for job in selected}) == 2


def test_elastic_net_variants_share_problem_data() -> None:
    config = load_synthetic_erm_config(CONFIG)
    vanilla = build_vanilla_elastic_net_manifest(config, "cpu")
    bounded = build_bounded_elastic_net_manifest(config, "cpu")
    vanilla_job = next(
        job
        for job in vanilla
        if job["seed"] == 0
        and job["solver"] == "rlaopt_sapphire"
        and job["problem_spec"]["n"] == 1024
        and job["problem_spec"]["regularization_fraction"] == 0.1
    )
    bounded_job = next(
        job
        for job in bounded
        if job["seed"] == 0
        and job["solver"] == "rlaopt_admm"
        and job["problem_spec"]["n"] == 1024
        and job["problem_spec"]["regularization_fraction"] == 0.1
    )

    assert vanilla_job["problem_spec"] == bounded_job["problem_spec"]
    assert vanilla_job["problem_id"] != bounded_job["problem_id"]
    assert bounded_job["solver_seed"] != vanilla_job["solver_seed"]


def test_multinomial_manifest_selects_backend_solvers() -> None:
    config = load_synthetic_erm_config(CONFIG)

    jobs = build_multinomial_manifest(config, "cuda")

    assert len(jobs) == 18
    assert {job["solver"] for job in jobs} == set(config.multinomial.solvers.cuda)
    assert {job["backend"] for job in jobs} == {"cuda"}


def test_multinomial_manifest_rejects_unknown_backend() -> None:
    config = load_synthetic_erm_config(CONFIG)

    with pytest.raises(ValueError, match="backend"):
        build_multinomial_manifest(config, "tpu")

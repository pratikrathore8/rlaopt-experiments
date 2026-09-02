from __future__ import annotations

from pathlib import Path

import pytest

from rlaopt_experiments.suites.synthetic_erm.config import load_synthetic_erm_config
from rlaopt_experiments.manifests import build_manifest
from rlaopt_experiments.suites.synthetic_erm.manifest import build_multinomial_manifest


CONFIG = Path(__file__).parents[1] / "configs" / "synthetic_erm_smoke.toml"


def test_multinomial_manifest_is_complete_deterministic_and_self_contained() -> None:
    config = load_synthetic_erm_config(CONFIG)

    first = build_manifest(CONFIG, "cpu")
    second = build_multinomial_manifest(config, "cpu")

    assert first == second
    assert len(first) == 18
    assert len({job["problem_id"] for job in first}) == 6
    assert len({(job["problem_id"], job["solver"], job["backend"]) for job in first}) == len(first)
    assert {job["solver"] for job in first} == set(config.multinomial.solvers.cpu)
    assert {job["backend"] for job in first} == {"cpu"}
    assert all(job["suite"] == "synthetic_erm" for job in first)
    assert all(job["problem_type"] == "multinomial" for job in first)
    assert all(job["problem_spec"]["feature_seed"] >= 0 for job in first)
    assert all(job["problem_spec"]["target_seed"] >= 0 for job in first)


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

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlaopt_experiments.manifests import build_manifest, configured_suite
from rlaopt_experiments.suites.real_erm.config import load_real_erm_config


CONFIG = Path("configs/real_erm.toml")


def test_load_real_erm_production_config() -> None:
    config = load_real_erm_config(CONFIG)

    assert config.suite == "real_erm"
    assert config.seeds == (300,)
    assert config.repetitions == 1
    assert config.timeout_seconds == 3600
    assert config.accuracy.stationarity == 1e-4
    assert config.accuracy.feasibility == 1e-6
    assert config.elastic_net.regularization_fractions == (0.1,)
    assert set(config.multinomial.solvers.cpu) == {
        "rlaopt_sapphire",
        "projected_gradient",
        "projected_gradient_no_jit",
        "jaxopt_lbfgsb",
        "jaxopt_lbfgsb_no_jit",
    }
    for backend in ("cpu", "cuda"):
        multinomial_tolerances = config.multinomial.execution.native_tolerances
        assert multinomial_tolerances is not None
        assert multinomial_tolerances.for_solver(
            backend, "projected_gradient_no_jit"
        ) == multinomial_tolerances.for_solver(backend, "projected_gradient")
        assert multinomial_tolerances.for_solver(
            backend, "jaxopt_lbfgsb_no_jit"
        ) == multinomial_tolerances.for_solver(backend, "jaxopt_lbfgsb")
    assert config.multinomial.datasets == (
        "cifar10",
        "rcv1",
        "svhn",
        "news20",
        "fashion_mnist",
    )
    assert config.elastic_net.datasets == (
        "acsincome",
        "e2006",
        "realsim",
        "yearpredictionmsd",
        "yolanda",
    )
    assert configured_suite(CONFIG) == "real_erm"


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_real_erm_manifest_is_complete_and_deterministic(backend: str) -> None:
    first = build_manifest(CONFIG, backend)
    second = build_manifest(CONFIG, backend)

    assert first == second
    assert len(first) == 40
    assert len({json.dumps(job, sort_keys=True) for job in first}) == len(first)
    assert {job["suite"] for job in first} == {"real_erm"}
    assert {job["backend"] for job in first} == {backend}
    assert {job["seed"] for job in first} == {300}
    by_problem = {
        kind: [j for j in first if j["problem_type"] == kind]
        for kind in ("multinomial", "bounded_elastic_net")
    }
    assert len(by_problem["multinomial"]) == 25
    assert len(by_problem["bounded_elastic_net"]) == 15
    assert len({job["problem_id"] for job in by_problem["multinomial"]}) == 5
    assert len({job["problem_id"] for job in by_problem["bounded_elastic_net"]}) == 5
    assert {
        job["problem_spec"]["regularization_fraction"] for job in by_problem["bounded_elastic_net"]
    } == {0.1}
    assert all(job["problem_spec"]["data_root"] for job in first)


def test_real_erm_config_rejects_unknown_fields(tmp_path: Path) -> None:
    contents = CONFIG.read_text().replace(
        'suite = "real_erm"',
        'suite = "real_erm"\nundeclared = true',
    )
    path = tmp_path / "invalid.toml"
    path.write_text(contents)

    with pytest.raises(ValueError, match="unknown fields: undeclared"):
        load_real_erm_config(path)


def test_real_erm_config_rejects_solver_tolerance_drift(tmp_path: Path) -> None:
    contents = CONFIG.read_text().replace("projected_gradient = 1.0e-6\n", "", 1)
    path = tmp_path / "invalid.toml"
    path.write_text(contents)

    with pytest.raises(ValueError, match="solvers and native tolerances must match"):
        load_real_erm_config(path)


def test_real_erm_config_requires_both_backends_to_be_calibrated(tmp_path: Path) -> None:
    contents = CONFIG.read_text().replace(
        "[multinomial.execution.native_tolerances_calibrated]\ncpu = true",
        "[multinomial.execution.native_tolerances_calibrated]\ncpu = false",
        1,
    )
    path = tmp_path / "invalid.toml"
    path.write_text(contents)

    with pytest.raises(ValueError, match="calibrated for both backends"):
        load_real_erm_config(path)


def test_real_erm_cpu_and_cuda_manifests_use_the_same_problems() -> None:
    cpu = build_manifest(CONFIG, "cpu")
    cuda = build_manifest(CONFIG, "cuda")

    def problems(jobs: list[dict[str, object]]) -> set[tuple[object, ...]]:
        return {(job["problem_type"], job["problem_id"], job["seed"]) for job in jobs}

    assert problems(cpu) == problems(cuda)


def test_retired_settings_preserve_bounded_campaign_config(tmp_path: Path) -> None:
    source = Path("configs/real_erm.toml").read_text()
    legacy = source.replace("[accuracy]", "[accuracy]\nrelative_duality_gap = 1e-4")
    legacy += "\n[elastic_net.vanilla_solvers]\ncpu = ['retired_solver']\ncuda = []\n"
    legacy += "\n[elastic_net.vanilla_execution]\nmax_iterations = 100000\n"
    path = tmp_path / "legacy.toml"
    path.write_text(legacy)
    assert load_real_erm_config(path) == load_real_erm_config(Path("configs/real_erm.toml"))

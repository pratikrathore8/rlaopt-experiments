from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from rlaopt_experiments.execution import run_manifest_job
from rlaopt_experiments.manifests import build_manifest
from rlaopt_experiments.records import read_record
from rlaopt_experiments.suites.real_erm.config import load_real_erm_config
from rlaopt_experiments.suites.real_erm.execution import (
    run_real_bounded_elastic_net_job,
    run_real_multinomial_job,
    run_real_vanilla_elastic_net_job,
)


CONFIG = Path(__file__).parents[1] / "configs" / "real_erm.toml"


class FakeWorker:
    instances: list[FakeWorker] = []

    def __init__(
        self,
        specification: dict[str, Any],
        backend: str,
        suite: str,
    ) -> None:
        self.specification = specification
        self.backend = backend
        self.suite = suite
        self.commands: list[dict[str, Any]] = []
        self.alive = True
        self.closed = False
        self.__class__.instances.append(self)

    def wait_until_ready(self, timeout_seconds: int) -> dict[str, Any]:
        assert timeout_seconds == 1800
        return {
            "kind": "ready",
            "worker_metadata": {
                "problem_generator": "real-erm1",
                "source_sha256": "fake",
            },
        }

    def solve(self, command: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
        assert timeout_seconds == 3600
        self.commands.append(command)
        return {
            "kind": "result",
            "runtime_seconds": 1.25,
            "iterations": 8,
            "native_status": "converged",
            "native_success": True,
            "runtime_eligible": True,
            "accuracy": {
                "stationarity": 1e-5,
                "feasibility": 0.0,
                "relative_duality_gap": 1e-5,
                "objective": 0.5,
                "external_success": True,
            },
            "solver_metadata": {"native_error": 1e-7},
            "diagnostics": {"lambda_max": 2.0},
            "trace": [],
            "peak_memory_bytes": 123,
        }

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def reset_fake_workers() -> None:
    FakeWorker.instances.clear()


@pytest.mark.parametrize(
    ("problem_type", "solver", "runner", "expected_tolerance"),
    [
        ("multinomial", "jaxopt_lbfgsb", run_real_multinomial_job, 1e-6),
        (
            "vanilla_elastic_net",
            "sklearn_coordinate_descent",
            run_real_vanilla_elastic_net_job,
            1e-4,
        ),
        (
            "bounded_elastic_net",
            "clarabel_qdldl",
            run_real_bounded_elastic_net_job,
            1e-10,
        ),
    ],
)
def test_real_job_execution_records_production_controls(
    problem_type: str,
    solver: str,
    runner: Any,
    expected_tolerance: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_real_erm_config(CONFIG)
    job = next(
        item
        for item in build_manifest(CONFIG, "cpu")
        if item["problem_type"] == problem_type and item["solver"] == solver
    )
    monkeypatch.setattr(
        "rlaopt_experiments.suites.real_erm.execution.ProblemWorker",
        FakeWorker,
    )
    monkeypatch.setattr(
        "rlaopt_experiments.runner.wandb_run",
        lambda *_args, **_kwargs: nullcontext(None),
    )

    records = runner(
        job,
        config,
        native_tolerance=expected_tolerance,
        max_iterations=100_000,
        batch_size=256,
        output_dir=tmp_path,
    )

    worker = FakeWorker.instances[0]
    assert worker.closed
    assert worker.suite == "real_erm"
    assert worker.specification["problem_spec"] == job["problem_spec"]
    assert worker.specification["pre_torch_runtime"] == (
        "clarabel" if solver == "clarabel_qdldl" else None
    )
    assert len(worker.commands) == 1
    assert worker.commands[0]["native_tolerance"] == expected_tolerance
    assert worker.commands[0]["max_iters"] == 100_000
    assert records[0].suite == "real_erm"
    assert records[0].problem["dataset"] == job["problem_spec"]["dataset"]
    assert records[0].problem["n"] > 0
    assert records[0].problem["p"] > 0
    assert records[0].metadata["native_tolerances_calibrated"]
    assert records[0].metadata["accuracy_thresholds_calibrated"]
    assert records[0].metadata["source_sha256"] == "fake"
    assert (
        read_record(next((tmp_path / "records").glob("*.json")))["dataset"]
        == job["problem_spec"]["dataset"]
    )


@pytest.mark.parametrize(
    "problem_type",
    ["multinomial", "vanilla_elastic_net", "bounded_elastic_net"],
)
def test_top_level_dispatch_selects_real_runner(
    problem_type: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = next(
        item
        for item in build_manifest(CONFIG, "cpu")
        if item["problem_type"] == problem_type
    )
    captured: dict[str, Any] = {}

    def fake_run(selected_job: dict[str, Any], config: Any, **controls: Any) -> list:
        captured.update({"job": selected_job, "config": config, **controls})
        return []

    runner_name = {
        "multinomial": "run_real_multinomial_job",
        "vanilla_elastic_net": "run_real_vanilla_elastic_net_job",
        "bounded_elastic_net": "run_real_bounded_elastic_net_job",
    }[problem_type]
    monkeypatch.setattr(f"rlaopt_experiments.execution.{runner_name}", fake_run)

    assert run_manifest_job(job, CONFIG, tmp_path) == []
    assert captured["job"] == job
    assert captured["config"].suite == "real_erm"
    assert captured["max_iterations"] == 100_000
    assert captured["batch_size"] == 256
    assert captured["output_dir"] == tmp_path
    assert "record_run_key" not in captured
    assert "record_metadata" not in captured


def test_real_execution_rejects_tampered_data_root(tmp_path: Path) -> None:
    config = load_real_erm_config(CONFIG)
    job = build_manifest(CONFIG, "cpu")[0]
    tampered = job | {"problem_spec": job["problem_spec"] | {"data_root": "/tmp/wrong-data"}}

    with pytest.raises(ValueError, match="problem_spec"):
        run_real_multinomial_job(
            tampered,
            config,
            native_tolerance=1e-6,
            max_iterations=100,
            batch_size=64,
            output_dir=tmp_path,
        )


def test_real_execution_uses_distinct_record_ids_for_solver_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "three-seed-real-erm.toml"
    config_path.write_text(
        CONFIG.read_text().replace("seeds = [300]", "seeds = [300, 301, 302]")
    )
    config = load_real_erm_config(config_path)
    jobs = [
        job
        for job in build_manifest(config_path, "cpu")
        if job["problem_type"] == "multinomial"
        and job["solver"] == "jaxopt_lbfgsb"
        and job["problem_spec"]["dataset"] == "cifar10"
    ]
    monkeypatch.setattr(
        "rlaopt_experiments.suites.real_erm.execution.ProblemWorker",
        FakeWorker,
    )
    monkeypatch.setattr(
        "rlaopt_experiments.runner.wandb_run",
        lambda *_args, **_kwargs: nullcontext(None),
    )

    records = [
        run_real_multinomial_job(
            job,
            config,
            native_tolerance=1e-6,
            max_iterations=100,
            batch_size=64,
            output_dir=tmp_path,
        )[0]
        for job in jobs
    ]

    assert len(records) == 3
    assert len({record.run_id for record in records}) == 3
    assert len(list((tmp_path / "records").glob("*.json"))) == 3

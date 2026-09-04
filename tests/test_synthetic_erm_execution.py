from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rlaopt_experiments.manifests import build_manifest
from rlaopt_experiments.problems.synthetic_erm import ElasticNetSpec, MultinomialSpec
from rlaopt_experiments.records import read_record
from rlaopt_experiments.suites.synthetic_erm.config import (
    load_synthetic_erm_calibration_config,
    load_synthetic_erm_config,
)
from rlaopt_experiments.suites.synthetic_erm.execution import (
    run_bounded_elastic_net_job,
    run_multinomial_job,
    run_vanilla_elastic_net_job,
)
from rlaopt_experiments.suites.synthetic_erm.manifest import (
    build_synthetic_erm_manifest,
)


CONFIG = Path(__file__).parents[1] / "configs" / "synthetic_erm_smoke.toml"
CALIBRATION_CONFIG = Path(__file__).parents[1] / "configs" / "synthetic_erm_calibration.toml"


class FakeWorker:
    instances: list[FakeWorker] = []
    external_success = True

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
        assert timeout_seconds > 0
        return {
            "kind": "ready",
            "worker_metadata": {"problem_generator": "fake"},
        }

    def solve(
        self,
        command: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        assert timeout_seconds > 0
        self.commands.append(command)
        runtime = float(len(self.commands))
        return {
            "kind": "result",
            "runtime_seconds": runtime,
            "iterations": 7,
            "native_status": "converged",
            "native_success": True,
            "runtime_eligible": True,
            "accuracy": {
                "stationarity": 1e-8,
                "feasibility": 0.0,
                "objective": 0.5,
                "relative_duality_gap": 1e-9,
                "external_success": self.external_success,
            },
            "solver_metadata": {
                "native_error": 1e-8,
                "native_tolerance": command["native_tolerance"],
            },
            "diagnostics": {"class_counts": [10, 10, 10]},
            "trace": [],
            "peak_memory_bytes": 123,
        }

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def reset_fake_workers() -> None:
    FakeWorker.instances.clear()
    FakeWorker.external_success = True


@pytest.mark.parametrize(
    ("problem_type", "solver", "runner"),
    [
        ("multinomial", "jaxopt_lbfgsb", run_multinomial_job),
        (
            "vanilla_elastic_net",
            "sklearn_coordinate_descent",
            run_vanilla_elastic_net_job,
        ),
        ("bounded_elastic_net", "clarabel_qdldl", run_bounded_elastic_net_job),
    ],
)
def test_calibration_execution_records_unfrozen_native_tolerances(
    problem_type: str,
    solver: str,
    runner: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_synthetic_erm_calibration_config(CALIBRATION_CONFIG).experiment
    job = next(
        item
        for item in build_synthetic_erm_manifest(config, "cpu")
        if item["problem_type"] == problem_type and item["solver"] == solver
    )
    monkeypatch.setattr(
        "rlaopt_experiments.suites.synthetic_erm.execution.ProblemWorker",
        FakeWorker,
    )
    monkeypatch.setattr(
        "rlaopt_experiments.runner.wandb_run",
        lambda *_args, **_kwargs: nullcontext(None),
    )

    records = runner(
        job,
        config,
        native_tolerance=1e-6,
        max_iterations=100,
        batch_size=64,
        output_dir=tmp_path / problem_type,
    )

    assert len(records) == 1
    assert records[0].metadata["native_tolerances_calibrated"] is False


def test_multinomial_job_runs_warmup_repetitions_and_writes_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_synthetic_erm_config(CONFIG),
        warmups=1,
        repetitions=2,
    )
    job = next(job for job in build_manifest(CONFIG, "cpu") if job["solver"] == "jaxopt_lbfgsb")
    monkeypatch.setattr(
        "rlaopt_experiments.suites.synthetic_erm.execution.ProblemWorker",
        FakeWorker,
    )
    monkeypatch.setattr(
        "rlaopt_experiments.runner.wandb_run",
        lambda *_args, **_kwargs: nullcontext(None),
    )

    records = run_multinomial_job(
        job,
        config,
        native_tolerance=1e-7,
        max_iterations=500,
        batch_size=256,
        output_dir=tmp_path,
    )

    worker = FakeWorker.instances[0]
    assert worker.closed
    assert worker.specification["problem_spec"] == job["problem_spec"]
    assert [command["max_iters"] for command in worker.commands] == [10, 500, 500]
    assert len(records) == 2
    assert records[0].timings["runtime_median_seconds"] == 2.5
    assert records[0].problem["problem_type"] == "multinomial"
    assert records[0].problem["n"] == job["problem_spec"]["n"]
    assert records[0].metrics["stationarity"] == 1e-8
    assert records[0].success
    assert records[0].native_success
    assert records[0].external_success
    assert records[0].runtime_eligible
    assert records[0].metadata["native_tolerance"] == 1e-7
    assert records[0].metadata["execution_phase"] == "measurement"
    assert records[0].metadata["max_iterations"] == 500
    assert records[0].metadata["stationarity_tolerance"] == 1e-6
    assert records[0].metadata["feasibility_tolerance"] == 1e-6
    assert records[0].metadata["accuracy_thresholds_calibrated"]
    assert records[0].metadata["native_tolerances_calibrated"]
    assert "includes solver-side JIT compilation" in records[0].metadata["timing_scope"]
    paths = sorted((tmp_path / "records").glob("*.json"))
    assert len(paths) == 2
    assert read_record(paths[0])["problem_type"] == "multinomial"


def test_native_success_controls_repetitions_when_external_metric_misses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_synthetic_erm_config(CONFIG),
        warmups=0,
        repetitions=2,
    )
    job = next(job for job in build_manifest(CONFIG, "cpu") if job["solver"] == "jaxopt_lbfgsb")
    FakeWorker.external_success = False
    monkeypatch.setattr(
        "rlaopt_experiments.suites.synthetic_erm.execution.ProblemWorker",
        FakeWorker,
    )
    monkeypatch.setattr(
        "rlaopt_experiments.runner.wandb_run",
        lambda *_args, **_kwargs: nullcontext(None),
    )

    records = run_multinomial_job(
        job,
        config,
        native_tolerance=1e-7,
        max_iterations=500,
        batch_size=256,
        output_dir=tmp_path,
    )

    assert len(FakeWorker.instances[0].commands) == 2
    assert len(records) == 2
    assert all(record.native_success for record in records)
    assert all(record.runtime_eligible for record in records)
    assert not any(record.external_success for record in records)


def test_readiness_exception_closes_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_synthetic_erm_config(CONFIG)
    job = build_manifest(CONFIG, "cpu")[0]

    def fail_readiness(
        self: FakeWorker,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        raise RuntimeError("readiness failed")

    monkeypatch.setattr(FakeWorker, "wait_until_ready", fail_readiness)
    monkeypatch.setattr(
        "rlaopt_experiments.suites.synthetic_erm.execution.ProblemWorker",
        FakeWorker,
    )

    with pytest.raises(RuntimeError, match="readiness failed"):
        run_multinomial_job(
            job,
            config,
            native_tolerance=1e-6,
            max_iterations=100,
            batch_size=64,
            output_dir=tmp_path,
        )

    assert FakeWorker.instances[0].closed


def test_multinomial_job_rejects_tampered_problem_spec(tmp_path: Path) -> None:
    config = load_synthetic_erm_config(CONFIG)
    job = build_manifest(CONFIG, "cpu")[0]
    problem_spec = job["problem_spec"] | {"teacher_scale": 2.0}
    tampered = job | {
        "problem_spec": problem_spec,
        "problem_id": MultinomialSpec(**problem_spec).problem_id,
    }

    with pytest.raises(ValueError, match="problem_spec"):
        run_multinomial_job(
            tampered,
            config,
            native_tolerance=1e-6,
            max_iterations=100,
            batch_size=64,
            output_dir=tmp_path,
        )


def test_multinomial_job_rejects_tampered_problem_id(tmp_path: Path) -> None:
    config = load_synthetic_erm_config(CONFIG)
    job = build_manifest(CONFIG, "cpu")[0] | {"problem_id": "tampered"}

    with pytest.raises(ValueError, match="problem_id"):
        run_multinomial_job(
            job,
            config,
            native_tolerance=1e-6,
            max_iterations=100,
            batch_size=64,
            output_dir=tmp_path,
        )


def test_vanilla_elastic_net_job_runs_and_writes_canonical_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_synthetic_erm_config(CONFIG), warmups=1, repetitions=2)
    job = next(
        item
        for item in build_manifest(CONFIG, "cpu")
        if item["problem_type"] == "vanilla_elastic_net"
        and item["solver"] == "sklearn_coordinate_descent"
    )
    monkeypatch.setattr(
        "rlaopt_experiments.suites.synthetic_erm.execution.ProblemWorker",
        FakeWorker,
    )
    monkeypatch.setattr(
        "rlaopt_experiments.runner.wandb_run",
        lambda *_args, **_kwargs: nullcontext(None),
    )

    records = run_vanilla_elastic_net_job(
        job,
        config,
        native_tolerance=1e-7,
        max_iterations=500,
        batch_size=256,
        output_dir=tmp_path,
    )

    worker = FakeWorker.instances[0]
    assert worker.closed
    assert worker.specification["problem_type"] == "vanilla_elastic_net"
    assert [command["max_iters"] for command in worker.commands] == [10, 500, 500]
    assert len(records) == 2
    assert records[0].problem["problem_type"] == "vanilla_elastic_net"
    assert records[0].problem["regularization_fraction"] in {0.1, 0.01}
    assert records[0].metrics["relative_duality_gap"] == 1e-9
    assert records[0].metadata["relative_duality_gap_tolerance"] == 1e-6
    assert records[0].metadata["feasibility_tolerance"] == 1e-6
    assert records[0].metadata["native_tolerances_calibrated"]


def test_vanilla_elastic_net_job_rejects_tampered_fraction(tmp_path: Path) -> None:
    config = load_synthetic_erm_config(CONFIG)
    job = next(
        item
        for item in build_manifest(CONFIG, "cpu")
        if item["problem_type"] == "vanilla_elastic_net"
    )
    problem_spec = job["problem_spec"] | {"regularization_fraction": 0.5}
    tampered = job | {
        "problem_spec": problem_spec,
        "problem_id": ElasticNetSpec(**problem_spec).problem_id(bounded=False),
    }

    with pytest.raises(ValueError, match="regularization fraction"):
        run_vanilla_elastic_net_job(
            tampered,
            config,
            native_tolerance=1e-6,
            max_iterations=100,
            batch_size=64,
            output_dir=tmp_path,
        )


def test_bounded_elastic_net_job_runs_and_requests_julia_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_synthetic_erm_config(CONFIG), warmups=1, repetitions=2)
    job = next(
        item
        for item in build_manifest(CONFIG, "cpu")
        if item["problem_type"] == "bounded_elastic_net" and item["solver"] == "clarabel_qdldl"
    )
    monkeypatch.setattr(
        "rlaopt_experiments.suites.synthetic_erm.execution.ProblemWorker",
        FakeWorker,
    )
    monkeypatch.setattr(
        "rlaopt_experiments.runner.wandb_run",
        lambda *_args, **_kwargs: nullcontext(None),
    )

    records = run_bounded_elastic_net_job(
        job,
        config,
        native_tolerance=1e-7,
        max_iterations=500,
        batch_size=256,
        output_dir=tmp_path,
    )

    worker = FakeWorker.instances[0]
    assert worker.closed
    assert worker.specification["problem_type"] == "bounded_elastic_net"
    assert worker.specification["pre_torch_runtime"] == "clarabel"
    assert [command["max_iters"] for command in worker.commands] == [10, 500, 500]
    assert len(records) == 2
    assert records[0].problem["problem_type"] == "bounded_elastic_net"
    assert records[0].metrics["stationarity"] == 1e-8
    assert records[0].metrics["feasibility"] == 0.0
    assert records[0].metadata["stationarity_tolerance"] == 1e-6
    assert records[0].metadata["feasibility_tolerance"] == 1e-6
    assert records[0].metadata["native_tolerances_calibrated"]
    assert "conic construction and format conversion" in records[0].metadata["timing_scope"]
    assert "includes solver-side JIT compilation" in records[0].metadata["timing_scope"]


def test_bounded_elastic_net_job_rejects_vanilla_problem_id(tmp_path: Path) -> None:
    config = load_synthetic_erm_config(CONFIG)
    job = next(
        item
        for item in build_manifest(CONFIG, "cpu")
        if item["problem_type"] == "bounded_elastic_net"
    )
    tampered = job | {"problem_id": ElasticNetSpec(**job["problem_spec"]).problem_id(bounded=False)}

    with pytest.raises(ValueError, match="problem_id"):
        run_bounded_elastic_net_job(
            tampered,
            config,
            native_tolerance=1e-6,
            max_iterations=100,
            batch_size=64,
            output_dir=tmp_path,
        )

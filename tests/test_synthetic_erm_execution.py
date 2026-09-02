from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rlaopt_experiments.manifests import build_manifest
from rlaopt_experiments.problems.synthetic_erm import MultinomialSpec
from rlaopt_experiments.records import read_record
from rlaopt_experiments.suites.synthetic_erm.config import load_synthetic_erm_config
from rlaopt_experiments.suites.synthetic_erm.execution import run_multinomial_job


CONFIG = Path(__file__).parents[1] / "configs" / "synthetic_erm_smoke.toml"


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
            "accuracy": {
                "stationarity": 1e-8,
                "feasibility": 0.0,
                "objective": 0.5,
                "success": True,
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
    assert records[0].metadata["native_tolerance"] == 1e-7
    assert records[0].metadata["execution_phase"] == "measurement"
    assert records[0].metadata["max_iterations"] == 500
    assert records[0].metadata["stationarity_tolerance"] == 1e-6
    assert records[0].metadata["feasibility_tolerance"] == 1e-8
    assert not records[0].metadata["accuracy_thresholds_calibrated"]
    assert "JIT compilation" in records[0].metadata["timing_scope"]
    paths = sorted((tmp_path / "records").glob("*.json"))
    assert len(paths) == 2
    assert read_record(paths[0])["problem_type"] == "multinomial"


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

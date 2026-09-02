from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rlaopt_experiments.execution import read_manifest_job, run_manifest_job
from rlaopt_experiments.manifests import build_manifest


CONFIG = Path(__file__).parents[1] / "configs" / "synthetic_erm_smoke.toml"


def test_read_manifest_job_uses_zero_based_index(tmp_path: Path) -> None:
    jobs = build_manifest(CONFIG, "cpu")
    path = tmp_path / "jobs.jsonl"
    path.write_text("\n".join(json.dumps(job) for job in jobs) + "\n")

    assert read_manifest_job(path, 1) == jobs[1]
    with pytest.raises(ValueError, match="nonnegative"):
        read_manifest_job(path, -1)
    with pytest.raises(IndexError, match="out of range"):
        read_manifest_job(path, len(jobs))


def test_run_manifest_job_uses_strict_configured_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = build_manifest(CONFIG, "cpu")[0]
    captured: dict[str, Any] = {}

    def fake_run(
        selected_job: dict[str, Any],
        config: Any,
        **controls: Any,
    ) -> list:
        captured.update(
            {
                "job": selected_job,
                "config": config,
                **controls,
            }
        )
        return []

    monkeypatch.setattr(
        "rlaopt_experiments.execution.run_multinomial_job",
        fake_run,
    )

    assert run_manifest_job(job, CONFIG, tmp_path) == []
    assert captured["job"] == job
    assert captured["native_tolerance"] == 1e-6
    assert captured["max_iterations"] == 10_000
    assert captured["batch_size"] == 256
    assert captured["output_dir"] == tmp_path
    assert not captured["config"].multinomial.execution.native_tolerances_calibrated

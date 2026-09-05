from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from rlaopt_experiments.suites.real_erm.production import plan_real_production


CONFIG = Path(__file__).parents[1] / "configs" / "real_erm.toml"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_real_production_plan_is_complete_balanced_and_qos_bounded(tmp_path: Path) -> None:
    root = tmp_path / "production"
    plan = plan_real_production(CONFIG, root)

    assert plan["problem_types"] == ["bounded_elastic_net", "multinomial"]
    assert plan["jobs"] == {
        "cpu-soal-8": 20,
        "cpu-soal-9": 20,
        "cuda-soal-12": 40,
    }
    assert plan["batches"] == {
        "cpu-soal-8": 3,
        "cpu-soal-9": 3,
        "cuda-soal-12": 6,
    }
    assert [wave["tasks"] for wave in plan["waves"]] == [12]
    assert all(wave["tasks"] <= 18 for wave in plan["waves"])
    assert (root / "config.toml").read_bytes() == CONFIG.read_bytes()
    assert (root / "config.toml.sha256").is_file()

    cpu_master = _read_jsonl(root / "manifests" / "cpu.jsonl")
    cuda_master = _read_jsonl(root / "manifests" / "cuda.jsonl")
    reconstructed: dict[str, list[dict]] = {}
    for target in plan["batches"]:
        batches = sorted((root / "batches" / target).glob("*.jsonl"))
        assert all(1 <= len(_read_jsonl(path)) <= 7 for path in batches)
        assert all(path.with_suffix(".jsonl.sha256").is_file() for path in batches)
        reconstructed[target] = [job for path in batches for job in _read_jsonl(path)]

    assert Counter(
        json.dumps(job, sort_keys=True)
        for job in reconstructed["cpu-soal-8"] + reconstructed["cpu-soal-9"]
    ) == Counter(json.dumps(job, sort_keys=True) for job in cpu_master)
    assert reconstructed["cuda-soal-12"] == cuda_master

    cpu_solver_counts = {
        target: Counter((job["problem_type"], job["solver"]) for job in jobs)
        for target, jobs in reconstructed.items()
        if target.startswith("cpu-")
    }
    for key in set(cpu_solver_counts["cpu-soal-8"]) | set(cpu_solver_counts["cpu-soal-9"]):
        assert abs(cpu_solver_counts["cpu-soal-8"][key] - cpu_solver_counts["cpu-soal-9"][key]) <= 1


def test_real_production_plan_refuses_nonempty_destination(tmp_path: Path) -> None:
    root = tmp_path / "production"
    root.mkdir()
    (root / "stale.txt").write_text("stale")

    with pytest.raises(FileExistsError, match="not empty"):
        plan_real_production(CONFIG, root)


@pytest.mark.parametrize(
    ("max_jobs", "max_batches"),
    [(0, 6), (11, 6), (10, 0), (10, 7)],
)
def test_real_production_plan_enforces_safe_limits(
    tmp_path: Path,
    max_jobs: int,
    max_batches: int,
) -> None:
    with pytest.raises(ValueError):
        plan_real_production(
            CONFIG,
            tmp_path / f"production-{max_jobs}-{max_batches}",
            max_jobs_per_batch=max_jobs,
            max_batches_per_node_per_wave=max_batches,
        )

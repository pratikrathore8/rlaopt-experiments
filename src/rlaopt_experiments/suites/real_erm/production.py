"""Deterministic, QOS-bounded production batching for real-data ERM."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from rlaopt_experiments.manifests import build_manifest


PRODUCTION_TARGETS = (
    ("cpu-soal-8", "cpu"),
    ("cpu-soal-9", "cpu"),
    ("cuda-soal-12", "cuda"),
)
MAX_SAFE_JOBS_PER_BATCH = 10
MAX_SAFE_BATCHES_PER_NODE_PER_WAVE = 6


def shard_real_cpu_jobs(
    jobs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Spread every problem/seed solver group deterministically across two CPU nodes."""
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    order: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for job in jobs:
        if job.get("backend") != "cpu":
            raise ValueError("real CPU sharding requires CPU manifest jobs")
        serialized = json.dumps(job, sort_keys=True)
        if serialized in seen:
            raise ValueError("real CPU manifest contains duplicate jobs")
        seen.add(serialized)
        key = (job["problem_type"], job["problem_id"], job["seed"])
        if key not in groups:
            order.append(key)
        groups[key].append(job)

    shards: tuple[list[dict[str, Any]], list[dict[str, Any]]] = ([], [])
    for group_index, key in enumerate(order):
        for solver_index, job in enumerate(groups[key]):
            shards[(group_index + solver_index) % 2].append(job)
    return shards


def _write_jsonl(path: Path, jobs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(job, sort_keys=True)}\n" for job in jobs))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(f"{path.suffix}.sha256").write_text(f"{digest}  {path.resolve()}\n")


def _batches(
    jobs: list[dict[str, Any]],
    directory: Path,
    max_jobs: int,
) -> list[Path]:
    paths: list[Path] = []
    for index in range(0, len(jobs), max_jobs):
        path = directory / f"batch-{index // max_jobs:03d}.jsonl"
        _write_jsonl(path, jobs[index : index + max_jobs])
        paths.append(path.resolve())
    return paths


def plan_real_production(
    config_path: Path,
    output_dir: Path,
    *,
    max_jobs_per_batch: int = 10,
    max_batches_per_node_per_wave: int = 6,
) -> dict[str, Any]:
    """Write immutable manifests, bounded batches, wave lists, and plan metadata."""
    if not 1 <= max_jobs_per_batch <= MAX_SAFE_JOBS_PER_BATCH:
        raise ValueError(f"max_jobs_per_batch must be between 1 and {MAX_SAFE_JOBS_PER_BATCH}")
    if not 1 <= max_batches_per_node_per_wave <= MAX_SAFE_BATCHES_PER_NODE_PER_WAVE:
        raise ValueError(
            "max_batches_per_node_per_wave must be between 1 and "
            f"{MAX_SAFE_BATCHES_PER_NODE_PER_WAVE}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"production plan directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_bytes = config_path.read_bytes()
    frozen_config = output_dir / "config.toml"
    frozen_config.write_bytes(config_bytes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    (output_dir / "config.toml.sha256").write_text(f"{config_sha256}  {frozen_config.resolve()}\n")

    cpu = build_manifest(frozen_config, "cpu")
    cuda = build_manifest(frozen_config, "cuda")
    cpu_8, cpu_9 = shard_real_cpu_jobs(cpu)
    target_jobs = {
        "cpu-soal-8": cpu_8,
        "cpu-soal-9": cpu_9,
        "cuda-soal-12": cuda,
    }
    _write_jsonl(output_dir / "manifests" / "cpu.jsonl", cpu)
    _write_jsonl(output_dir / "manifests" / "cuda.jsonl", cuda)

    target_batches = {
        target: _batches(
            target_jobs[target],
            output_dir / "batches" / target,
            max_jobs_per_batch,
        )
        for target, _ in PRODUCTION_TARGETS
    }
    wave_count = max(
        math.ceil(len(paths) / max_batches_per_node_per_wave) for paths in target_batches.values()
    )
    waves: list[dict[str, Any]] = []
    for wave_index in range(wave_count):
        wave_dir = output_dir / "waves" / f"wave-{wave_index:03d}"
        task_count = 0
        target_counts: dict[str, int] = {}
        start = wave_index * max_batches_per_node_per_wave
        stop = start + max_batches_per_node_per_wave
        for target, _ in PRODUCTION_TARGETS:
            selected = target_batches[target][start:stop]
            target_counts[target] = len(selected)
            task_count += len(selected)
            if selected:
                wave_dir.mkdir(parents=True, exist_ok=True)
                (wave_dir / f"{target}.txt").write_text("".join(f"{path}\n" for path in selected))
        waves.append(
            {
                "index": wave_index,
                "tasks": task_count,
                "targets": target_counts,
            }
        )

    plan = {
        "schema_version": 1,
        "config": str(frozen_config.resolve()),
        "config_sha256": config_sha256,
        "max_jobs_per_batch": max_jobs_per_batch,
        "max_batches_per_node_per_wave": max_batches_per_node_per_wave,
        "max_tasks_per_wave": len(PRODUCTION_TARGETS) * max_batches_per_node_per_wave,
        "jobs": {target: len(jobs) for target, jobs in target_jobs.items()},
        "batches": {target: len(paths) for target, paths in target_batches.items()},
        "waves": waves,
    }
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan

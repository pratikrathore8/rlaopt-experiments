"""Versioned JSON records with backward-compatible flattened views."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3

_LEGACY_NATIVE_SUCCESS_STATUSES = {
    "converged",
    "direct",
    "istop_1",
    "istop_2",
    "native_complete",
    "native_converged",
}


@dataclass
class TrialRecord:
    """Suite-neutral persisted result for one measured solver invocation."""

    run_id: str
    problem_id: str
    suite: str
    solver: str
    backend: str
    seed: int
    repetition: int
    timings: dict[str, float]
    iterations: int | None
    native_status: str
    metrics: dict[str, float | bool | None]
    native_success: bool
    external_success: bool
    runtime_eligible: bool
    timed_out: bool
    problem: dict[str, Any] = field(default_factory=dict)
    peak_memory_bytes: int | None = None
    trace: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = field(default=SCHEMA_VERSION, init=False)

    @property
    def runtime_seconds(self) -> float:
        return self.timings["runtime_seconds"]

    @property
    def n(self) -> int:
        return self.problem["n"]

    @property
    def p(self) -> int:
        return self.problem["p"]

    @property
    def alpha(self) -> float:
        return self.problem["alpha"]

    @property
    def ridge(self) -> float:
        return self.problem["ridge"]

    @property
    def relative_kkt(self) -> float | None:
        return self.metrics.get("relative_kkt")

    @property
    def relative_solution_error(self) -> float | None:
        return self.metrics.get("relative_solution_error")

    @property
    def success(self) -> bool:
        """Backward-compatible alias for external mathematical success."""
        return self.external_success


def _infer_legacy_native_success(data: dict[str, Any]) -> bool:
    worker_outcome = data.get("metadata", {}).get("worker_outcome")
    if worker_outcome is not None and worker_outcome != "result":
        return False
    return data.get("native_status") in _LEGACY_NATIVE_SUCCESS_STATUSES


def _add_outcome_semantics(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize old records while retaining success as an analysis alias."""
    external_success = bool(data.get("external_success", data.get("success", False)))
    native_success = bool(data.get("native_success", _infer_legacy_native_success(data)))
    runtime_eligible = bool(data.get("runtime_eligible", native_success))
    return data | {
        "native_success": native_success,
        "external_success": external_success,
        "runtime_eligible": runtime_eligible,
        "success": external_success,
    }


def record_view(data: dict[str, Any]) -> dict[str, Any]:
    """Return a flat analysis view for either a legacy or current record."""
    version = data.get("schema_version", 1)
    if version == 1:
        return _add_outcome_semantics({"suite": "synthetic_ridge", **data})
    if version not in {2, SCHEMA_VERSION}:
        raise ValueError(f"unsupported record schema version: {data['schema_version']}")
    if version == SCHEMA_VERSION:
        required = {"native_success", "external_success", "runtime_eligible"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"schema-v3 record is missing outcome fields: {sorted(missing)}")

    problem = dict(data.get("problem", {}))
    diagnostics = problem.pop("diagnostics", {})
    return _add_outcome_semantics(
        {
            **data,
            **problem,
            **data.get("timings", {}),
            **data.get("metrics", {}),
            "diagnostics": diagnostics,
        }
    )


def read_record(path: Path) -> dict[str, Any]:
    """Read one persisted record and normalize it for analysis."""
    return record_view(json.loads(path.read_text()))


def write_record(path: Path, record: TrialRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(asdict(record), output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

"""Stable JSON record schema and atomic local persistence."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrialRecord:
    run_id: str
    problem_id: str
    solver: str
    backend: str
    n: int
    p: int
    alpha: float
    ridge: float
    seed: int
    repetition: int
    runtime_seconds: float
    iterations: int | None
    native_status: str
    relative_kkt: float | None
    relative_solution_error: float | None
    success: bool
    timed_out: bool
    peak_memory_bytes: int | None = None
    trace: list[dict[str, float]] = field(default_factory=list)
    diagnostics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


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

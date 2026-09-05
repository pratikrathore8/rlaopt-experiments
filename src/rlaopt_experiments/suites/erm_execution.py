"""Execution lifecycle shared by synthetic and real ERM suites."""

from __future__ import annotations

import math
from typing import Any, Callable, Protocol

from rlaopt_experiments.isolation import ProblemWorker
from rlaopt_experiments.records import TrialRecord


class ErmExecutionConfig(Protocol):
    warmups: int
    repetitions: int
    timeout_seconds: int


def validate_controls(
    native_tolerance: float,
    max_iterations: int,
    batch_size: int,
) -> None:
    if not math.isfinite(native_tolerance) or native_tolerance <= 0:
        raise ValueError("native_tolerance must be finite and positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")


def run_repetitions(
    *,
    worker: ProblemWorker,
    ready: dict[str, Any],
    config: ErmExecutionConfig,
    command: dict[str, Any],
    max_iterations: int,
    persist: Callable[..., TrialRecord],
) -> list[TrialRecord]:
    """Apply the common warmup, repetition, and worker-lifecycle protocol."""
    try:
        if ready["kind"] != "ready":
            outcome = ready | {"runtime_seconds": 0.0}
            return [
                persist(
                    outcome,
                    repetition=0,
                    runtimes=[0.0],
                    phase="startup",
                    iteration_limit=None,
                )
            ]

        for _ in range(config.warmups):
            warmup_limit = min(max_iterations, 10)
            warmup = worker.solve(
                command | {"max_iters": warmup_limit},
                config.timeout_seconds,
            )
            if warmup["kind"] != "result":
                return [
                    persist(
                        warmup,
                        repetition=0,
                        runtimes=[warmup["runtime_seconds"]],
                        phase="warmup",
                        iteration_limit=warmup_limit,
                    )
                ]

        first = worker.solve(command, config.timeout_seconds)
        outcomes = [first]
        if first["kind"] == "result" and first["runtime_eligible"]:
            for _ in range(config.repetitions - 1):
                repeated = worker.solve(command, config.timeout_seconds)
                outcomes.append(repeated)
                if not worker.alive:
                    break
        runtimes = [outcome["runtime_seconds"] for outcome in outcomes]
        return [
            persist(
                outcome,
                repetition=repetition,
                runtimes=runtimes,
                phase="measurement",
                iteration_limit=max_iterations,
            )
            for repetition, outcome in enumerate(outcomes)
        ]
    finally:
        worker.close()

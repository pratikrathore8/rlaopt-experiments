"""Isolated execution for self-contained synthetic ERM manifest jobs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from rlaopt_experiments.isolation import ProblemWorker
from rlaopt_experiments.problems.synthetic_erm import ElasticNetSpec, MultinomialSpec
from rlaopt_experiments.records import TrialRecord
from rlaopt_experiments.runner import record_outcome
from rlaopt_experiments.seeds import derive_seed
from rlaopt_experiments.suites.synthetic_erm.config import SyntheticErmConfig


def _validate_controls(
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


def _run_repetitions(
    *,
    worker: ProblemWorker,
    ready: dict[str, Any],
    config: SyntheticErmConfig,
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


def _validate_multinomial_job(
    job: dict[str, Any],
    config: SyntheticErmConfig,
) -> MultinomialSpec:
    if job.get("suite") != config.suite:
        raise ValueError("manifest job suite does not match the configuration")
    if job.get("problem_type") != "multinomial":
        raise ValueError("execution requires a multinomial manifest job")
    backend = job.get("backend")
    if backend not in {"cpu", "cuda"}:
        raise ValueError("manifest job backend must be 'cpu' or 'cuda'")
    if job.get("solver") not in getattr(config.multinomial.solvers, backend):
        raise ValueError("manifest solver is not configured for its backend")
    spec = MultinomialSpec(**job["problem_spec"])
    seed = job.get("seed")
    if not isinstance(seed, int) or seed not in config.seeds:
        raise ValueError("manifest seed is not configured")
    if (spec.n, spec.p) not in {(shape.n, shape.p) for shape in config.multinomial.shapes}:
        raise ValueError("manifest shape is not configured")
    expected_spec = MultinomialSpec(
        n=spec.n,
        p=spec.p,
        n_classes=config.multinomial.n_classes,
        feature_seed=derive_seed(seed, "multinomial_features"),
        target_seed=derive_seed(seed, "multinomial_targets"),
        teacher_scale=config.multinomial.teacher_scale,
        box_lower=config.multinomial.box_lower,
        box_upper=config.multinomial.box_upper,
    )
    if spec != expected_spec:
        raise ValueError("manifest problem_spec does not match the configuration")
    if job.get("problem_id") != spec.problem_id:
        raise ValueError("manifest problem_id does not match problem_spec")
    if job.get("solver_seed") != derive_seed(seed, "multinomial_solver"):
        raise ValueError("manifest solver_seed does not match its master seed")
    return spec


def run_multinomial_job(
    job: dict[str, Any],
    config: SyntheticErmConfig,
    *,
    native_tolerance: float,
    max_iterations: int,
    batch_size: int,
    output_dir: Path,
    record_run_key: str = "multinomial",
    record_metadata: dict[str, Any] | None = None,
) -> list[TrialRecord]:
    """Execute one multinomial manifest job through an isolated worker."""
    _validate_controls(native_tolerance, max_iterations, batch_size)
    spec = _validate_multinomial_job(job, config)
    backend = job["backend"]
    solver = job["solver"]
    specification = {
        "problem_type": "multinomial",
        "problem_spec": job["problem_spec"],
    }
    command = {
        "solver": solver,
        "native_tolerance": native_tolerance,
        "max_iters": max_iterations,
        "batch_size": batch_size,
        "solver_seed": job["solver_seed"],
        "stationarity_tolerance": config.accuracy.stationarity,
        "feasibility_tolerance": config.accuracy.feasibility,
    }
    problem_fields = {
        "problem_type": "multinomial",
        "n": spec.n,
        "p": spec.p,
        "n_classes": spec.n_classes,
        "teacher_scale": spec.teacher_scale,
        "box_lower": spec.box_lower,
        "box_upper": spec.box_upper,
        "feature_seed": spec.feature_seed,
        "target_seed": spec.target_seed,
    }

    worker = ProblemWorker(specification, backend, config.suite)
    try:
        ready = worker.wait_until_ready(config.startup_timeout_seconds)
    except BaseException:
        worker.close()
        raise

    def persist(
        outcome: dict[str, Any],
        repetition: int,
        runtimes: list[float],
        *,
        phase: str,
        iteration_limit: int | None,
    ) -> TrialRecord:
        metadata = (
            outcome.get("solver_metadata", {})
            | {
                "execution_phase": phase,
                "native_tolerance": native_tolerance,
                "max_iterations": iteration_limit,
                "solver_seed": job["solver_seed"],
                "stationarity_tolerance": config.accuracy.stationarity,
                "feasibility_tolerance": config.accuracy.feasibility,
                "accuracy_thresholds_calibrated": config.accuracy.calibrated,
                "native_tolerances_calibrated": (
                    config.multinomial.execution.tolerances_calibrated_for(backend)
                ),
            }
            | (record_metadata or {})
        )
        annotated = outcome | {"solver_metadata": metadata}
        return record_outcome(
            suite=config.suite,
            problem_id=spec.problem_id,
            run_key=record_run_key,
            problem=problem_fields,
            solver=solver,
            backend=backend,
            seed=job["seed"],
            repetition=repetition,
            outcome=annotated,
            runtimes=runtimes,
            output_dir=output_dir,
            worker_metadata=ready.get("worker_metadata", {}),
            timing_scope=(
                "device-resident solver invocation; excludes problem generation, "
                "worker startup, format conversion, and JIT compilation"
            ),
        )

    return _run_repetitions(
        worker=worker,
        ready=ready,
        config=config,
        command=command,
        max_iterations=max_iterations,
        persist=persist,
    )


def _validate_elastic_net_job(
    job: dict[str, Any],
    config: SyntheticErmConfig,
    *,
    bounded: bool,
) -> ElasticNetSpec:
    problem_type = "bounded_elastic_net" if bounded else "vanilla_elastic_net"
    variant = "bounded" if bounded else "vanilla"
    if job.get("suite") != config.suite:
        raise ValueError("manifest job suite does not match the configuration")
    if job.get("problem_type") != problem_type:
        raise ValueError(f"execution requires a {variant} elastic-net manifest job")
    backend = job.get("backend")
    if backend not in {"cpu", "cuda"}:
        raise ValueError("manifest job backend must be cpu or cuda")
    solvers = config.elastic_net.bounded_solvers if bounded else config.elastic_net.vanilla_solvers
    if job.get("solver") not in getattr(solvers, backend):
        raise ValueError("manifest solver is not configured for its backend")
    spec = ElasticNetSpec(**job["problem_spec"])
    seed = job.get("seed")
    if not isinstance(seed, int) or seed not in config.seeds:
        raise ValueError("manifest seed is not configured")
    if (spec.n, spec.p) not in {(shape.n, shape.p) for shape in config.elastic_net.shapes}:
        raise ValueError("manifest shape is not configured")
    if spec.regularization_fraction not in config.elastic_net.regularization_fractions:
        raise ValueError("manifest regularization fraction is not configured")
    expected_spec = ElasticNetSpec(
        n=spec.n,
        p=spec.p,
        feature_seed=derive_seed(seed, "elastic_net_features"),
        target_seed=derive_seed(seed, "elastic_net_targets"),
        teacher_density=config.elastic_net.teacher_density,
        noise_ratio=config.elastic_net.noise_ratio,
        teacher_intercept=config.elastic_net.teacher_intercept,
        regularization_fraction=spec.regularization_fraction,
    )
    if spec != expected_spec:
        raise ValueError("manifest problem_spec does not match the configuration")
    if job.get("problem_id") != spec.problem_id(bounded=bounded):
        raise ValueError("manifest problem_id does not match problem_spec")
    solver_seed_name = "bounded_elastic_net_solver" if bounded else "vanilla_elastic_net_solver"
    if job.get("solver_seed") != derive_seed(seed, solver_seed_name):
        raise ValueError("manifest solver_seed does not match its master seed")
    return spec


def run_vanilla_elastic_net_job(
    job: dict[str, Any],
    config: SyntheticErmConfig,
    *,
    native_tolerance: float,
    max_iterations: int,
    batch_size: int,
    output_dir: Path,
    record_run_key: str = "vanilla_elastic_net",
    record_metadata: dict[str, Any] | None = None,
) -> list[TrialRecord]:
    """Execute one vanilla elastic-net manifest job through an isolated worker."""
    _validate_controls(native_tolerance, max_iterations, batch_size)
    spec = _validate_elastic_net_job(job, config, bounded=False)
    backend = job["backend"]
    solver = job["solver"]
    specification = {
        "problem_type": "vanilla_elastic_net",
        "problem_spec": job["problem_spec"],
    }
    command = {
        "solver": solver,
        "native_tolerance": native_tolerance,
        "max_iters": max_iterations,
        "batch_size": batch_size,
        "solver_seed": job["solver_seed"],
        "relative_duality_gap_tolerance": config.accuracy.relative_duality_gap,
        "feasibility_tolerance": config.accuracy.feasibility,
    }
    problem_fields = {
        "problem_type": "vanilla_elastic_net",
        "n": spec.n,
        "p": spec.p,
        "teacher_density": spec.teacher_density,
        "noise_ratio": spec.noise_ratio,
        "teacher_intercept": spec.teacher_intercept,
        "regularization_fraction": spec.regularization_fraction,
        "feature_seed": spec.feature_seed,
        "target_seed": spec.target_seed,
    }

    worker = ProblemWorker(specification, backend, config.suite)
    try:
        ready = worker.wait_until_ready(config.startup_timeout_seconds)
    except BaseException:
        worker.close()
        raise

    def persist(
        outcome: dict[str, Any],
        repetition: int,
        runtimes: list[float],
        *,
        phase: str,
        iteration_limit: int | None,
    ) -> TrialRecord:
        metadata = (
            outcome.get("solver_metadata", {})
            | {
                "execution_phase": phase,
                "native_tolerance": native_tolerance,
                "max_iterations": iteration_limit,
                "solver_seed": job["solver_seed"],
                "relative_duality_gap_tolerance": config.accuracy.relative_duality_gap,
                "feasibility_tolerance": config.accuracy.feasibility,
                "accuracy_thresholds_calibrated": config.accuracy.calibrated,
                "native_tolerances_calibrated": (
                    config.elastic_net.vanilla_execution.tolerances_calibrated_for(backend)
                ),
            }
            | (record_metadata or {})
        )
        return record_outcome(
            suite=config.suite,
            problem_id=spec.problem_id(bounded=False),
            run_key=record_run_key,
            problem=problem_fields,
            solver=solver,
            backend=backend,
            seed=job["seed"],
            repetition=repetition,
            outcome=outcome | {"solver_metadata": metadata},
            runtimes=runtimes,
            output_dir=output_dir,
            worker_metadata=ready.get("worker_metadata", {}),
            timing_scope=(
                "device-resident solver invocation; excludes problem generation, "
                "worker startup, format conversion, and JIT compilation"
            ),
        )

    return _run_repetitions(
        worker=worker,
        ready=ready,
        config=config,
        command=command,
        max_iterations=max_iterations,
        persist=persist,
    )


def run_bounded_elastic_net_job(
    job: dict[str, Any],
    config: SyntheticErmConfig,
    *,
    native_tolerance: float,
    max_iterations: int,
    batch_size: int,
    output_dir: Path,
    record_run_key: str = "bounded_elastic_net",
    record_metadata: dict[str, Any] | None = None,
) -> list[TrialRecord]:
    """Execute one bounded elastic-net manifest job through an isolated worker."""
    _validate_controls(native_tolerance, max_iterations, batch_size)
    spec = _validate_elastic_net_job(job, config, bounded=True)
    backend = job["backend"]
    solver = job["solver"]
    clarabel_solvers = {"clarabel_qdldl", "cuclarabel_cudss"}
    specification = {
        "problem_type": "bounded_elastic_net",
        "problem_spec": job["problem_spec"],
        "pre_torch_runtime": "clarabel" if solver in clarabel_solvers else None,
    }
    command = {
        "solver": solver,
        "native_tolerance": native_tolerance,
        "max_iters": max_iterations,
        "batch_size": batch_size,
        "solver_seed": job["solver_seed"],
        "stationarity_tolerance": config.accuracy.stationarity,
        "feasibility_tolerance": config.accuracy.feasibility,
    }
    problem_fields = {
        "problem_type": "bounded_elastic_net",
        "n": spec.n,
        "p": spec.p,
        "teacher_density": spec.teacher_density,
        "noise_ratio": spec.noise_ratio,
        "teacher_intercept": spec.teacher_intercept,
        "regularization_fraction": spec.regularization_fraction,
        "feature_seed": spec.feature_seed,
        "target_seed": spec.target_seed,
    }

    worker = ProblemWorker(specification, backend, config.suite)
    try:
        ready = worker.wait_until_ready(config.startup_timeout_seconds)
    except BaseException:
        worker.close()
        raise

    def persist(
        outcome: dict[str, Any],
        repetition: int,
        runtimes: list[float],
        *,
        phase: str,
        iteration_limit: int | None,
    ) -> TrialRecord:
        metadata = (
            outcome.get("solver_metadata", {})
            | {
                "execution_phase": phase,
                "native_tolerance": native_tolerance,
                "max_iterations": iteration_limit,
                "solver_seed": job["solver_seed"],
                "stationarity_tolerance": config.accuracy.stationarity,
                "feasibility_tolerance": config.accuracy.feasibility,
                "accuracy_thresholds_calibrated": config.accuracy.calibrated,
                "native_tolerances_calibrated": (
                    config.elastic_net.bounded_execution.tolerances_calibrated_for(backend)
                ),
            }
            | (record_metadata or {})
        )
        return record_outcome(
            suite=config.suite,
            problem_id=spec.problem_id(bounded=True),
            run_key=record_run_key,
            problem=problem_fields,
            solver=solver,
            backend=backend,
            seed=job["seed"],
            repetition=repetition,
            outcome=outcome | {"solver_metadata": metadata},
            runtimes=runtimes,
            output_dir=output_dir,
            worker_metadata=ready.get("worker_metadata", {}),
            timing_scope=(
                "native solver invocation; excludes problem generation, worker startup, "
                "benchmark-side conic construction and format conversion, and JIT compilation; "
                "includes solver-internal setup and transfers"
            ),
        )

    return _run_repetitions(
        worker=worker,
        ready=ready,
        config=config,
        command=command,
        max_iterations=max_iterations,
        persist=persist,
    )

"""Isolated execution for self-contained real-data ERM manifest jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlaopt_experiments.isolation import ProblemWorker
from rlaopt_experiments.problems.real_erm import RealElasticNetSpec, RealMultinomialSpec
from rlaopt_experiments.records import TrialRecord
from rlaopt_experiments.runner import record_outcome
from rlaopt_experiments.seeds import derive_seed
from rlaopt_experiments.suites.erm_execution import run_repetitions, validate_controls
from rlaopt_experiments.suites.real_erm.config import RealErmConfig
from rlaopt_experiments.suites.synthetic_erm.config import BackendSolvers, SolverExecution


def _validate_common_job(
    job: dict[str, Any],
    config: RealErmConfig,
    problem_type: str,
    solvers: BackendSolvers,
) -> tuple[str, str, int]:
    if job.get("suite") != config.suite:
        raise ValueError("manifest job suite does not match the configuration")
    if job.get("problem_type") != problem_type:
        raise ValueError(f"execution requires a {problem_type} manifest job")
    backend = job.get("backend")
    if backend not in {"cpu", "cuda"}:
        raise ValueError("manifest job backend must be cpu or cuda")
    solver = job.get("solver")
    if solver not in getattr(solvers, backend):
        raise ValueError("manifest solver is not configured for its backend")
    seed = job.get("seed")
    if not isinstance(seed, int) or seed not in config.seeds:
        raise ValueError("manifest seed is not configured")
    return backend, solver, seed


def _validate_multinomial_job(
    job: dict[str, Any],
    config: RealErmConfig,
) -> RealMultinomialSpec:
    _, _, seed = _validate_common_job(
        job,
        config,
        "multinomial",
        config.multinomial.solvers,
    )
    spec = RealMultinomialSpec(**job["problem_spec"])
    expected = RealMultinomialSpec(
        dataset=spec.dataset,
        data_root=config.data_root,
        box_lower=config.multinomial.box_lower,
        box_upper=config.multinomial.box_upper,
    )
    if spec.dataset not in config.multinomial.datasets or spec != expected:
        raise ValueError("manifest problem_spec does not match the configuration")
    if job.get("problem_id") != spec.problem_id:
        raise ValueError("manifest problem_id does not match problem_spec")
    if job.get("solver_seed") != derive_seed(seed, "multinomial_solver"):
        raise ValueError("manifest solver_seed does not match its master seed")
    return spec


def _validate_elastic_net_job(
    job: dict[str, Any],
    config: RealErmConfig,
    *,
    bounded: bool,
) -> RealElasticNetSpec:
    problem_type = "bounded_elastic_net" if bounded else "vanilla_elastic_net"
    configured = (
        config.elastic_net.bounded_solvers if bounded else config.elastic_net.vanilla_solvers
    )
    _, _, seed = _validate_common_job(
        job,
        config,
        problem_type,
        configured,
    )
    spec = RealElasticNetSpec(**job["problem_spec"])
    expected = RealElasticNetSpec(
        dataset=spec.dataset,
        data_root=config.data_root,
        regularization_fraction=spec.regularization_fraction,
    )
    if (
        spec.dataset not in config.elastic_net.datasets
        or spec.regularization_fraction not in config.elastic_net.regularization_fractions
        or spec != expected
    ):
        raise ValueError("manifest problem_spec does not match the configuration")
    if job.get("problem_id") != spec.problem_id(bounded=bounded):
        raise ValueError("manifest problem_id does not match problem_spec")
    if job.get("solver_seed") != derive_seed(seed, f"{problem_type}_solver"):
        raise ValueError("manifest solver_seed does not match its master seed")
    return spec


def _run_job(
    job: dict[str, Any],
    config: RealErmConfig,
    *,
    spec: RealMultinomialSpec | RealElasticNetSpec,
    problem_type: str,
    execution: SolverExecution,
    native_tolerance: float,
    max_iterations: int,
    batch_size: int,
    output_dir: Path,
) -> list[TrialRecord]:
    validate_controls(native_tolerance, max_iterations, batch_size)
    backend = job["backend"]
    solver = job["solver"]
    bounded = problem_type == "bounded_elastic_net"
    clarabel_solvers = {"clarabel_qdldl", "cuclarabel_cudss"}
    specification = {
        "problem_type": problem_type,
        "problem_spec": job["problem_spec"],
        "pre_torch_runtime": "clarabel" if solver in clarabel_solvers else None,
    }
    accuracy_fields = (
        {
            "relative_duality_gap_tolerance": config.accuracy.relative_duality_gap,
            "feasibility_tolerance": config.accuracy.feasibility,
        }
        if problem_type == "vanilla_elastic_net"
        else {
            "stationarity_tolerance": config.accuracy.stationarity,
            "feasibility_tolerance": config.accuracy.feasibility,
        }
    )
    command = {
        "solver": solver,
        "native_tolerance": native_tolerance,
        "max_iters": max_iterations,
        "batch_size": batch_size,
        "solver_seed": job["solver_seed"],
        **accuracy_fields,
    }
    problem_fields: dict[str, Any] = {
        "problem_type": problem_type,
        "dataset": spec.dataset,
        "n": spec.n,
        "p": spec.p,
        "feature_generator": spec.feature_generator,
        "feature_decay_exponent": spec.feature_decay_exponent,
    }
    if isinstance(spec, RealMultinomialSpec):
        problem_id = spec.problem_id
        problem_fields |= {
            "n_classes": spec.n_classes,
            "box_lower": spec.box_lower,
            "box_upper": spec.box_upper,
        }
    else:
        problem_id = spec.problem_id(bounded=bounded)
        problem_fields["regularization_fraction"] = spec.regularization_fraction

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
        metadata = outcome.get("solver_metadata", {}) | {
            "execution_phase": phase,
            "native_tolerance": native_tolerance,
            "max_iterations": iteration_limit,
            "solver_seed": job["solver_seed"],
            **accuracy_fields,
            "accuracy_thresholds_calibrated": config.accuracy.calibrated,
            "native_tolerances_calibrated": execution.tolerances_calibrated_for(backend),
        }
        timing_scope = (
            "native solver invocation; excludes dataset loading, random-feature "
            "materialization, worker startup, and benchmark-side conic construction and "
            "format conversion; includes solver-side JIT compilation, internal setup, "
            "and transfers when required"
            if bounded
            else "device-resident solver invocation; excludes dataset loading, "
            "random-feature materialization, worker startup, and format conversion; "
            "includes solver-side JIT compilation when required"
        )
        return record_outcome(
            suite=config.suite,
            problem_id=problem_id,
            run_key=f"{problem_type}-seed{job['seed']}",
            problem=problem_fields,
            solver=solver,
            backend=backend,
            seed=job["seed"],
            repetition=repetition,
            outcome=outcome | {"solver_metadata": metadata},
            runtimes=runtimes,
            output_dir=output_dir,
            worker_metadata=ready.get("worker_metadata", {}),
            timing_scope=timing_scope,
        )

    return run_repetitions(
        worker=worker,
        ready=ready,
        config=config,
        command=command,
        max_iterations=max_iterations,
        persist=persist,
    )


def run_real_multinomial_job(
    job: dict[str, Any],
    config: RealErmConfig,
    *,
    native_tolerance: float,
    max_iterations: int,
    batch_size: int,
    output_dir: Path,
) -> list[TrialRecord]:
    """Execute one real multinomial manifest job through an isolated worker."""
    spec = _validate_multinomial_job(job, config)
    return _run_job(
        job,
        config,
        spec=spec,
        problem_type="multinomial",
        execution=config.multinomial.execution,
        native_tolerance=native_tolerance,
        max_iterations=max_iterations,
        batch_size=batch_size,
        output_dir=output_dir,
    )


def run_real_vanilla_elastic_net_job(
    job: dict[str, Any],
    config: RealErmConfig,
    *,
    native_tolerance: float,
    max_iterations: int,
    batch_size: int,
    output_dir: Path,
) -> list[TrialRecord]:
    """Execute one real vanilla elastic-net job through an isolated worker."""
    spec = _validate_elastic_net_job(job, config, bounded=False)
    return _run_job(
        job,
        config,
        spec=spec,
        problem_type="vanilla_elastic_net",
        execution=config.elastic_net.vanilla_execution,
        native_tolerance=native_tolerance,
        max_iterations=max_iterations,
        batch_size=batch_size,
        output_dir=output_dir,
    )


def run_real_bounded_elastic_net_job(
    job: dict[str, Any],
    config: RealErmConfig,
    *,
    native_tolerance: float,
    max_iterations: int,
    batch_size: int,
    output_dir: Path,
) -> list[TrialRecord]:
    """Execute one real bounded elastic-net job through an isolated worker."""
    spec = _validate_elastic_net_job(job, config, bounded=True)
    return _run_job(
        job,
        config,
        spec=spec,
        problem_type="bounded_elastic_net",
        execution=config.elastic_net.bounded_execution,
        native_tolerance=native_tolerance,
        max_iterations=max_iterations,
        batch_size=batch_size,
        output_dir=output_dir,
    )

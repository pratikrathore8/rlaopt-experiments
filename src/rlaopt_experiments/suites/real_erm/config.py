"""Strict configuration for real-data ERM production experiments."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlaopt_experiments.suites.real_erm.data import DATASETS
from rlaopt_experiments.suites.synthetic_erm.config import (
    AccuracyThresholds,
    BackendCalibrationState,
    BackendSolvers,
    BackendTolerances,
    SolverExecution,
)


def _require_keys(data: dict[str, Any], expected: set[str], name: str) -> None:
    missing = expected - data.keys()
    unknown = data.keys() - expected
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


def _unique_nonempty(values: tuple[Any, ...], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")


def _validate_datasets(names: tuple[str, ...], problem: str) -> None:
    _unique_nonempty(names, f"{problem} datasets")
    unknown = set(names) - DATASETS.keys()
    if unknown:
        raise ValueError(f"unknown real datasets: {', '.join(sorted(unknown))}")
    wrong_problem = [name for name in names if DATASETS[name].problem != problem]
    if wrong_problem:
        raise ValueError(f"datasets do not belong to {problem}: {', '.join(wrong_problem)}")


def _validate_solver_tolerances(
    solvers: BackendSolvers,
    execution: SolverExecution,
    name: str,
) -> None:
    if execution.native_tolerances is None:
        raise ValueError(f"{name} requires frozen native tolerances")
    if execution.native_tolerances_calibrated is None:
        raise ValueError(f"{name} requires native-tolerance calibration state")
    if not all(
        execution.native_tolerances_calibrated.for_backend(backend) for backend in ("cpu", "cuda")
    ):
        raise ValueError(f"{name} native tolerances must be calibrated for both backends")
    for backend in ("cpu", "cuda"):
        configured = set(getattr(solvers, backend))
        tolerance_names = {solver for solver, _ in getattr(execution.native_tolerances, backend)}
        if configured != tolerance_names:
            raise ValueError(f"{name} {backend} solvers and native tolerances must match")


@dataclass(frozen=True)
class RealMultinomialExperiment:
    datasets: tuple[str, ...]
    box_lower: float
    box_upper: float
    solvers: BackendSolvers
    execution: SolverExecution

    def __post_init__(self) -> None:
        _validate_datasets(self.datasets, "multinomial")
        if not (
            math.isfinite(self.box_lower)
            and math.isfinite(self.box_upper)
            and self.box_lower < 0 < self.box_upper
        ):
            raise ValueError("multinomial bounds must contain zero")
        _validate_solver_tolerances(self.solvers, self.execution, "multinomial")


@dataclass(frozen=True)
class RealElasticNetExperiment:
    datasets: tuple[str, ...]
    regularization_fractions: tuple[float, ...]
    vanilla_solvers: BackendSolvers
    bounded_solvers: BackendSolvers
    vanilla_execution: SolverExecution
    bounded_execution: SolverExecution

    def __post_init__(self) -> None:
        _validate_datasets(self.datasets, "elastic_net")
        _unique_nonempty(self.regularization_fractions, "regularization fractions")
        if any(
            not math.isfinite(fraction) or fraction <= 0
            for fraction in self.regularization_fractions
        ):
            raise ValueError("regularization fractions must be finite and positive")
        _validate_solver_tolerances(
            self.vanilla_solvers,
            self.vanilla_execution,
            "vanilla elastic net",
        )
        _validate_solver_tolerances(
            self.bounded_solvers,
            self.bounded_execution,
            "bounded elastic net",
        )


@dataclass(frozen=True)
class RealErmConfig:
    suite: str
    data_root: str
    seeds: tuple[int, ...]
    warmups: int
    repetitions: int
    timeout_seconds: int
    startup_timeout_seconds: int
    accuracy: AccuracyThresholds
    multinomial: RealMultinomialExperiment
    elastic_net: RealElasticNetExperiment

    def __post_init__(self) -> None:
        if self.suite != "real_erm":
            raise ValueError("suite must be 'real_erm'")
        if not self.data_root:
            raise ValueError("data_root must be nonempty")
        _unique_nonempty(self.seeds, "seeds")
        if any(not isinstance(seed, int) or seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if self.warmups < 0:
            raise ValueError("warmups must be nonnegative")
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if self.timeout_seconds < 1 or self.startup_timeout_seconds < 1:
            raise ValueError("timeouts must be positive")
        if not self.accuracy.calibrated:
            raise ValueError("real-data production accuracy thresholds must be calibrated")


def _backend_solvers(data: dict[str, Any], name: str) -> BackendSolvers:
    _require_keys(data, {"cpu", "cuda"}, name)
    return BackendSolvers(cpu=tuple(data["cpu"]), cuda=tuple(data["cuda"]))


def _execution(data: dict[str, Any], name: str) -> SolverExecution:
    values = dict(data)
    _require_keys(
        values,
        {
            "max_iterations",
            "batch_size",
            "native_tolerances_calibrated",
            "native_tolerances",
        },
        name,
    )
    states = dict(values.pop("native_tolerances_calibrated"))
    tolerances = dict(values.pop("native_tolerances"))
    _require_keys(states, {"cpu", "cuda"}, f"{name} calibration state")
    _require_keys(tolerances, {"cpu", "cuda"}, f"{name} native tolerances")
    if not all(isinstance(states[backend], bool) for backend in ("cpu", "cuda")):
        raise ValueError(f"{name} calibration states must be Boolean")
    return SolverExecution(
        native_tolerances_calibrated=BackendCalibrationState(**states),
        native_tolerances=BackendTolerances(
            cpu=tuple(sorted(tolerances["cpu"].items())),
            cuda=tuple(sorted(tolerances["cuda"].items())),
        ),
        **values,
    )


def load_real_erm_config(path: Path) -> RealErmConfig:
    """Load a complete real-data production protocol, rejecting unknown fields."""
    root = tomllib.loads(path.read_text())
    _require_keys(root, {"experiment", "accuracy", "multinomial", "elastic_net"}, "root")
    experiment = dict(root["experiment"])
    _require_keys(
        experiment,
        {
            "suite",
            "data_root",
            "seeds",
            "warmups",
            "repetitions",
            "timeout_seconds",
            "startup_timeout_seconds",
        },
        "experiment",
    )
    accuracy_data = dict(root["accuracy"])
    _require_keys(
        accuracy_data,
        {"stationarity", "feasibility", "relative_duality_gap", "calibrated"},
        "accuracy",
    )

    multinomial_data = dict(root["multinomial"])
    _require_keys(
        multinomial_data,
        {"datasets", "box_lower", "box_upper", "solvers", "execution"},
        "multinomial",
    )
    multinomial = RealMultinomialExperiment(
        datasets=tuple(multinomial_data.pop("datasets")),
        solvers=_backend_solvers(multinomial_data.pop("solvers"), "multinomial solvers"),
        execution=_execution(multinomial_data.pop("execution"), "multinomial execution"),
        **multinomial_data,
    )

    elastic_net_data = dict(root["elastic_net"])
    _require_keys(
        elastic_net_data,
        {
            "datasets",
            "regularization_fractions",
            "vanilla_solvers",
            "bounded_solvers",
            "vanilla_execution",
            "bounded_execution",
        },
        "elastic_net",
    )
    elastic_net = RealElasticNetExperiment(
        datasets=tuple(elastic_net_data.pop("datasets")),
        regularization_fractions=tuple(elastic_net_data.pop("regularization_fractions")),
        vanilla_solvers=_backend_solvers(
            elastic_net_data.pop("vanilla_solvers"), "vanilla elastic-net solvers"
        ),
        bounded_solvers=_backend_solvers(
            elastic_net_data.pop("bounded_solvers"), "bounded elastic-net solvers"
        ),
        vanilla_execution=_execution(
            elastic_net_data.pop("vanilla_execution"), "vanilla elastic-net execution"
        ),
        bounded_execution=_execution(
            elastic_net_data.pop("bounded_execution"), "bounded elastic-net execution"
        ),
    )
    return RealErmConfig(
        accuracy=AccuracyThresholds(**accuracy_data),
        multinomial=multinomial,
        elastic_net=elastic_net,
        seeds=tuple(experiment.pop("seeds")),
        **experiment,
    )

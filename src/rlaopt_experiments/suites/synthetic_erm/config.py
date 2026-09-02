"""Strict configuration for the synthetic ERM development grid."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require_keys(data: dict[str, Any], expected: set[str], name: str) -> None:
    missing = expected - data.keys()
    unknown = data.keys() - expected
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _unique_nonempty(values: tuple[Any, ...], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True)
class ErmShape:
    n: int
    p: int

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("shape n must be at least two")
        if self.p < 1:
            raise ValueError("shape p must be positive")


@dataclass(frozen=True)
class BackendSolvers:
    cpu: tuple[str, ...]
    cuda: tuple[str, ...]

    def __post_init__(self) -> None:
        _unique_nonempty(self.cpu, "CPU solvers")
        _unique_nonempty(self.cuda, "CUDA solvers")
        if any(
            not isinstance(solver, str) or not solver.strip() for solver in self.cpu + self.cuda
        ):
            raise ValueError("solver names must be nonempty strings")


@dataclass(frozen=True)
class AccuracyThresholds:
    stationarity: float
    feasibility: float
    relative_duality_gap: float
    calibrated: bool

    def __post_init__(self) -> None:
        _positive(self.stationarity, "stationarity threshold")
        _positive(self.feasibility, "feasibility threshold")
        _positive(self.relative_duality_gap, "relative-duality-gap threshold")


@dataclass(frozen=True)
class BackendTolerances:
    cpu: tuple[tuple[str, float], ...]
    cuda: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        for backend, values in (("cpu", self.cpu), ("cuda", self.cuda)):
            names = tuple(name for name, _ in values)
            _unique_nonempty(names, f"{backend} native-tolerance solvers")
            for name, tolerance in values:
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("native-tolerance solver names must be nonempty")
                _positive(tolerance, f"native tolerance for {backend}/{name}")

    def for_solver(self, backend: str, solver: str) -> float:
        if backend not in {"cpu", "cuda"}:
            raise ValueError("backend must be cpu or cuda")
        try:
            return dict(getattr(self, backend))[solver]
        except KeyError as error:
            raise ValueError(f"no native tolerance for {backend}/{solver}") from error


@dataclass(frozen=True)
class SolverExecution:
    max_iterations: int
    batch_size: int
    native_tolerances_calibrated: bool
    native_tolerances: BackendTolerances

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class MultinomialExperiment:
    shapes: tuple[ErmShape, ...]
    n_classes: int
    teacher_scale: float
    box_lower: float
    box_upper: float
    solvers: BackendSolvers
    execution: SolverExecution

    def __post_init__(self) -> None:
        _unique_nonempty(self.shapes, "multinomial shapes")
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least two")
        _positive(self.teacher_scale, "teacher_scale")
        for backend in ("cpu", "cuda"):
            configured = set(getattr(self.solvers, backend))
            tolerance_names = {
                name for name, _ in getattr(self.execution.native_tolerances, backend)
            }
            if tolerance_names != configured:
                raise ValueError(
                    f"{backend} native tolerances must exactly match multinomial solvers"
                )
        if not (
            math.isfinite(self.box_lower)
            and math.isfinite(self.box_upper)
            and self.box_lower < 0 < self.box_upper
        ):
            raise ValueError("multinomial bounds must contain zero")


@dataclass(frozen=True)
class ElasticNetExperiment:
    shapes: tuple[ErmShape, ...]
    teacher_density: float
    noise_ratio: float
    teacher_intercept: float
    regularization_fractions: tuple[float, ...]
    vanilla_solvers: BackendSolvers
    bounded_solvers: BackendSolvers
    vanilla_execution: SolverExecution

    def __post_init__(self) -> None:
        _unique_nonempty(self.shapes, "elastic-net shapes")
        if not 0 < self.teacher_density <= 1:
            raise ValueError("teacher_density must lie in (0, 1]")
        if not math.isfinite(self.noise_ratio) or self.noise_ratio < 0:
            raise ValueError("noise_ratio must be finite and nonnegative")
        if not math.isfinite(self.teacher_intercept):
            raise ValueError("teacher_intercept must be finite")
        _unique_nonempty(self.regularization_fractions, "regularization_fractions")
        for fraction in self.regularization_fractions:
            _positive(fraction, "regularization fraction")
        for backend in ("cpu", "cuda"):
            configured = set(getattr(self.vanilla_solvers, backend))
            tolerance_names = {
                name for name, _ in getattr(self.vanilla_execution.native_tolerances, backend)
            }
            if tolerance_names != configured:
                raise ValueError(
                    f"{backend} native tolerances must exactly match vanilla elastic-net solvers"
                )


@dataclass(frozen=True)
class SyntheticErmConfig:
    suite: str
    seeds: tuple[int, ...]
    warmups: int
    repetitions: int
    timeout_seconds: int
    startup_timeout_seconds: int
    accuracy: AccuracyThresholds
    multinomial: MultinomialExperiment
    elastic_net: ElasticNetExperiment

    def __post_init__(self) -> None:
        if self.suite != "synthetic_erm":
            raise ValueError("suite must be 'synthetic_erm'")
        _unique_nonempty(self.seeds, "seeds")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative")
        if self.warmups < 0:
            raise ValueError("warmups must be nonnegative")
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        if self.startup_timeout_seconds < 1:
            raise ValueError("startup_timeout_seconds must be positive")


def _backend_solvers(data: dict[str, Any]) -> BackendSolvers:
    _require_keys(data, {"cpu", "cuda"}, "solver table")
    return BackendSolvers(cpu=tuple(data["cpu"]), cuda=tuple(data["cuda"]))


def _shapes(data: list[dict[str, Any]]) -> tuple[ErmShape, ...]:
    return tuple(ErmShape(**shape) for shape in data)


def _solver_execution(data: dict[str, Any], name: str) -> SolverExecution:
    execution_data = dict(data)
    _require_keys(
        execution_data,
        {
            "max_iterations",
            "batch_size",
            "native_tolerances_calibrated",
            "native_tolerances",
        },
        name,
    )
    tolerance_data = dict(execution_data.pop("native_tolerances"))
    _require_keys(tolerance_data, {"cpu", "cuda"}, f"{name} native tolerances")
    return SolverExecution(
        native_tolerances=BackendTolerances(
            cpu=tuple(sorted(tolerance_data["cpu"].items())),
            cuda=tuple(sorted(tolerance_data["cuda"].items())),
        ),
        **execution_data,
    )


def load_synthetic_erm_config(path: Path) -> SyntheticErmConfig:
    """Load a synthetic-ERM TOML file, rejecting missing or unknown fields."""
    root = tomllib.loads(path.read_text())
    _require_keys(root, {"experiment", "accuracy", "multinomial", "elastic_net"}, "root")
    experiment = dict(root["experiment"])
    _require_keys(
        experiment,
        {
            "suite",
            "seeds",
            "warmups",
            "repetitions",
            "timeout_seconds",
            "startup_timeout_seconds",
        },
        "experiment",
    )
    _require_keys(
        root["accuracy"],
        {"stationarity", "feasibility", "relative_duality_gap", "calibrated"},
        "accuracy",
    )
    accuracy = AccuracyThresholds(**root["accuracy"])

    multinomial_data = dict(root["multinomial"])
    _require_keys(
        multinomial_data,
        {
            "shapes",
            "n_classes",
            "teacher_scale",
            "box_lower",
            "box_upper",
            "solvers",
            "execution",
        },
        "multinomial",
    )
    execution = _solver_execution(multinomial_data.pop("execution"), "multinomial execution")
    multinomial = MultinomialExperiment(
        shapes=_shapes(multinomial_data.pop("shapes")),
        solvers=_backend_solvers(multinomial_data.pop("solvers")),
        execution=execution,
        **multinomial_data,
    )

    elastic_net_data = dict(root["elastic_net"])
    _require_keys(
        elastic_net_data,
        {
            "shapes",
            "teacher_density",
            "noise_ratio",
            "teacher_intercept",
            "regularization_fractions",
            "vanilla_solvers",
            "bounded_solvers",
            "vanilla_execution",
        },
        "elastic_net",
    )
    vanilla_execution = _solver_execution(
        elastic_net_data.pop("vanilla_execution"), "vanilla elastic-net execution"
    )
    elastic_net = ElasticNetExperiment(
        shapes=_shapes(elastic_net_data.pop("shapes")),
        vanilla_execution=vanilla_execution,
        regularization_fractions=tuple(elastic_net_data.pop("regularization_fractions")),
        vanilla_solvers=_backend_solvers(elastic_net_data.pop("vanilla_solvers")),
        bounded_solvers=_backend_solvers(elastic_net_data.pop("bounded_solvers")),
        **elastic_net_data,
    )
    return SyntheticErmConfig(
        accuracy=accuracy,
        multinomial=multinomial,
        elastic_net=elastic_net,
        seeds=tuple(experiment.pop("seeds")),
        **experiment,
    )

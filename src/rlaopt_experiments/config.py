"""Configuration loading and canonical benchmark grid."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Shape:
    n: int
    p: int
    family: str


@dataclass(frozen=True)
class ExperimentConfig:
    shapes: tuple[Shape, ...]
    alphas: tuple[float, ...]
    lambdas: tuple[float, ...]
    seeds: tuple[int, ...]
    kkt_tolerance: float
    timeout_seconds: int
    nystrom_rank: int
    warmups: int
    fast_repetitions: int
    fast_threshold_seconds: int


def load_experiment(path: Path) -> ExperimentConfig:
    data = tomllib.loads(path.read_text())["experiment"]
    shapes = tuple(Shape(**shape) for shape in data.pop("shapes"))
    return ExperimentConfig(shapes=shapes, **{
        key: tuple(value) if key in {"alphas", "lambdas", "seeds"} else value
        for key, value in data.items()
    })


def load_tolerances(path: Path, backend: str) -> dict[str, float]:
    return tomllib.loads(path.read_text())[backend]

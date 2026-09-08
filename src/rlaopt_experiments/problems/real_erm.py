"""Construct real-data ERM problems for the shared solver adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import torch

from rlaopt_experiments.problems.synthetic_erm import ElasticNetProblem, MultinomialProblem
from rlaopt_experiments.suites.real_erm.data import DATASETS, DatasetSpec, load_prepared_dataset
from rlaopt_experiments.suites.real_erm.features import materialize_random_features


PROBLEM_VERSION = "real-erm1"


def _catalog_entry(name: str, expected_problem: str) -> DatasetSpec:
    try:
        dataset = DATASETS[name]
    except KeyError as error:
        raise ValueError(f"unknown real dataset: {name}") from error
    if dataset.problem != expected_problem:
        raise ValueError(f"{name} is configured for {dataset.problem}, not {expected_problem}")
    return dataset


def _feature_generator(dataset: DatasetSpec) -> str:
    if dataset.random_features is None:
        return "prepared_real"
    return f"promise_{dataset.random_features.kind}"


@dataclass(frozen=True)
class RealMultinomialSpec:
    dataset: str
    data_root: str
    box_lower: float
    box_upper: float

    def __post_init__(self) -> None:
        _catalog_entry(self.dataset, "multinomial")
        if not self.data_root:
            raise ValueError("data_root must be nonempty")
        if not (
            math.isfinite(self.box_lower)
            and math.isfinite(self.box_upper)
            and self.box_lower < 0 < self.box_upper
        ):
            raise ValueError("multinomial bounds must contain zero")

    @property
    def dataset_spec(self) -> DatasetSpec:
        return _catalog_entry(self.dataset, "multinomial")

    @property
    def n(self) -> int:
        return self.dataset_spec.training_rows

    @property
    def p(self) -> int:
        return self.dataset_spec.solver_shape[1]

    @property
    def n_classes(self) -> int:
        classes = self.dataset_spec.classes
        if classes is None:
            raise RuntimeError("multinomial catalog entry has no class count")
        return classes

    @property
    def feature_generator(self) -> str:
        return _feature_generator(self.dataset_spec)

    @property
    def feature_decay_exponent(self) -> None:
        return None

    @property
    def problem_id(self) -> str:
        digest = self.dataset_spec.sha256[:12]
        return (
            f"{PROBLEM_VERSION}-multinomial-{self.dataset}-{digest}"
            f"-lo{self.box_lower:g}-hi{self.box_upper:g}"
        )


@dataclass(frozen=True)
class RealElasticNetSpec:
    dataset: str
    data_root: str
    regularization_fraction: float

    def __post_init__(self) -> None:
        _catalog_entry(self.dataset, "elastic_net")
        if not self.data_root:
            raise ValueError("data_root must be nonempty")
        if not math.isfinite(self.regularization_fraction) or self.regularization_fraction <= 0:
            raise ValueError("regularization_fraction must be finite and positive")

    @property
    def dataset_spec(self) -> DatasetSpec:
        return _catalog_entry(self.dataset, "elastic_net")

    @property
    def n(self) -> int:
        return self.dataset_spec.training_rows

    @property
    def p(self) -> int:
        return self.dataset_spec.solver_shape[1]

    @property
    def feature_generator(self) -> str:
        return _feature_generator(self.dataset_spec)

    @property
    def feature_decay_exponent(self) -> None:
        return None

    def problem_id(self, *, bounded: bool) -> str:
        digest = self.dataset_spec.sha256[:12]
        variant = "bounded" if bounded else "unbounded"
        return (
            f"{PROBLEM_VERSION}-elastic-net-{self.dataset}-{digest}"
            f"-{variant}-rf{self.regularization_fraction:g}"
        )


def _materialize_solver_matrix(
    dataset: DatasetSpec,
    matrix: np.ndarray | sparse.csr_matrix,
    device: torch.device,
) -> torch.Tensor:
    if dataset.random_features is not None:
        transformed = materialize_random_features(
            matrix,
            dataset.random_features,
            backend=device.type,
        )
        if isinstance(transformed, torch.Tensor):
            return transformed
        return torch.from_numpy(transformed).to(device)
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    return torch.as_tensor(dense, dtype=torch.float64, device=device)


def build_real_multinomial_problem(
    spec: RealMultinomialSpec,
    device: torch.device | str,
) -> MultinomialProblem:
    """Load one training corpus and construct its box-constrained softmax problem."""
    target_device = torch.device(device)
    prepared = load_prepared_dataset(spec.dataset, Path(spec.data_root))
    features = _materialize_solver_matrix(prepared.spec, prepared.matrix, target_device)
    labels = torch.as_tensor(prepared.target, dtype=torch.int64, device=target_device)
    return MultinomialProblem(
        spec=spec,
        X=features,
        y=labels,
        teacher=None,
        feature_frobenius_norm=float(torch.linalg.vector_norm(features)),
    )


def build_real_elastic_net_problem(
    spec: RealElasticNetSpec,
    *,
    bounded: bool,
    device: torch.device | str,
) -> ElasticNetProblem:
    """Load one regression corpus and set both penalties from its lambda-max scale."""
    if not bounded:
        raise ValueError("Only bounded elastic-net experiments are supported")
    target_device = torch.device(device)
    prepared = load_prepared_dataset(spec.dataset, Path(spec.data_root))
    features = _materialize_solver_matrix(prepared.spec, prepared.matrix, target_device)
    response = torch.as_tensor(prepared.target, dtype=torch.float64, device=target_device)
    centered_response = response - response.mean()
    lambda_max = float(
        torch.linalg.vector_norm(features.mT @ centered_response, ord=float("inf")) / spec.n
    )
    if not math.isfinite(lambda_max) or lambda_max <= 0:
        raise ValueError(f"{spec.dataset} has invalid lambda_max {lambda_max}")
    regularization = spec.regularization_fraction * lambda_max
    return ElasticNetProblem(
        spec=spec,
        X=features,
        y=response,
        teacher_weights=None,
        teacher_intercept=None,
        lambda_max=lambda_max,
        lambda_l1=regularization,
        lambda_l2=regularization,
        bounded=bounded,
        feature_frobenius_norm=float(torch.linalg.vector_norm(features)),
    )

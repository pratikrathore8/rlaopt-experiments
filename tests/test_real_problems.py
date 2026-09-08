from __future__ import annotations

import bz2
import hashlib
from dataclasses import asdict

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import torch

from rlaopt_experiments.problems.real_erm import (
    RealElasticNetSpec,
    RealMultinomialSpec,
    build_real_elastic_net_problem,
    build_real_multinomial_problem,
)
from rlaopt_experiments.suites import get_suite
from rlaopt_experiments.suites.real_erm import data


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_real_elastic_net_problem_materializes_random_features(tmp_path, monkeypatch) -> None:
    source = tmp_path / "regression.parquet"
    pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, -1.0, 0.5, 3.0],
            "target": [1.0, -2.0, 0.0, 4.0],
        }
    ).to_parquet(source, index=False)
    dataset = data.DatasetSpec(
        name="regression",
        problem="elastic_net",
        source="openml_parquet",
        url=source.as_uri(),
        raw_filename=source.name,
        source_rows=4,
        training_rows=4,
        features=2,
        classes=None,
        target="target",
        random_features=data.RandomFeatureSpec("gaussian", 5, bandwidth=1.0),
        sha256=_digest(source),
    )
    monkeypatch.setitem(data.DATASETS, dataset.name, dataset)
    data.prepare_datasets([dataset.name], tmp_path / "cache")
    spec = RealElasticNetSpec(
        dataset=dataset.name,
        data_root=str(tmp_path / "cache"),
        regularization_fraction=0.1,
    )

    bounded = build_real_elastic_net_problem(spec, bounded=True, device="cpu")

    assert bounded.X.shape == (4, 5)
    assert bounded.X.dtype == torch.float64
    assert bounded.y.dtype == torch.float64
    assert bounded.teacher_weights is None
    assert bounded.teacher_intercept is None
    centered = bounded.y - bounded.y.mean()
    expected_lambda_max = float((bounded.X.mT @ centered).abs().max() / spec.n)
    assert bounded.lambda_max == expected_lambda_max
    assert bounded.lambda_l1 == 0.1 * expected_lambda_max
    assert bounded.lambda_l2 == 0.1 * expected_lambda_max
    assert bounded.bounded


def test_real_suite_builds_multinomial_problem_from_sparse_cache(tmp_path, monkeypatch) -> None:
    source = tmp_path / "classification.bz2"
    source.write_bytes(bz2.compress(b"10 1:3 2:4\n20 2:2\n10 1:1 3:1\n"))
    dataset = data.DatasetSpec(
        name="classification",
        problem="multinomial",
        source="libsvm",
        url=source.as_uri(),
        raw_filename=source.name,
        source_rows=3,
        training_rows=3,
        features=3,
        classes=2,
        compression="bz2",
        sha256=_digest(source),
    )
    monkeypatch.setitem(data.DATASETS, dataset.name, dataset)
    data.prepare_datasets([dataset.name], tmp_path / "cache")
    spec = RealMultinomialSpec(
        dataset=dataset.name,
        data_root=str(tmp_path / "cache"),
        box_lower=-1.0,
        box_upper=1.0,
    )

    problem = build_real_multinomial_problem(spec, device="cpu")

    assert problem.X.shape == (3, 3)
    assert problem.X.layout == torch.strided
    assert problem.X.dtype == torch.float64
    assert problem.y.dtype == torch.int64
    assert problem.teacher is None
    np.testing.assert_allclose(
        torch.linalg.vector_norm(problem.X, dim=1).numpy(),
        np.ones(3),
    )
    suite = get_suite("real_erm")
    generated = suite.generate(
        {"problem_type": "multinomial", "problem_spec": asdict(spec)},
        torch.device("cpu"),
    )
    torch.testing.assert_close(generated.X, problem.X)
    metadata = suite.problem_metadata(generated)
    assert metadata["dataset"] == dataset.name
    assert metadata["base_shape"] == [3, 3]
    assert metadata["solver_shape"] == [3, 3]
    assert metadata["random_features"] is None
    assert metadata["matrix_representation"] == "materialized_dense"
    assert sparse.issparse(data.load_prepared_dataset(dataset.name, tmp_path / "cache").matrix)

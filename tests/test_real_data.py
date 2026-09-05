from __future__ import annotations

import bz2
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse

from rlaopt_experiments.suites.real_erm import data


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_catalog_freezes_requested_datasets_and_random_features() -> None:
    assert set(data.DATASETS) == {
        "acsincome",
        "e2006",
        "realsim",
        "yearpredictionmsd",
        "yolanda",
        "cifar10",
        "rcv1",
        "svhn",
        "news20",
        "fashion_mnist",
    }
    assert data.DATASETS["acsincome"].random_features.dimension == 1_000
    assert data.DATASETS["yolanda"].random_features.dimension == 1_000
    assert data.DATASETS["yearpredictionmsd"].random_features.dimension == 4_367
    assert all(spec.sha256 is not None for spec in data.DATASETS.values())
    assert data.DATASETS["rcv1"].classes == 51
    assert data.DATASETS["news20"].raw_filename == "news20.bz2"
    assert data.DATASETS["rcv1"].raw_filename == "rcv1_train.multiclass.bz2"


def test_prepare_libsvm_is_resumable_and_redownloadable(tmp_path, monkeypatch) -> None:
    source = tmp_path / "fixture.libsvm.bz2"
    source.write_bytes(bz2.compress(b"10 1:3 2:4\n20 2:2 3:0\n10 1:1 3:1\n"))
    spec = data.DatasetSpec(
        name="fixture",
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
    monkeypatch.setattr(data, "DATASETS", {"fixture": spec})
    root = tmp_path / "cache"

    [prepared] = data.prepare_datasets(["fixture"], root)
    matrix = sparse.load_npz(prepared / "matrix.npz")
    target = np.load(prepared / "target.npy")
    metadata = json.loads((prepared / "metadata.json").read_text())
    np.testing.assert_allclose(sparse.linalg.norm(matrix, axis=1), [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(target, [0, 1, 0])
    assert metadata["preprocessing"]["class_labels"] == [10.0, 20.0]
    assert metadata["preprocessing"]["representation"] == "csr"
    loaded = data.load_prepared_dataset("fixture", root)
    assert sparse.isspmatrix_csr(loaded.matrix)
    np.testing.assert_array_equal(loaded.target, target)

    data.prepare_datasets(["fixture"], root)
    raw = root / "raw" / "fixture" / source.name
    raw.write_bytes(b"corrupt")
    data.prepare_datasets(["fixture"], root, redownload=True)
    assert _digest(raw) == spec.sha256

    source.write_bytes(bz2.compress(b"10 1:9\n20 2:9\n10 3:9\n"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        data.prepare_datasets(["fixture"], root, redownload=True)
    assert _digest(raw) == spec.sha256
    assert not (raw.parent / f".{raw.name}.part").exists()


def test_prepare_openml_uses_training_prefix_and_standardizes(tmp_path, monkeypatch) -> None:
    source = tmp_path / "fixture.parquet"
    pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 100.0],
            "b": [2.0, 4.0, 6.0, 200.0],
            "class": [0, 1, 0, 1],
        }
    ).to_parquet(source, index=False)
    spec = data.DatasetSpec(
        name="fixture",
        problem="multinomial",
        source="openml_parquet",
        url=source.as_uri(),
        raw_filename=source.name,
        source_rows=4,
        training_rows=3,
        features=2,
        classes=2,
        target="class",
        random_features=data.RandomFeatureSpec("gaussian", 1_000, bandwidth=1.0),
        sha256=_digest(source),
    )
    monkeypatch.setattr(data, "DATASETS", {"fixture": spec})

    [prepared] = data.prepare_datasets(["fixture"], tmp_path / "cache")
    matrix = np.load(prepared / "matrix.npy")
    target = np.load(prepared / "target.npy")
    metadata = json.loads((prepared / "metadata.json").read_text())
    assert matrix.shape == (3, 2)
    np.testing.assert_allclose(matrix.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(matrix.std(axis=0), 1.0, atol=1e-15)
    np.testing.assert_array_equal(target, [0, 1, 0])
    assert metadata["contains_only_training_rows"] is True
    assert metadata["training_selection"] == "official_training_prefix"
    assert metadata["classes"] == 2
    assert metadata["random_features"]["dimension"] == 1_000
    assert {path.name for path in prepared.iterdir()} == {
        "matrix.npy",
        "target.npy",
        "metadata.json",
    }
    loaded = data.load_prepared_dataset("fixture", tmp_path / "cache")
    assert isinstance(loaded.matrix, np.ndarray)
    np.testing.assert_array_equal(loaded.target, target)


def test_load_prepared_dataset_rejects_stale_metadata(tmp_path, monkeypatch) -> None:
    source = tmp_path / "fixture.parquet"
    pd.DataFrame({"a": [1.0, 2.0], "target": [3.0, 4.0]}).to_parquet(source, index=False)
    spec = data.DatasetSpec(
        name="fixture",
        problem="elastic_net",
        source="openml_parquet",
        url=source.as_uri(),
        raw_filename=source.name,
        source_rows=2,
        training_rows=2,
        features=1,
        classes=None,
        target="target",
        sha256=_digest(source),
    )
    monkeypatch.setattr(data, "DATASETS", {"fixture": spec})
    [prepared] = data.prepare_datasets(["fixture"], tmp_path / "cache")
    metadata_path = prepared / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["rows"] = 3
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="does not match"):
        data.load_prepared_dataset("fixture", tmp_path / "cache")


def test_load_prepared_dataset_rejects_corrupt_artifact(tmp_path, monkeypatch) -> None:
    source = tmp_path / "fixture.parquet"
    pd.DataFrame({"a": [1.0, 2.0], "target": [3.0, 4.0]}).to_parquet(source, index=False)
    spec = data.DatasetSpec(
        name="fixture",
        problem="elastic_net",
        source="openml_parquet",
        url=source.as_uri(),
        raw_filename=source.name,
        source_rows=2,
        training_rows=2,
        features=1,
        classes=None,
        target="target",
        sha256=_digest(source),
    )
    monkeypatch.setattr(data, "DATASETS", {"fixture": spec})
    [prepared] = data.prepare_datasets(["fixture"], tmp_path / "cache")
    (prepared / "target.npy").write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        data.load_prepared_dataset("fixture", tmp_path / "cache")

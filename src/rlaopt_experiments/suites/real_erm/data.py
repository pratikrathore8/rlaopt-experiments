"""Download and preprocess the frozen real-data benchmark corpus.

Random-feature matrices are deliberately not materialized here. The processed
artifacts contain the compact base matrix and enough metadata to regenerate the
PROMISE feature map deterministically in a benchmark worker.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import lzma
import os
import shutil
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize


CATALOG_VERSION = "real-erm-v1"
PROMISE_NUMPY_SEED = 2468


@dataclass(frozen=True)
class RandomFeatureSpec:
    kind: Literal["gaussian", "relu"]
    dimension: int
    seed: int = PROMISE_NUMPY_SEED
    bandwidth: float | None = None
    convention: str = "promise"

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("random-feature dimension must be positive")
        if self.seed < 0:
            raise ValueError("random-feature seed must be nonnegative")
        if self.kind == "gaussian":
            if self.bandwidth is None or self.bandwidth <= 0:
                raise ValueError("Gaussian random features require a positive bandwidth")
        elif self.bandwidth is not None:
            raise ValueError("ReLU random features do not use a bandwidth")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    problem: Literal["elastic_net", "multinomial"]
    source: Literal["libsvm", "openml_parquet"]
    url: str
    raw_filename: str
    source_rows: int
    training_rows: int
    features: int
    classes: int | None
    target: str | None = None
    compression: Literal["bz2", "xz"] | None = None
    random_features: RandomFeatureSpec | None = None
    sha256: str | None = None

    @property
    def base_shape(self) -> tuple[int, int]:
        return self.training_rows, self.features

    @property
    def solver_shape(self) -> tuple[int, int]:
        columns = self.random_features.dimension if self.random_features else self.features
        return self.training_rows, columns

    def __post_init__(self) -> None:
        if not self.name or not self.raw_filename:
            raise ValueError("dataset name and raw filename must be nonempty")
        if self.source_rows < 1 or self.training_rows < 1 or self.features < 1:
            raise ValueError("dataset dimensions must be positive")
        if self.training_rows > self.source_rows:
            raise ValueError("training rows cannot exceed source rows")
        if self.problem == "multinomial":
            if self.classes is None or self.classes < 2:
                raise ValueError("multinomial datasets require at least two classes")
        elif self.classes is not None:
            raise ValueError("elastic-net datasets cannot specify classes")
        if self.source == "openml_parquet" and self.target is None:
            raise ValueError("OpenML datasets require a target column")
        if self.source == "libsvm" and self.target is not None:
            raise ValueError("LIBSVM targets are stored in the first column")
        if self.sha256 is not None and len(self.sha256) != 64:
            raise ValueError("SHA-256 digests must contain 64 hexadecimal characters")


_REGRESSION = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression"
_BINARY = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
_MULTICLASS = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass"
_OPENML = "https://data.openml.org/datasets/0004"


DATASETS: dict[str, DatasetSpec] = {
    "acsincome": DatasetSpec(
        name="acsincome",
        problem="elastic_net",
        source="openml_parquet",
        url=f"{_OPENML}/43141/dataset_43141.pq",
        raw_filename="dataset_43141.pq",
        source_rows=1_664_500,
        training_rows=1_664_500,
        features=11,
        classes=None,
        target="PINCP",
        random_features=RandomFeatureSpec("gaussian", 1_000, bandwidth=1.0),
        sha256=("2b627b6d2b0a72138b76430c640353eff2f32578e46cd96ef7e1b91ced688256"),
    ),
    "e2006": DatasetSpec(
        name="e2006",
        problem="elastic_net",
        source="libsvm",
        url=f"{_REGRESSION}/E2006.train.bz2",
        raw_filename="E2006.train.bz2",
        source_rows=16_087,
        training_rows=16_087,
        features=150_360,
        classes=None,
        compression="bz2",
        sha256=("944066984ed3b6c9c3b137a4968afead81dc80f6b77f16f1988b0ecb591c4683"),
    ),
    "realsim": DatasetSpec(
        name="realsim",
        problem="elastic_net",
        source="libsvm",
        url=f"{_BINARY}/real-sim.bz2",
        raw_filename="real-sim.bz2",
        source_rows=72_309,
        training_rows=72_309,
        features=20_958,
        classes=None,
        compression="bz2",
        sha256=("7cd52374f55ce62e803ef33ca1c44689427c23d481eff64026f8ba9f53b8cc27"),
    ),
    "yearpredictionmsd": DatasetSpec(
        name="yearpredictionmsd",
        problem="elastic_net",
        source="libsvm",
        url=f"{_REGRESSION}/YearPredictionMSD.bz2",
        raw_filename="YearPredictionMSD.bz2",
        source_rows=463_715,
        training_rows=463_715,
        features=90,
        classes=None,
        compression="bz2",
        random_features=RandomFeatureSpec("relu", 4_367),
        sha256=("bd820472d9a6d87ae30d6208dcbfd0e80386e96e6e6832f78c498cd78df6da29"),
    ),
    "yolanda": DatasetSpec(
        name="yolanda",
        problem="elastic_net",
        source="openml_parquet",
        url=f"{_OPENML}/42705/dataset_42705.pq",
        raw_filename="dataset_42705.pq",
        source_rows=400_000,
        training_rows=400_000,
        features=100,
        classes=None,
        target="101",
        random_features=RandomFeatureSpec("gaussian", 1_000, bandwidth=1.0),
        sha256=("28acc6892d750abe6f7b4f01e622b41811a92597984c363db7a1f3a39505c28b"),
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        problem="multinomial",
        source="libsvm",
        url=f"{_MULTICLASS}/cifar10.bz2",
        raw_filename="cifar10.bz2",
        source_rows=50_000,
        training_rows=50_000,
        features=3_072,
        classes=10,
        compression="bz2",
        sha256=("d219329bafdf9f0c125ca609576e0ebf696c50d9dc1dfc8c659b5491e3043d58"),
    ),
    "rcv1": DatasetSpec(
        name="rcv1",
        problem="multinomial",
        source="libsvm",
        url=f"{_MULTICLASS}/rcv1_train.multiclass.bz2",
        raw_filename="rcv1_train.multiclass.bz2",
        source_rows=15_564,
        training_rows=15_564,
        features=47_236,
        classes=51,
        compression="bz2",
        sha256=("79691e3f974aa6a08d5e3116c61aff822fbdb2725e99ed27fde4351af51076a8"),
    ),
    "svhn": DatasetSpec(
        name="svhn",
        problem="multinomial",
        source="libsvm",
        url=f"{_MULTICLASS}/SVHN.xz",
        raw_filename="SVHN.xz",
        source_rows=73_257,
        training_rows=73_257,
        features=3_072,
        classes=10,
        compression="xz",
        sha256=("edcd68156695f4f4bd36ac1c8e7bf263823dfc251ddd8c2ddec8fd2385875269"),
    ),
    "news20": DatasetSpec(
        name="news20",
        problem="multinomial",
        source="libsvm",
        url=f"{_MULTICLASS}/news20.bz2",
        raw_filename="news20.bz2",
        source_rows=15_935,
        training_rows=15_935,
        features=62_061,
        classes=20,
        compression="bz2",
        sha256=("8c79851f6f5fdc5540a0d04a5ab728353c42d26b84e73f93bda301c06c15ffe9"),
    ),
    "fashion_mnist": DatasetSpec(
        name="fashion_mnist",
        problem="multinomial",
        source="openml_parquet",
        url=f"{_OPENML}/40996/dataset_40996.pq",
        raw_filename="dataset_40996.pq",
        source_rows=70_000,
        training_rows=60_000,
        features=784,
        classes=10,
        target="class",
        sha256=("78ede986ed6edc4fb68d32a4a59be384ed00bbe2130b90cf55ffb4da21ae11f3"),
    ),
}


@dataclass(frozen=True)
class PreparedDataset:
    """One validated base matrix and target loaded from the prepared cache."""

    spec: DatasetSpec
    matrix: np.ndarray | sparse.csr_matrix
    target: np.ndarray
    metadata: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(spec: DatasetSpec, destination: Path, *, redownload: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not redownload:
        digest = _sha256(destination)
    else:
        partial = destination.with_name(f".{destination.name}.part")
        print(f"downloading {spec.name}: {spec.url}", flush=True)
        request = urllib.request.Request(
            spec.url,
            headers={"User-Agent": "rlaopt-experiments/0.1.0"},
        )
        for attempt in range(1, 5):
            partial.unlink(missing_ok=True)
            try:
                with (
                    urllib.request.urlopen(request, timeout=120) as response,
                    partial.open("wb") as output,
                ):
                    shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            except Exception:
                partial.unlink(missing_ok=True)
                if attempt == 4:
                    raise
                delay = 2**attempt
                print(f"download failed; retrying {spec.name} in {delay}s", flush=True)
                time.sleep(delay)
                continue
            digest = _sha256(partial)
            if spec.sha256 is not None and digest != spec.sha256:
                partial.unlink(missing_ok=True)
                raise ValueError(
                    f"SHA-256 mismatch for {spec.name}: expected {spec.sha256}, observed {digest}"
                )
            os.replace(partial, destination)
            break
    if spec.sha256 is not None and digest != spec.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {spec.name}: expected {spec.sha256}, observed {digest}"
        )
    return digest


def _compressed_stream(path: Path, compression: str | None) -> BinaryIO:
    if compression == "bz2":
        return bz2.open(path, "rb")
    if compression == "xz":
        return lzma.open(path, "rb")
    return path.open("rb")


def _standardize_dense(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    values = np.array(values, dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("dense features contain NaN or infinite values")
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    zero_variance = scale == 0
    scale[zero_variance] = 1.0
    values -= mean
    values /= scale
    return values, mean, scale, int(np.count_nonzero(zero_variance))


def _standardize_target(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("regression target must be a finite vector")
    mean = float(values.mean(dtype=np.float64))
    scale = float(values.std(dtype=np.float64))
    if scale == 0:
        raise ValueError("regression target has zero variance")
    return (values - mean) / scale, mean, scale


def _encode_classes(values: np.ndarray) -> tuple[np.ndarray, list[float | int | str]]:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("classification target must be a vector")
    labels, encoded = np.unique(values, return_inverse=True)
    if labels.size < 2:
        raise ValueError("classification target must contain at least two classes")
    serializable = [label.item() if hasattr(label, "item") else label for label in labels]
    return encoded.astype(np.int64, copy=False), serializable


def _load_libsvm(spec: DatasetSpec, raw_path: Path) -> tuple[sparse.csr_matrix, np.ndarray]:
    with _compressed_stream(raw_path, spec.compression) as source:
        matrix, target = load_svmlight_file(
            source,
            n_features=spec.features,
            dtype=np.float64,
            zero_based="auto",
        )
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    matrix = normalize(matrix, norm="l2", axis=1, copy=False)
    matrix.sort_indices()
    if matrix.shape != (spec.training_rows, spec.features):
        raise ValueError(
            f"unexpected {spec.name} shape {matrix.shape}; "
            f"expected {(spec.training_rows, spec.features)}"
        )
    if not np.isfinite(matrix.data).all() or not np.isfinite(target).all():
        raise ValueError(f"{spec.name} contains NaN or infinite values")
    return matrix, np.asarray(target, dtype=np.float64)


def _load_openml(
    spec: DatasetSpec, raw_path: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    frame = pd.read_parquet(raw_path)
    if len(frame) != spec.source_rows:
        raise ValueError(
            f"unexpected {spec.name} source rows {len(frame)}; expected {spec.source_rows}"
        )
    if spec.target not in frame.columns:
        raise ValueError(f"target column {spec.target!r} is absent from {spec.name}")
    # OpenML 40996 retains the original Fashion-MNIST ordering: the official
    # 60,000 training examples precede the 10,000 test examples.
    frame = frame.iloc[: spec.training_rows]
    target = frame.pop(spec.target).to_numpy()
    if frame.shape[1] != spec.features:
        raise ValueError(
            f"unexpected {spec.name} feature count {frame.shape[1]}; expected {spec.features}"
        )
    matrix, mean, scale, zero_variance_features = _standardize_dense(frame.to_numpy())
    return (
        matrix,
        target,
        {
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            "zero_variance_features": zero_variance_features,
        },
    )


def _artifact_hashes(directory: Path, matrix_filename: str) -> dict[str, str]:
    return {
        matrix_filename: _sha256(directory / matrix_filename),
        "target.npy": _sha256(directory / "target.npy"),
    }


def _prepare_one(
    spec: DatasetSpec,
    data_root: Path,
    *,
    redownload: bool,
    reprocess: bool,
) -> Path:
    raw_path = data_root / "raw" / spec.name / spec.raw_filename
    source_sha256 = _download(spec, raw_path, redownload=redownload)
    destination = data_root / "processed" / spec.name
    metadata_path = destination / "metadata.json"
    if metadata_path.exists() and not reprocess and not redownload:
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("catalog_version") == CATALOG_VERSION
            and metadata.get("source_sha256") == source_sha256
        ):
            print(f"using prepared {spec.name}: {destination}", flush=True)
            return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{spec.name}-", dir=destination.parent))
    try:
        preprocessing: dict[str, object]
        if spec.source == "libsvm":
            matrix, target = _load_libsvm(spec, raw_path)
            matrix_filename = "matrix.npz"
            sparse.save_npz(temporary / matrix_filename, matrix, compressed=True)
            preprocessing = {"features": "row_l2_normalized", "representation": "csr"}
        else:
            matrix, target, preprocessing = _load_openml(spec, raw_path)
            matrix_filename = "matrix.npy"
            np.save(temporary / matrix_filename, matrix, allow_pickle=False)
            preprocessing |= {"features": "column_standardized", "representation": "dense"}

        target_metadata: dict[str, object]
        if spec.problem == "multinomial":
            target, labels = _encode_classes(target)
            if len(labels) != spec.classes:
                raise ValueError(
                    f"unexpected {spec.name} class count {len(labels)}; expected {spec.classes}"
                )
            target_metadata = {"target": "contiguous_class_indices", "class_labels": labels}
        elif spec.source == "openml_parquet":
            target, target_mean, target_scale = _standardize_target(target)
            target_metadata = {
                "target": "standardized",
                "target_mean": target_mean,
                "target_scale": target_scale,
            }
        else:
            target = np.asarray(target, dtype=np.float64)
            target_metadata = {"target": "source_values"}
        np.save(temporary / "target.npy", target, allow_pickle=False)

        metadata = {
            "catalog_version": CATALOG_VERSION,
            "dataset": spec.name,
            "problem": spec.problem,
            "source_url": spec.url,
            "source_filename": spec.raw_filename,
            "source_sha256": source_sha256,
            "contains_only_training_rows": True,
            "training_selection": (
                "official_training_prefix"
                if spec.training_rows < spec.source_rows
                else "entire_source_training_file"
            ),
            "source_rows": spec.source_rows,
            "rows": spec.training_rows,
            "features": spec.features,
            "classes": spec.classes,
            "dtype": "float64",
            "matrix_file": matrix_filename,
            "preprocessing": preprocessing | target_metadata,
            "random_features": (
                asdict(spec.random_features) if spec.random_features is not None else None
            ),
        }
        metadata["artifact_sha256"] = _artifact_hashes(temporary, matrix_filename)
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"prepared {spec.name}: {destination}", flush=True)
    return destination


def prepare_datasets(
    names: list[str],
    data_root: Path,
    *,
    redownload: bool = False,
    reprocess: bool = False,
) -> list[Path]:
    """Download and preprocess selected datasets into a deterministic cache."""
    unknown = sorted(set(names) - DATASETS.keys())
    if unknown:
        raise ValueError(f"unknown real datasets: {', '.join(unknown)}")
    if not names:
        raise ValueError("at least one real dataset is required")
    data_root = data_root.expanduser().resolve()
    return [
        _prepare_one(
            DATASETS[name],
            data_root,
            redownload=redownload,
            reprocess=reprocess or redownload,
        )
        for name in names
    ]


def load_prepared_dataset(name: str, data_root: Path) -> PreparedDataset:
    """Load one prepared artifact after validating it against the frozen catalog."""
    try:
        spec = DATASETS[name]
    except KeyError as error:
        raise ValueError(f"unknown real dataset: {name}") from error
    directory = data_root.expanduser().resolve() / "processed" / name
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"prepared metadata is missing for {name}: run prepare-real-data first"
        )
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "catalog_version": CATALOG_VERSION,
        "dataset": name,
        "problem": spec.problem,
        "source_sha256": spec.sha256,
        "rows": spec.training_rows,
        "features": spec.features,
        "classes": spec.classes,
        "dtype": "float64",
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"prepared metadata does not match {name} catalog: {mismatches}")

    matrix_filename = metadata.get("matrix_file")
    if matrix_filename not in {"matrix.npz", "matrix.npy"}:
        raise ValueError(f"prepared {name} has an unsupported matrix file: {matrix_filename!r}")
    artifact_sha256 = metadata.get("artifact_sha256")
    expected_artifacts = {matrix_filename, "target.npy"}
    if not isinstance(artifact_sha256, dict) or set(artifact_sha256) != expected_artifacts:
        raise ValueError(f"prepared {name} has incomplete artifact digests")
    for filename, expected_digest in artifact_sha256.items():
        artifact_path = directory / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"prepared artifact is missing: {artifact_path}")
        observed_digest = _sha256(artifact_path)
        if observed_digest != expected_digest:
            raise ValueError(
                f"prepared artifact SHA-256 mismatch for {name}/{filename}: "
                f"expected {expected_digest}, observed {observed_digest}"
            )
    if matrix_filename == "matrix.npz":
        matrix = sparse.load_npz(directory / matrix_filename).tocsr()
    else:
        matrix = np.load(directory / matrix_filename, mmap_mode="c")
    target = np.load(directory / "target.npy", mmap_mode="c")
    if matrix.shape != spec.base_shape or matrix.dtype != np.float64:
        raise ValueError(
            f"prepared {name} matrix is {matrix.shape}/{matrix.dtype}; "
            f"expected {spec.base_shape}/float64"
        )
    if target.shape != (spec.training_rows,):
        raise ValueError(f"prepared {name} target has unexpected shape {target.shape}")
    expected_target_dtype = np.int64 if spec.problem == "multinomial" else np.float64
    if target.dtype != expected_target_dtype:
        raise ValueError(
            f"prepared {name} target is {target.dtype}; expected {expected_target_dtype}"
        )
    return PreparedDataset(spec=spec, matrix=matrix, target=target, metadata=metadata)

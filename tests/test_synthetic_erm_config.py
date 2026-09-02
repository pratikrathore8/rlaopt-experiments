from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rlaopt_experiments.suites.synthetic_erm.config import (
    AccuracyThresholds,
    BackendSolvers,
    ElasticNetExperiment,
    ErmShape,
    SyntheticErmConfig,
    load_synthetic_erm_config,
)


CONFIG = Path(__file__).parents[1] / "configs" / "synthetic_erm_smoke.toml"


def test_load_synthetic_erm_smoke_config() -> None:
    config = load_synthetic_erm_config(CONFIG)

    assert isinstance(config, SyntheticErmConfig)
    assert config.suite == "synthetic_erm"
    assert config.seeds == (0, 1, 2)
    assert config.repetitions == 1
    assert not config.accuracy.calibrated
    assert config.multinomial.execution.max_iterations == 10_000
    assert config.multinomial.execution.batch_size == 256
    assert not config.multinomial.execution.native_tolerances_calibrated
    assert (
        config.multinomial.execution.native_tolerances.for_solver("cpu", "rlaopt_sapphire") == 1e-6
    )
    assert config.multinomial.shapes == (ErmShape(1024, 64), ErmShape(4096, 256))
    assert config.elastic_net.regularization_fractions == (0.1, 0.01)
    assert "sklearn_coordinate_descent" in config.elastic_net.vanilla_solvers.cpu
    assert "cuclarabel_cudss" in config.elastic_net.bounded_solvers.cuda


def test_elastic_net_variants_share_one_data_grid() -> None:
    config = load_synthetic_erm_config(CONFIG)

    assert not hasattr(config.elastic_net, "vanilla_shapes")
    assert not hasattr(config.elastic_net, "bounded_shapes")
    assert config.elastic_net.shapes


def test_suite_and_calibration_state_are_required() -> None:
    config = load_synthetic_erm_config(CONFIG)

    with pytest.raises(ValueError, match="suite"):
        replace(config, suite="synthetic_ridge")
    with pytest.raises(TypeError, match="calibrated"):
        AccuracyThresholds(
            stationarity=1e-6,
            feasibility=1e-8,
            relative_duality_gap=1e-6,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("teacher_density", 0.0, "teacher_density"),
        ("noise_ratio", -1.0, "noise_ratio"),
        ("regularization_fractions", (), "regularization_fractions"),
    ],
)
def test_invalid_elastic_net_scales_are_rejected(field: str, value: object, message: str) -> None:
    config = load_synthetic_erm_config(CONFIG)

    with pytest.raises(ValueError, match=message):
        replace(config.elastic_net, **{field: value})


def test_duplicate_or_empty_solver_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        BackendSolvers(cpu=("same", "same"), cuda=("solver",))
    with pytest.raises(ValueError, match="nonempty strings"):
        BackendSolvers(cpu=("",), cuda=("solver",))


def test_common_run_controls_are_validated() -> None:
    config = load_synthetic_erm_config(CONFIG)

    with pytest.raises(ValueError, match="repetitions"):
        replace(config, repetitions=0)
    with pytest.raises(ValueError, match="seeds"):
        replace(config, seeds=(0, 0))


def test_unknown_configuration_fields_are_rejected(tmp_path: Path) -> None:
    contents = CONFIG.read_text().replace(
        'suite = "synthetic_erm"',
        'suite = "synthetic_erm"\nunknown = 1',
    )
    path = tmp_path / "unknown.toml"
    path.write_text(contents)

    with pytest.raises(ValueError, match="unknown"):
        load_synthetic_erm_config(path)


def test_accuracy_thresholds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="stationarity"):
        AccuracyThresholds(
            stationarity=0.0,
            feasibility=1e-8,
            relative_duality_gap=1e-6,
            calibrated=False,
        )


def test_shapes_must_be_nonempty() -> None:
    config = load_synthetic_erm_config(CONFIG)

    with pytest.raises(ValueError, match="elastic-net shapes"):
        ElasticNetExperiment(
            shapes=(),
            teacher_density=config.elastic_net.teacher_density,
            noise_ratio=config.elastic_net.noise_ratio,
            teacher_intercept=config.elastic_net.teacher_intercept,
            regularization_fractions=config.elastic_net.regularization_fractions,
            vanilla_solvers=config.elastic_net.vanilla_solvers,
            bounded_solvers=config.elastic_net.bounded_solvers,
        )

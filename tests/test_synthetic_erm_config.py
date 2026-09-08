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
    load_synthetic_erm_calibration_config,
    load_synthetic_erm_config,
)


CONFIG = Path(__file__).parents[1] / "configs" / "synthetic_erm_smoke.toml"
CALIBRATION_CONFIG = Path(__file__).parents[1] / "configs" / "synthetic_erm_calibration.toml"


def test_load_synthetic_erm_smoke_config() -> None:
    config = load_synthetic_erm_config(CONFIG)

    assert isinstance(config, SyntheticErmConfig)
    assert config.suite == "synthetic_erm"
    assert config.seeds == (0, 1, 2)
    assert config.repetitions == 1
    assert config.accuracy.calibrated
    assert config.multinomial.execution.max_iterations == 10_000
    assert config.multinomial.execution.batch_size == 256
    assert config.multinomial.execution.native_tolerances_calibrated is not None
    assert config.multinomial.execution.native_tolerances_calibrated.cpu
    assert config.multinomial.execution.native_tolerances_calibrated.cuda
    assert dict(config.multinomial.execution.native_tolerances.cuda) == {
        "jaxopt_lbfgsb": 1e-6,
        "projected_gradient": 1e-6,
        "rlaopt_sapphire": 1e-7,
    }
    assert (
        config.multinomial.execution.native_tolerances.for_solver("cpu", "rlaopt_sapphire") == 1e-7
    )
    assert config.multinomial.shapes == (ErmShape(1024, 64), ErmShape(4096, 256))
    assert config.elastic_net.regularization_fractions == (0.1, 0.01)
    assert "cuclarabel_cudss" in config.elastic_net.bounded_solvers.cuda
    assert config.elastic_net.bounded_execution.max_iterations == 10_000
    assert config.elastic_net.bounded_execution.native_tolerances_calibrated is not None
    assert config.elastic_net.bounded_execution.native_tolerances_calibrated.cpu
    assert config.elastic_net.bounded_execution.native_tolerances_calibrated.cuda
    assert dict(config.elastic_net.bounded_execution.native_tolerances.cuda) == {
        "cuclarabel_cudss": 1e-10,
        "rlaopt_admm": 1e-7,
        "scs_cuda": 1e-7,
    }


def test_load_synthetic_erm_calibration_config_without_frozen_tolerances() -> None:
    calibration = load_synthetic_erm_calibration_config(CALIBRATION_CONFIG)

    assert calibration.candidates == (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
    assert calibration.experiment.accuracy.stationarity == 1e-4
    assert calibration.experiment.accuracy.feasibility == 1e-6
    for execution in (
        calibration.experiment.multinomial.execution,
        calibration.experiment.elastic_net.bounded_execution,
    ):
        assert execution.native_tolerances is None
        assert execution.native_tolerances_calibrated is None


def test_calibration_config_rejects_frozen_native_tolerances(tmp_path: Path) -> None:
    contents = CALIBRATION_CONFIG.read_text().replace(
        "[multinomial.execution]\nmax_iterations = 10000\nbatch_size = 256",
        "[multinomial.execution]\nmax_iterations = 10000\nbatch_size = 256\n"
        "[multinomial.execution.native_tolerances_calibrated]\ncpu = false\ncuda = false",
    )
    path = tmp_path / "invalid-calibration.toml"
    path.write_text(contents)

    with pytest.raises(ValueError, match="unknown fields.*native_tolerances_calibrated"):
        load_synthetic_erm_calibration_config(path)


def test_elastic_net_variants_share_one_data_grid() -> None:
    config = load_synthetic_erm_config(CONFIG)

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
            bounded_solvers=config.elastic_net.bounded_solvers,
            bounded_execution=config.elastic_net.bounded_execution,
        )


def test_retired_settings_preserve_bounded_campaign_config(tmp_path: Path) -> None:
    source = Path("configs/synthetic_erm_smoke.toml").read_text()
    legacy = source.replace("[accuracy]", "[accuracy]\nrelative_duality_gap = 1e-4")
    legacy += "\n[elastic_net.vanilla_solvers]\ncpu = ['retired_solver']\ncuda = []\n"
    legacy += "\n[elastic_net.vanilla_execution]\nmax_iterations = 100000\n"
    path = tmp_path / "legacy.toml"
    path.write_text(legacy)
    assert load_synthetic_erm_config(path) == load_synthetic_erm_config(
        Path("configs/synthetic_erm_smoke.toml")
    )

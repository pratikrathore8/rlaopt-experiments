import json
import pytest

from rlaopt_experiments.records import TrialRecord, read_record, record_view, write_record


def test_current_record_round_trips_to_flat_analysis_view(tmp_path):
    record = TrialRecord(
        run_id="run",
        problem_id="problem",
        suite="synthetic_ridge",
        solver="cg",
        backend="cpu",
        seed=7,
        repetition=0,
        timings={
            "runtime_seconds": 1.5,
            "runtime_median_seconds": 1.5,
            "runtime_min_seconds": 1.0,
            "runtime_max_seconds": 2.0,
        },
        iterations=4,
        native_status="native_converged",
        metrics={
            "relative_kkt": 1e-9,
            "relative_solution_error": 2e-9,
        },
        native_success=True,
        external_success=True,
        runtime_eligible=True,
        timed_out=False,
        problem={
            "n": 16,
            "p": 8,
            "alpha": 1.0,
            "ridge": 1e-4,
            "diagnostics": {"condition_number": 10.0},
        },
    )
    path = tmp_path / "record.json"

    write_record(path, record)
    raw = json.loads(path.read_text())
    view = read_record(path)

    assert raw["schema_version"] == 3
    assert "success" not in raw
    assert raw["native_success"]
    assert raw["external_success"]
    assert raw["runtime_eligible"]
    assert "n" not in raw
    assert view["suite"] == "synthetic_ridge"
    assert view["n"] == 16
    assert view["runtime_seconds"] == 1.5
    assert view["relative_kkt"] == 1e-9
    assert view["success"]
    assert view["diagnostics"] == {"condition_number": 10.0}


def test_legacy_ridge_record_remains_readable(tmp_path):
    legacy = {
        "run_id": "legacy",
        "problem_id": "problem",
        "solver": "cg",
        "backend": "cpu",
        "n": 16,
        "p": 8,
        "alpha": 1.0,
        "ridge": 1e-4,
        "runtime_seconds": 1.5,
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))

    view = read_record(path)

    assert view["suite"] == "synthetic_ridge"
    assert view["n"] == 16
    assert view["runtime_seconds"] == 1.5


def test_schema_two_outcomes_are_normalized(tmp_path):
    previous = {
        "schema_version": 2,
        "run_id": "previous",
        "problem_id": "problem",
        "suite": "synthetic_ridge",
        "solver": "cuml_lsmr",
        "backend": "cuda",
        "native_status": "native_complete",
        "success": False,
        "problem": {},
        "timings": {"runtime_seconds": 2.0},
        "metrics": {},
        "metadata": {"worker_outcome": "result"},
    }
    path = tmp_path / "previous.json"
    path.write_text(json.dumps(previous))

    view = read_record(path)

    assert view["native_success"]
    assert view["runtime_eligible"]
    assert not view["external_success"]
    assert not view["success"]


def test_schema_three_requires_explicit_outcome_fields():
    with pytest.raises(ValueError, match="missing outcome fields"):
        record_view({"schema_version": 3, "problem": {}, "timings": {}, "metrics": {}})

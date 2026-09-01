import json

from rlaopt_experiments.records import TrialRecord, read_record, write_record


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
            "success": True,
        },
        success=True,
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

    assert raw["schema_version"] == 2
    assert "n" not in raw
    assert view["suite"] == "synthetic_ridge"
    assert view["n"] == 16
    assert view["runtime_seconds"] == 1.5
    assert view["relative_kkt"] == 1e-9
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

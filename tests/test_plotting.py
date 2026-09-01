from rlaopt_experiments.plotting import _aggregate, _native_success


def _row(*, status: str, outcome: str, success: bool, runtime: float) -> dict:
    return {
        "solver": "cuml_lsmr",
        "backend": "cuda",
        "n": 65536,
        "p": 16384,
        "alpha": 2.0,
        "ridge": 1e-6,
        "runtime_seconds": runtime,
        "native_status": status,
        "success": success,
        "metadata": {"worker_outcome": outcome},
    }


def test_native_success_is_independent_of_external_kkt_result():
    marginal = _row(status="native_complete", outcome="result", success=False, runtime=2.0)
    timeout = _row(status="timeout", outcome="timeout", success=False, runtime=900.0)

    assert _native_success(marginal)
    assert not _native_success(timeout)


def test_runtime_aggregation_includes_native_success_and_excludes_timeout():
    records = [
        _row(status="native_complete", outcome="result", success=False, runtime=2.0),
        _row(status="native_complete", outcome="result", success=True, runtime=4.0),
        _row(status="timeout", outcome="timeout", success=False, runtime=900.0),
    ]

    [aggregate] = _aggregate(records, "fixed_p")

    assert aggregate["runtime"] == 3.0
    assert aggregate["runtime_min"] == 2.0
    assert aggregate["runtime_max"] == 4.0
    assert aggregate["native_success_rate"] == 2 / 3
    assert aggregate["external_kkt_success_rate"] == 1 / 3

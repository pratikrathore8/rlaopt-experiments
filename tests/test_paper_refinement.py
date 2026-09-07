from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[1] / "scripts/paper_refinement.py"
    spec = importlib.util.spec_from_file_location("paper_refinement", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def row(tol, seconds, native=True, external=False):
    return {
        "backend": "cuda",
        "solver": "scs_cuda_direct",
        "seed": 300,
        "native_success": native,
        "external_success": external,
        "native_status": "converged" if native else "timeout",
        "metadata": {"native_tolerance": tol},
        "runtime_seconds": seconds,
    }


def test_first_passing_setting_not_fastest_and_all_costs_retained():
    original = row(1e-7, 365.0)
    selected, audit = module().select_attempt(
        original, [row(1e-8, 800.0, external=True), row(1e-9, 600.0, external=True)]
    )
    assert selected["runtime_seconds"] == 800.0
    assert selected["total_measured_solver_seconds"] == 1765.0
    assert selected["refined"] and selected["plot_eligible"]
    assert [r["selected"] for r in audit] == [False, True, False]
    assert "plot_eligible" not in original


def test_scs_timeout_sweep_retains_original_accuracy_miss():
    selected, audit = module().select_attempt(
        row(1e-7, 365.0), [row(1e-8, 3600.0, native=False), row(1e-9, 3600.0, native=False)]
    )
    assert not selected["plot_eligible"]
    assert selected["native_success"] and not selected["external_success"]
    assert selected["refinement_last_status"] == "timeout"
    assert selected["total_measured_solver_seconds"] == 7565.0
    assert not any(r["selected"] for r in audit)


def test_pending_miss_is_not_a_passing_point():
    selected, _ = module().select_attempt(row(1e-9, 10.0), [])
    assert not selected["plot_eligible"]
    assert selected["refinement_attempts"] == 0


@pytest.mark.parametrize(
    "attempts",
    [
        [row(1e-7, 10.0)],
        [row(1e-8, 10.0), row(1e-8, 10.0)],
        [row(1e-9, 10.0), row(1e-8, 10.0)],
    ],
)
def test_rejects_duplicate_or_unordered_tolerances(attempts):
    with pytest.raises(ValueError, match="strictly decreasing"):
        module().select_attempt(row(1e-7, 10.0), attempts)


def test_rejects_refinement_of_original_success():
    with pytest.raises(ValueError, match="accuracy miss"):
        module().select_attempt(row(1e-7, 10.0, external=True), [row(1e-8, 5.0)])

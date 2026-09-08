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


def test_scs_timeout_sweep_displays_final_timeout_and_retains_original_in_audit():
    selected, audit = module().select_attempt(
        row(1e-7, 365.0), [row(1e-8, 3600.0, native=False), row(1e-9, 3600.0, native=False)]
    )
    assert not selected["plot_eligible"]
    assert not selected["native_success"] and not selected["external_success"]
    assert selected["native_status"] == "timeout"
    assert selected["runtime_seconds"] == 3600.0
    assert selected["displayed_native_tolerance"] == 1e-9
    assert audit[0]["native_success"] and not audit[0]["external_success"]
    assert [r["displayed"] for r in audit] == [False, False, True]
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


@pytest.mark.parametrize(
    ("native", "external", "expected"),
    [
        (True, False, "ACC"),
        (False, False, "TO"),
        (False, True, "TO"),
        (True, True, "OK"),
    ],
)
def test_final_attempt_classification(native, external, expected, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "paper_figures", scripts / "make_paper_figures.py"
    )
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    selected, audit = module().select_attempt(
        row(1e-7, 365.0), [row(1e-8, 3600.0, native=native, external=external)]
    )
    assert figures.status(selected) == expected
    assert figures.success(selected) == (native and external)
    assert len(audit) == 2


def test_scs_order_and_colors_are_consistent(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "paper_figures", scripts / "make_paper_figures.py"
    )
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    sets = figures.real_solver_sets(
        [{"solver": "scs_cpu_indirect"}, {"solver": "scs_cuda_direct"}], "bounded_elastic_net"
    )
    for backend in ("cpu", "cuda"):
        assert [figures.LABELS[s] for s in sets[backend]][1:3] == ["SCS indirect", "SCS direct"]
    assert figures.SOLVER_COLORS["scs"] == figures.SOLVER_COLORS["scs_cuda_direct"]
    assert figures.SOLVER_COLORS["scs_cpu_indirect"] == figures.SOLVER_COLORS["scs_cuda"]


@pytest.mark.parametrize(
    "changed",
    ["stationarity_tolerance", "feasibility_tolerance", "source_sha256", "solver_seed", "problem"],
)
def test_real_refinement_rejects_changed_problem_or_accuracy(tmp_path, changed):
    import copy
    import json

    original = row(1e-7, 365.0) | {"problem_id": "case", "problem": {"n": 10, "p": 5}}
    original["metadata"].update(
        {
            "stationarity_tolerance": 1e-4,
            "feasibility_tolerance": 1e-6,
            "source_sha256": "frozen",
            "solver_seed": 12,
        }
    )
    attempt = copy.deepcopy(original)
    attempt["metadata"]["native_tolerance"] = 1e-8
    if changed == "problem":
        attempt["problem"]["n"] = 20
    else:
        attempt["metadata"][changed] = "changed"
    records = tmp_path / "results" / "records"
    records.mkdir(parents=True)
    (records / "result.json").write_text(json.dumps(attempt))

    def key(r):
        return r["solver"], r["problem_id"], r["seed"]

    with pytest.raises(ValueError, match="changed"):
        module().apply_real_refinement({key(original): original}, [tmp_path], key, [])

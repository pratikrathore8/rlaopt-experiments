from __future__ import annotations

from dataclasses import replace
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlaopt_experiments.problems.synthetic_erm import (
    ElasticNetSpec,
    generate_elastic_net_problem,
)
from rlaopt_experiments.suites.synthetic_erm.bounded_elastic_net_solvers import (
    _build_rlaopt_objective,
    build_bounded_elastic_net_conic_form,
    solve_clarabel_qdldl,
    solve_rlaopt_admm,
    solve_scs,
    solve_scs_cuda,
)
from rlaopt_experiments.clarabel_bridge import ClarabelRuntime


class FakeClarabelRuntime:
    backend = "cpu"

    def __init__(self, solution):
        self.solution = solution
        self.prepared = None
        self.controls = None

    def prepare(self, quadratic, linear, constraints, rhs):
        self.prepared = (quadratic, linear, constraints, rhs)
        return self.prepared

    def solve(self, data, **controls):
        assert data is self.prepared
        self.controls = controls
        return SimpleNamespace(solution=self.solution)


class FakeVectorFactory:
    def __getitem__(self, _dtype):
        return lambda values: np.asarray(values).copy()


class FakeJuliaMain:
    Float64 = "Float64"
    Int64 = "Int64"
    Vector = FakeVectorFactory()

    @staticmethod
    def SparseMatrixCSC(rows, columns, column_pointers, row_indices, values):
        return (rows, columns, column_pointers, row_indices, values)


def test_importing_clarabel_bridge_does_not_import_torch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rlaopt_experiments.clarabel_bridge; "
            "assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.fixture(scope="module")
def problem():
    return generate_elastic_net_problem(
        ElasticNetSpec(
            n=64,
            p=8,
            feature_seed=31,
            target_seed=32,
            teacher_density=0.25,
            noise_ratio=0.1,
            teacher_intercept=0.5,
            regularization_fraction=0.1,
        ),
        bounded=True,
        device="cpu",
    )


def test_conic_form_is_exactly_the_canonical_bounded_problem(problem) -> None:
    conic = build_bounded_elastic_net_conic_form(problem)
    weights = torch.linspace(0.1, 0.8, problem.spec.p, dtype=torch.float64)
    intercept = torch.tensor(-0.2, dtype=torch.float64)
    residual = problem.X @ weights + intercept - problem.y
    primal = torch.cat((residual, weights, intercept.reshape(1))).numpy()

    conic_objective = 0.5 * primal @ conic.quadratic @ primal + conic.linear @ primal
    assert conic_objective == pytest.approx(float(problem.objective(weights, intercept)))
    np.testing.assert_allclose(
        conic.constraints[: problem.spec.n] @ primal,
        conic.rhs[: problem.spec.n],
        rtol=1e-14,
        atol=1e-14,
    )
    assert np.max(conic.constraints[problem.spec.n :] @ primal - conic.rhs[problem.spec.n :]) <= 0
    assert conic.zero_cone_dim == problem.spec.n
    assert conic.nonnegative_cone_dim == 2 * problem.spec.p


def test_rlaopt_native_objective_matches_canonical_inside_box(problem) -> None:
    from rlaopt.data import DataLoader, Dataset

    loader = DataLoader(
        Dataset(problem.X, problem.y, device=problem.X.device, dtype=torch.float64),
        batch_size=problem.spec.n,
        shuffle=False,
    )
    native_objective, weights, intercept = _build_rlaopt_objective(problem, loader)
    candidate_weights = torch.linspace(0.1, 0.8, problem.spec.p, dtype=torch.float64)
    candidate_intercept = torch.tensor(-0.2, dtype=torch.float64)
    weights.value = candidate_weights
    intercept.value = candidate_intercept

    torch.testing.assert_close(
        native_objective.forward(),
        problem.objective(candidate_weights, candidate_intercept),
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    ("adapter", "extra"),
    [
        (solve_scs, {}),
        (solve_rlaopt_admm, {"batch_size": 32, "seed": 33}),
    ],
)
def test_cpu_adapters_solve_the_same_canonical_problem(problem, adapter, extra) -> None:
    result = adapter(
        problem,
        native_tolerance=1e-8,
        max_iterations=5_000,
        **extra,
    )

    assert result.native_status == "converged"
    assert result.weights.shape == (problem.spec.p,)
    assert result.weights.dtype == torch.float64
    assert result.intercept.shape == ()
    assert result.intercept.dtype == torch.float64
    assert result.runtime_seconds > 0
    assert problem.constraint_violation(result.weights) <= 1e-8
    assert (
        problem.kkt_residual(
            result.weights,
            result.intercept,
            activity_tolerance=1e-8,
        )
        <= 1e-6
    )


def test_rlaopt_admm_uses_default_float64_nystrom_path(problem) -> None:
    previous_default_dtype = torch.get_default_dtype()
    result = solve_rlaopt_admm(
        problem,
        native_tolerance=1e-8,
        max_iterations=5_000,
        batch_size=32,
        seed=33,
    )

    assert torch.get_default_dtype() == previous_default_dtype
    assert result.metadata["native_objective_scale"] == 1.0
    assert result.metadata["nystrom_rank"] == 50
    assert result.metadata["torch_default_dtype_workaround"] is True
    assert result.metadata["primal_residual_norm"] >= 0
    assert result.metadata["dual_residual_norm"] >= 0


def test_scs_records_direct_native_diagnostics(problem) -> None:
    result = solve_scs(
        problem,
        native_tolerance=1e-8,
        max_iterations=5_000,
    )

    assert result.native_error is None
    assert result.metadata["conic_preparation_seconds"] >= 0
    assert result.metadata["linear_solver"] == "cpu_direct"
    assert result.metadata["raw_status"] == "solved"
    assert result.metadata["native_setup_time_milliseconds"] >= 0
    assert result.metadata["native_solve_time_milliseconds"] >= 0
    assert result.metadata["native_primal_residual"] >= 0
    assert result.metadata["native_dual_residual"] >= 0


def test_clarabel_cpu_adapter_records_controls_and_native_diagnostics(problem) -> None:
    reference = solve_scs(problem, native_tolerance=1e-10, max_iterations=5_000)
    residual = problem.X @ reference.weights + reference.intercept - problem.y
    primal = torch.cat((residual, reference.weights, reference.intercept.reshape(1))).numpy()
    objective = float(problem.objective(reference.weights, reference.intercept))
    solution = SimpleNamespace(
        x=primal,
        status="SOLVED",
        iterations=9,
        r_prim=2e-9,
        r_dual=3e-9,
        obj_val=objective,
        obj_val_dual=objective - 4e-9,
        setup_phase_time=0.02,
        solve_phase_time=0.03,
    )
    runtime = FakeClarabelRuntime(solution)

    result = solve_clarabel_qdldl(
        problem,
        runtime=runtime,
        native_tolerance=1e-8,
        max_iterations=123,
    )

    assert result.native_status == "converged"
    assert result.native_error is None
    assert result.iterations == 9
    torch.testing.assert_close(result.weights, reference.weights)
    torch.testing.assert_close(result.intercept, reference.intercept)
    assert runtime.controls == {
        "equality_dim": problem.spec.n,
        "inequality_dim": 2 * problem.spec.p,
        "native_tolerance": 1e-8,
        "max_iterations": 123,
    }
    assert result.metadata["linear_solver"] == "qdldl"
    assert result.metadata["native_primal_residual"] == pytest.approx(2e-9)
    assert result.metadata["native_dual_residual"] == pytest.approx(3e-9)
    assert result.metadata["native_setup_time_seconds"] == pytest.approx(0.02)
    assert result.metadata["native_solve_time_seconds"] == pytest.approx(0.03)
    assert result.metadata["conic_preparation_seconds"] >= 0
    assert result.metadata["solution_extraction_seconds"] >= 0


def test_clarabel_does_not_treat_almost_solved_as_native_convergence(problem) -> None:
    variable_count = problem.spec.n + problem.spec.p + 1
    solution = SimpleNamespace(
        x=np.zeros(variable_count, dtype=np.float64),
        status="ALMOST_SOLVED",
        iterations=10,
        r_prim=1e-6,
        r_dual=1e-6,
        obj_val=1.0,
        obj_val_dual=0.9,
        setup_phase_time=0.0,
        solve_phase_time=0.0,
    )
    result = solve_clarabel_qdldl(
        problem,
        runtime=FakeClarabelRuntime(solution),
        native_tolerance=1e-8,
        max_iterations=10,
    )

    assert result.native_status == "ALMOST_SOLVED"


def test_cpu_clarabel_bridge_preserves_sparse_inputs_and_uses_one_based_indices(problem) -> None:
    conic = build_bounded_elastic_net_conic_form(problem)
    original_indptr = conic.constraints.indptr.copy()
    original_indices = conic.constraints.indices.copy()
    runtime = ClarabelRuntime(FakeJuliaMain(), "cpu")

    prepared = runtime.prepare(
        conic.quadratic,
        conic.linear,
        conic.constraints,
        conic.rhs,
    )

    _, _, column_pointers, row_indices, _ = prepared.constraints
    np.testing.assert_array_equal(column_pointers, original_indptr + 1)
    np.testing.assert_array_equal(row_indices, original_indices + 1)
    np.testing.assert_array_equal(conic.constraints.indptr, original_indptr)
    np.testing.assert_array_equal(conic.constraints.indices, original_indices)
    assert prepared.owners == ()


def test_clarabel_adapter_rejects_runtime_for_other_backend(problem) -> None:
    runtime = FakeClarabelRuntime(None)
    runtime.backend = "cuda"
    with pytest.raises(ValueError, match="runtime backend"):
        solve_clarabel_qdldl(
            problem,
            runtime=runtime,
            native_tolerance=1e-8,
            max_iterations=100,
        )


def test_scs_does_not_treat_inaccurate_status_as_native_convergence(
    problem, monkeypatch: pytest.MonkeyPatch
) -> None:
    variable_count = problem.spec.n + problem.spec.p + 1

    def inaccurate_result(*_args, **_kwargs):
        return {
            "x": np.zeros(variable_count, dtype=np.float64),
            "info": {
                "status": "solved (inaccurate - reached max_iters)",
                "iter": 10,
                "res_pri": 1e-3,
                "res_dual": 1e-3,
                "gap": 1e-3,
                "setup_time": 0.0,
                "solve_time": 0.0,
            },
        }

    monkeypatch.setattr("scs.solve", inaccurate_result)
    result = solve_scs(problem, native_tolerance=1e-8, max_iterations=10)

    assert result.native_status != "converged"
    assert "inaccurate" in result.native_status


def test_scs_cuda_never_falls_back_when_gpu_module_is_missing(
    problem, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "rlaopt_experiments.suites.synthetic_erm.bounded_elastic_net_solvers._require_device",
        lambda *_args: None,
    )
    monkeypatch.setitem(sys.modules, "scs._scs_gpu", None)

    with pytest.raises(RuntimeError, match="not built with its CUDA indirect backend"):
        solve_scs_cuda(
            problem,
            native_tolerance=1e-8,
            max_iterations=100,
        )


def test_bounded_adapters_reject_unbounded_problem(problem) -> None:
    unbounded = replace(problem, bounded=False)
    with pytest.raises(ValueError, match="bounded problem"):
        solve_scs(
            unbounded,
            native_tolerance=1e-8,
            max_iterations=100,
        )


def test_bounded_adapters_reject_invalid_controls(problem) -> None:
    with pytest.raises(ValueError, match="native_tolerance"):
        solve_scs(
            problem,
            native_tolerance=0.0,
            max_iterations=100,
        )
    with pytest.raises(ValueError, match="batch_size"):
        solve_rlaopt_admm(
            problem,
            native_tolerance=1e-8,
            max_iterations=100,
            batch_size=0,
            seed=33,
        )

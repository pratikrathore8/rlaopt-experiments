import pytest
import torch

from rlaopt_experiments.problem import ProblemSpec, generate_problem, relative_kkt


def test_prescribed_spectrum_and_oracle():
    problem = generate_problem(ProblemSpec(16, 8, 1.0, 11, 12))
    actual = torch.linalg.svdvals(problem.X)
    torch.testing.assert_close(actual, problem.singular_values, rtol=1e-11, atol=1e-12)
    assert torch.linalg.vector_norm(problem.y).item() == pytest.approx(1.0)
    assert relative_kkt(problem, problem.oracle(1e-3), 1e-3) < 1e-11


def test_condition_number_uses_smallest_positive_eigenvalue_for_tall_problem():
    problem = generate_problem(ProblemSpec(16, 8, 1.0, 11, 12))
    expected = (1.0 + 1e-3) / (8.0**-1 + 1e-3)
    assert problem.diagnostics(1e-3)["full_condition_number"] == pytest.approx(expected)


def test_condition_number_includes_nullspace_for_wide_problem():
    problem = generate_problem(ProblemSpec(8, 16, 1.0, 11, 12))
    expected = (1.0 + 1e-3) / 1e-3
    assert problem.diagnostics(1e-3)["full_condition_number"] == pytest.approx(expected)

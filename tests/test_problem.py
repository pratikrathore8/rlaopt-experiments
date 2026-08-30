import torch

from rlaopt_experiments.problem import ProblemSpec, generate_problem, relative_kkt


def test_prescribed_spectrum_and_oracle():
    problem = generate_problem(ProblemSpec(16, 8, 1.0, 11, 12))
    actual = torch.linalg.svdvals(problem.X)
    torch.testing.assert_close(actual, problem.singular_values, rtol=1e-11, atol=1e-12)
    assert torch.linalg.vector_norm(problem.y).item() == pytest.approx(1.0)
    assert relative_kkt(problem, problem.oracle(1e-3), 1e-3) < 1e-11


import pytest

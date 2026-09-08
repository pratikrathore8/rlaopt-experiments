import math
from dataclasses import replace

import pytest
import torch

from rlaopt_experiments.problems import (
    ElasticNetSpec,
    MultinomialSpec,
    generate_elastic_net_problem,
    generate_multinomial_problem,
)


def _multinomial_spec() -> MultinomialSpec:
    return MultinomialSpec(
        n=128,
        p=16,
        n_classes=4,
        feature_seed=11,
        target_seed=12,
        feature_generator="standardized_gaussian",
        feature_decay_exponent=None,
        teacher_scale=1.5,
        box_lower=-1.0,
        box_upper=1.0,
    )


def _elastic_net_spec() -> ElasticNetSpec:
    return ElasticNetSpec(
        n=128,
        p=16,
        feature_seed=21,
        target_seed=22,
        feature_generator="standardized_gaussian",
        feature_decay_exponent=None,
        teacher_density=0.25,
        noise_ratio=0.1,
        teacher_intercept=0.4,
        regularization_fraction=0.1,
    )


def test_multinomial_generation_is_deterministic_standardized_and_feasible():
    first = generate_multinomial_problem(_multinomial_spec(), device="cpu")
    second = generate_multinomial_problem(_multinomial_spec(), device="cpu")

    torch.testing.assert_close(first.X, second.X, rtol=0, atol=0)
    torch.testing.assert_close(first.y, second.y, rtol=0, atol=0)
    torch.testing.assert_close(first.teacher, second.teacher, rtol=0, atol=0)
    assert first.X.dtype == torch.float64
    assert first.y.dtype == torch.int64
    torch.testing.assert_close(
        first.X.mean(dim=0), torch.zeros(16, dtype=torch.float64), atol=1e-15, rtol=0
    )
    torch.testing.assert_close(
        first.X.square().mean(dim=0),
        torch.ones(16, dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )
    assert int(first.y.min()) >= 0
    assert int(first.y.max()) < first.spec.n_classes
    assert float(first.teacher.abs().max()) <= 0.8
    assert sum(first.diagnostics()["class_counts"]) == first.spec.n


def test_sorf_features_have_exact_normalized_power_law_spectrum():
    alpha = 1.0
    spec = replace(
        _multinomial_spec(),
        feature_generator="sorf_power_law",
        feature_decay_exponent=alpha,
    )
    first = generate_multinomial_problem(spec, device="cpu")
    second = generate_multinomial_problem(spec, device="cpu")
    rank = min(spec.n, spec.p)
    expected = torch.arange(1, rank + 1, dtype=torch.float64).pow(-alpha / 2)
    expected *= math.sqrt(spec.n * spec.p) / torch.linalg.vector_norm(expected)

    torch.testing.assert_close(first.X, second.X, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.linalg.svdvals(first.X),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    assert float(torch.linalg.vector_norm(first.X)) == pytest.approx(
        math.sqrt(spec.n * spec.p), rel=1e-13
    )
    diagnostics = first.diagnostics()
    assert diagnostics["feature_generator"] == "sorf_power_law"
    assert diagnostics["feature_decay_exponent"] == alpha


def test_multinomial_gradient_matches_autograd():
    problem = generate_multinomial_problem(_multinomial_spec(), device="cpu")
    coefficients = torch.randn(
        (problem.spec.p, problem.spec.n_classes),
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(13),
        requires_grad=True,
    )

    objective = problem.objective(coefficients)
    [automatic] = torch.autograd.grad(objective, coefficients)

    torch.testing.assert_close(problem.gradient(coefficients.detach()), automatic)
    feasible = coefficients.detach().clamp(problem.spec.box_lower, problem.spec.box_upper)
    assert float(problem.constraint_violation(feasible)) == 0.0
    infeasible = feasible.clone()
    infeasible[0, 0] = problem.spec.box_upper + 0.25
    assert float(problem.constraint_violation(infeasible)) == pytest.approx(0.25)

    mixed = torch.zeros_like(coefficients.detach())
    mixed[0] = problem.spec.box_lower
    mixed[1] = problem.spec.box_upper
    gradient = problem.gradient(mixed)
    expected = torch.cat(
        (
            torch.relu(-gradient[0]),
            torch.relu(gradient[1]),
            gradient[2:].abs().ravel(),
        )
    ).max()
    torch.testing.assert_close(
        problem.kkt_residual(mixed, activity_tolerance=1e-8),
        expected,
    )


def test_bounded_elastic_net_generation_is_reproducible_and_scaled():
    spec = _elastic_net_spec()
    problem = generate_elastic_net_problem(spec, bounded=True, device="cpu")
    bounded = generate_elastic_net_problem(spec, bounded=True, device="cpu")

    torch.testing.assert_close(problem.X, bounded.X, rtol=0, atol=0)
    torch.testing.assert_close(problem.y, bounded.y, rtol=0, atol=0)
    torch.testing.assert_close(
        problem.teacher_weights,
        bounded.teacher_weights,
        rtol=0,
        atol=0,
    )
    assert problem.X.dtype == torch.float64
    torch.testing.assert_close(
        problem.X.mean(dim=0), torch.zeros(16, dtype=torch.float64), atol=1e-15, rtol=0
    )
    assert float(problem.y.mean()) == pytest.approx(float(problem.teacher_intercept))
    assert float(problem.y.var(correction=0)) == pytest.approx(1.0)
    assert int(torch.count_nonzero(problem.teacher_weights)) == math.ceil(
        spec.teacher_density * spec.p
    )
    effective_signal = problem.X @ problem.teacher_weights
    realized_noise = problem.y - effective_signal - problem.teacher_intercept
    realized_ratio = realized_noise.square().mean().sqrt() / effective_signal.square().mean().sqrt()
    assert float(realized_ratio) == pytest.approx(spec.noise_ratio)

    centered_x = problem.X - problem.X.mean(dim=0)
    centered_y = problem.y - problem.y.mean()
    expected_lambda_max = float(
        torch.linalg.vector_norm(centered_x.mT @ centered_y, ord=float("inf")) / spec.n
    )
    assert problem.lambda_max == pytest.approx(expected_lambda_max)
    assert problem.lambda_l1 == pytest.approx(spec.regularization_fraction * expected_lambda_max)
    assert problem.lambda_l2 == pytest.approx(problem.lambda_l1)


def test_elastic_net_loss_gradient_matches_autograd_and_constraint_is_measured():
    problem = generate_elastic_net_problem(
        _elastic_net_spec(),
        bounded=True,
        device="cpu",
    )
    weights = torch.randn(
        problem.spec.p,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(23),
        requires_grad=True,
    )
    intercept = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    residual = problem.X @ weights + intercept - problem.y
    loss = 0.5 * residual.square().mean()
    automatic_weights, automatic_intercept = torch.autograd.grad(
        loss,
        (weights, intercept),
    )

    weight_gradient, intercept_gradient = problem.loss_gradient(
        weights.detach(),
        intercept.detach(),
    )
    torch.testing.assert_close(weight_gradient, automatic_weights)
    torch.testing.assert_close(intercept_gradient, automatic_intercept)
    assert float(problem.constraint_violation(weights.detach().clamp(0.0, 1.0))) == 0.0
    infeasible = weights.detach().clamp(0.0, 1.0)
    infeasible[0] = -0.25
    assert float(problem.constraint_violation(infeasible)) == pytest.approx(0.25)


def test_bounded_elastic_net_zero_solution_satisfies_kkt_at_lambda_max():
    spec = replace(_elastic_net_spec(), regularization_fraction=1.0)
    problem = generate_elastic_net_problem(spec, bounded=True, device="cpu")
    weights = torch.zeros(spec.p, dtype=torch.float64)
    assert (
        float(
            problem.kkt_residual(
                weights,
                problem.y.mean(),
                activity_tolerance=1e-8,
            )
        )
        < 1e-14
    )


def test_elastic_net_generator_rejects_retired_unbounded_variant():
    with pytest.raises(ValueError, match="Only bounded"):
        generate_elastic_net_problem(_elastic_net_spec(), bounded=False, device="cpu")


def test_metric_activity_tolerances_are_explicit_and_validated():
    multinomial = generate_multinomial_problem(_multinomial_spec(), device="cpu")
    coefficients = torch.zeros(
        (multinomial.spec.p, multinomial.spec.n_classes),
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        multinomial.kkt_residual(coefficients, activity_tolerance=-1.0)
    with pytest.raises(ValueError, match="bounds overlap"):
        multinomial.kkt_residual(coefficients, activity_tolerance=1.0)

    elastic_net = generate_elastic_net_problem(
        _elastic_net_spec(),
        bounded=True,
        device="cpu",
    )
    with pytest.raises(ValueError, match="bounds overlap"):
        elastic_net.kkt_residual(
            torch.zeros(elastic_net.spec.p, dtype=torch.float64),
            0.0,
            activity_tolerance=0.5,
        )


def test_synthetic_specs_reject_implicit_or_invalid_scales():
    with pytest.raises(ValueError, match="regularization_fraction"):
        ElasticNetSpec(
            n=16,
            p=4,
            feature_seed=1,
            target_seed=2,
            feature_generator="standardized_gaussian",
            feature_decay_exponent=None,
            teacher_density=0.25,
            noise_ratio=0.1,
            teacher_intercept=0.0,
            regularization_fraction=0.0,
        )
    with pytest.raises(ValueError, match="bounds"):
        MultinomialSpec(
            n=16,
            p=4,
            n_classes=3,
            feature_seed=1,
            target_seed=2,
            feature_generator="standardized_gaussian",
            feature_decay_exponent=None,
            teacher_scale=1.0,
            box_lower=0.0,
            box_upper=1.0,
        )
    with pytest.raises(ValueError, match="feature generator"):
        replace(_multinomial_spec(), feature_decay_exponent=1.0)
    with pytest.raises(ValueError, match="feature generator"):
        replace(
            _elastic_net_spec(),
            feature_generator="sorf_power_law",
            feature_decay_exponent=None,
        )

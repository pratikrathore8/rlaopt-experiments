"""Deterministic float64 synthetic problems for ERM solver development."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from rlaopt_experiments.structured_orthogonal import materialize_sorf_matrix

GENERATOR_VERSION = "synthetic-erm2"


def _validate_shape(n: int, p: int) -> None:
    if n < 2:
        raise ValueError("n must be at least two")
    if p < 1:
        raise ValueError("p must be positive")


def _validate_tolerance(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _cpu_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _features_configuration(generator: str, decay_exponent: float | None) -> None:
    if generator == "standardized_gaussian" and decay_exponent is None:
        return
    if (
        generator == "sorf_power_law"
        and decay_exponent is not None
        and math.isfinite(decay_exponent)
        and decay_exponent >= 0
    ):
        return
    raise ValueError("feature generator and decay exponent are inconsistent")


def _feature_id(generator: str, decay_exponent: float | None) -> str:
    if generator == "standardized_gaussian":
        return "fg-gaussian"
    return f"fg-sorf-a{decay_exponent:g}"


def _standardized_gaussian(n: int, p: int, seed: int) -> torch.Tensor:
    """Return Gaussian features with exactly zero mean and unit population RMS."""
    features = torch.randn((n, p), dtype=torch.float64, generator=_cpu_generator(seed))
    features -= features.mean(dim=0)
    scales = features.square().mean(dim=0).sqrt()
    if bool(torch.any(scales == 0)):
        raise RuntimeError("generated a constant feature column")
    return features / scales


def _features(
    n: int,
    p: int,
    seed: int,
    generator: str,
    decay_exponent: float | None,
) -> torch.Tensor:
    if generator == "standardized_gaussian":
        if decay_exponent is not None:
            raise ValueError("Gaussian features cannot specify a decay exponent")
        return _standardized_gaussian(n, p, seed)
    if generator != "sorf_power_law":
        raise ValueError(f"unknown feature generator: {generator}")
    if decay_exponent is None or not math.isfinite(decay_exponent) or decay_exponent < 0:
        raise ValueError("SORF power-law features require a finite nonnegative exponent")

    rank = min(n, p)
    indices = torch.arange(1, rank + 1, dtype=torch.float64)
    singular_values = indices.pow(-decay_exponent / 2)
    singular_values *= math.sqrt(n * p) / torch.linalg.vector_norm(singular_values)
    features, _, _ = materialize_sorf_matrix(
        n,
        p,
        singular_values,
        seed=seed,
        device="cpu",
    )
    return features


@dataclass(frozen=True)
class MultinomialSpec:
    """Inputs that uniquely identify one box-constrained multinomial problem."""

    n: int
    p: int
    n_classes: int
    feature_seed: int
    target_seed: int
    feature_generator: str
    feature_decay_exponent: float | None
    teacher_scale: float
    box_lower: float
    box_upper: float

    def __post_init__(self) -> None:
        _validate_shape(self.n, self.p)
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least two")
        _features_configuration(self.feature_generator, self.feature_decay_exponent)
        if self.teacher_scale <= 0:
            raise ValueError("teacher_scale must be positive")
        if not self.box_lower < 0 < self.box_upper:
            raise ValueError("multinomial coefficient bounds must contain zero")

    @property
    def problem_id(self) -> str:
        return (
            f"{GENERATOR_VERSION}-multinomial-n{self.n}-p{self.p}-k{self.n_classes}"
            f"-x{self.feature_seed}-y{self.target_seed}"
            f"-{_feature_id(self.feature_generator, self.feature_decay_exponent)}"
            f"-s{self.teacher_scale:g}"
            f"-lo{self.box_lower:g}-hi{self.box_upper:g}"
        )


@dataclass
class MultinomialProblem:
    """Average cross-entropy with elementwise bounds on the coefficient matrix."""

    spec: MultinomialSpec
    X: torch.Tensor
    y: torch.Tensor
    teacher: torch.Tensor

    def objective(self, coefficients: torch.Tensor) -> torch.Tensor:
        self._validate_coefficients(coefficients)
        return functional.cross_entropy(self.X @ coefficients, self.y, reduction="mean")

    def gradient(self, coefficients: torch.Tensor) -> torch.Tensor:
        self._validate_coefficients(coefficients)
        probabilities = torch.softmax(self.X @ coefficients, dim=1)
        targets = functional.one_hot(self.y, num_classes=self.spec.n_classes)
        return self.X.mT @ (probabilities - targets) / self.spec.n

    def constraint_violation(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Return the largest elementwise violation of the coefficient box."""
        self._validate_coefficients(coefficients)
        lower = torch.clamp(self.spec.box_lower - coefficients, min=0.0)
        upper = torch.clamp(coefficients - self.spec.box_upper, min=0.0)
        return torch.maximum(lower.max(), upper.max())

    def kkt_residual(
        self,
        coefficients: torch.Tensor,
        *,
        activity_tolerance: float,
    ) -> torch.Tensor:
        """Return the largest box KKT stationarity violation.

        Coefficients near a bound use that bound's one-sided stationarity
        condition. Feasibility is measured separately.
        """
        self._validate_coefficients(coefficients)
        _validate_tolerance(activity_tolerance, "activity_tolerance")
        if 2 * activity_tolerance >= self.spec.box_upper - self.spec.box_lower:
            raise ValueError("activity_tolerance makes the coefficient bounds overlap")
        gradient = self.gradient(coefficients)
        at_lower = coefficients <= self.spec.box_lower + activity_tolerance
        at_upper = coefficients >= self.spec.box_upper - activity_tolerance
        violations = torch.where(
            at_lower,
            torch.relu(-gradient),
            torch.where(at_upper, torch.relu(gradient), gradient.abs()),
        )
        return violations.max()

    def diagnostics(self) -> dict[str, float | list[int]]:
        counts = torch.bincount(self.y, minlength=self.spec.n_classes)
        return {
            "class_counts": counts.cpu().tolist(),
            "teacher_max_abs": float(self.teacher.abs().max()),
            "box_lower": self.spec.box_lower,
            "box_upper": self.spec.box_upper,
            "feature_generator": self.spec.feature_generator,
            "feature_decay_exponent": self.spec.feature_decay_exponent,
            "feature_frobenius_norm": float(torch.linalg.vector_norm(self.X)),
        }

    def _validate_coefficients(self, coefficients: torch.Tensor) -> None:
        expected = (self.spec.p, self.spec.n_classes)
        if coefficients.shape != expected:
            raise ValueError(f"coefficients must have shape {expected}")


def generate_multinomial_problem(
    spec: MultinomialSpec, device: torch.device | str
) -> MultinomialProblem:
    """Generate standardized Gaussian features and labels from a softmax teacher.

    The model has no intercept. Only the coefficient matrix is constrained.
    """
    features = _features(
        spec.n,
        spec.p,
        spec.feature_seed,
        spec.feature_generator,
        spec.feature_decay_exponent,
    )
    target_rng = _cpu_generator(spec.target_seed)
    teacher = torch.randn(
        (spec.p, spec.n_classes),
        dtype=torch.float64,
        generator=target_rng,
    )
    teacher -= teacher.mean(dim=1, keepdim=True)
    teacher *= spec.teacher_scale / math.sqrt(spec.p)
    maximum = float(teacher.abs().max())
    interior_limit = 0.8 * min(-spec.box_lower, spec.box_upper)
    if maximum > interior_limit:
        teacher *= interior_limit / maximum

    probabilities = torch.softmax(features @ teacher, dim=1)
    labels = torch.multinomial(
        probabilities,
        num_samples=1,
        replacement=True,
        generator=target_rng,
    ).squeeze(1)
    target = torch.device(device)
    return MultinomialProblem(
        spec=spec,
        X=features.to(target),
        y=labels.to(target),
        teacher=teacher.to(target),
    )


@dataclass(frozen=True)
class ElasticNetSpec:
    """Inputs shared by the unconstrained and bounded elastic-net problems."""

    n: int
    p: int
    feature_seed: int
    target_seed: int
    feature_generator: str
    feature_decay_exponent: float | None
    teacher_density: float
    noise_ratio: float
    teacher_intercept: float
    regularization_fraction: float

    def __post_init__(self) -> None:
        _validate_shape(self.n, self.p)
        _features_configuration(self.feature_generator, self.feature_decay_exponent)
        if not 0 < self.teacher_density <= 1:
            raise ValueError("teacher_density must lie in (0, 1]")
        if self.noise_ratio < 0:
            raise ValueError("noise_ratio must be nonnegative")
        if self.regularization_fraction <= 0:
            raise ValueError("regularization_fraction must be positive")

    @property
    def data_id(self) -> str:
        return (
            f"{GENERATOR_VERSION}-regression-n{self.n}-p{self.p}"
            f"-x{self.feature_seed}-y{self.target_seed}"
            f"-{_feature_id(self.feature_generator, self.feature_decay_exponent)}"
            f"-d{self.teacher_density:g}"
            f"-noise{self.noise_ratio:g}-b{self.teacher_intercept:g}"
        )

    def problem_id(self, *, bounded: bool) -> str:
        variant = "bounded" if bounded else "unbounded"
        return f"{self.data_id}-{variant}-rf{self.regularization_fraction:g}"


@dataclass
class ElasticNetProblem:
    """Average squared loss plus elastic net, optionally constrained to [0, 1]."""

    spec: ElasticNetSpec
    X: torch.Tensor
    y: torch.Tensor
    teacher_weights: torch.Tensor
    teacher_intercept: torch.Tensor
    lambda_max: float
    lambda_l1: float
    lambda_l2: float
    bounded: bool

    @property
    def problem_id(self) -> str:
        return self.spec.problem_id(bounded=self.bounded)

    def objective(self, weights: torch.Tensor, intercept: torch.Tensor | float) -> torch.Tensor:
        self._validate_weights(weights)
        residual = self.X @ weights + intercept - self.y
        return (
            0.5 * residual.square().mean()
            + self.lambda_l1 * weights.abs().sum()
            + 0.5 * self.lambda_l2 * weights.square().sum()
        )

    def loss_gradient(
        self, weights: torch.Tensor, intercept: torch.Tensor | float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gradient of the differentiable average squared-loss term."""
        self._validate_weights(weights)
        residual = self.X @ weights + intercept - self.y
        return self.X.mT @ residual / self.spec.n, residual.mean()

    def constraint_violation(self, weights: torch.Tensor) -> torch.Tensor:
        """Return zero when unbounded, otherwise the largest [0, 1] violation."""
        self._validate_weights(weights)
        if not self.bounded:
            return torch.zeros((), dtype=weights.dtype, device=weights.device)
        lower = torch.clamp(-weights, min=0.0)
        upper = torch.clamp(weights - 1.0, min=0.0)
        return torch.maximum(lower.max(), upper.max())

    def kkt_residual(
        self,
        weights: torch.Tensor,
        intercept: torch.Tensor | float,
        *,
        activity_tolerance: float,
    ) -> torch.Tensor:
        """Return the largest primal KKT stationarity violation.

        For unbounded elastic net, weights near zero use the l1 subgradient
        condition. For bounded elastic net, weights near zero or one use the
        corresponding one-sided box condition. Feasibility is separate.
        """
        self._validate_weights(weights)
        _validate_tolerance(activity_tolerance, "activity_tolerance")
        loss_gradient, intercept_gradient = self.loss_gradient(weights, intercept)
        smooth_gradient = loss_gradient + self.lambda_l2 * weights
        if self.bounded:
            if 2 * activity_tolerance >= 1:
                raise ValueError("activity_tolerance makes the weight bounds overlap")
            gradient = smooth_gradient + self.lambda_l1
            at_lower = weights <= activity_tolerance
            at_upper = weights >= 1 - activity_tolerance
            weight_violations = torch.where(
                at_lower,
                torch.relu(-gradient),
                torch.where(at_upper, torch.relu(gradient), gradient.abs()),
            )
        else:
            at_zero = weights.abs() <= activity_tolerance
            signed_residual = smooth_gradient + self.lambda_l1 * torch.sign(weights)
            weight_violations = torch.where(
                at_zero,
                torch.relu(smooth_gradient.abs() - self.lambda_l1),
                signed_residual.abs(),
            )
        return torch.maximum(weight_violations.max(), intercept_gradient.abs())

    def dual_candidate(
        self,
        weights: torch.Tensor,
        intercept: torch.Tensor | float,
    ) -> torch.Tensor:
        """Construct a dual-feasible point for unbounded elastic net."""
        self._require_unbounded()
        self._validate_weights(weights)
        residual = self.X @ weights + intercept - self.y
        return (residual - residual.mean()) / self.spec.n

    def dual_equality_violation(self, dual: torch.Tensor) -> torch.Tensor:
        """Return the absolute violation of the intercept's dual equality."""
        self._require_unbounded()
        self._validate_dual(dual)
        return dual.sum().abs()

    def dual_objective(self, dual: torch.Tensor) -> torch.Tensor:
        """Evaluate the unbounded elastic-net dual at a feasible point."""
        self._require_unbounded()
        self._validate_dual(dual)
        conjugate_argument = -(self.X.mT @ dual)
        soft_thresholded = torch.sign(conjugate_argument) * torch.relu(
            conjugate_argument.abs() - self.lambda_l1
        )
        return (
            -(self.y @ dual)
            - 0.5 * self.spec.n * dual.square().sum()
            - 0.5 * soft_thresholded.square().sum() / self.lambda_l2
        )

    def relative_duality_gap(
        self,
        weights: torch.Tensor,
        intercept: torch.Tensor | float,
    ) -> torch.Tensor:
        """Return the gap from the uniformly constructed dual certificate."""
        self._require_unbounded()
        primal = self.objective(weights, intercept)
        dual = self.dual_objective(self.dual_candidate(weights, intercept))
        scale = torch.maximum(
            torch.ones((), dtype=primal.dtype, device=primal.device),
            torch.maximum(primal.abs(), dual.abs()),
        )
        return (primal - dual) / scale

    def diagnostics(self) -> dict[str, float | int | bool]:
        return {
            "lambda_max": self.lambda_max,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "teacher_nonzeros": int(torch.count_nonzero(self.teacher_weights)),
            "bounded": self.bounded,
            "feature_generator": self.spec.feature_generator,
            "feature_decay_exponent": self.spec.feature_decay_exponent,
            "feature_frobenius_norm": float(torch.linalg.vector_norm(self.X)),
        }

    def _validate_weights(self, weights: torch.Tensor) -> None:
        if weights.shape != (self.spec.p,):
            raise ValueError(f"weights must have shape {(self.spec.p,)}")

    def _validate_dual(self, dual: torch.Tensor) -> None:
        if dual.shape != (self.spec.n,):
            raise ValueError(f"dual must have shape {(self.spec.n,)}")

    def _require_unbounded(self) -> None:
        if self.bounded:
            raise ValueError("the vanilla elastic-net dual is not valid for bounded elastic net")


def generate_elastic_net_problem(
    spec: ElasticNetSpec,
    *,
    bounded: bool,
    device: torch.device | str,
) -> ElasticNetProblem:
    """Generate one standardized regression dataset for both EN variants.

    The response is scaled but not centered after adding noise, preserving the
    explicit unregularized intercept. Both elastic-net weights equal
    regularization_fraction times lambda_max, where lambda_max is computed after
    centering for the fitted intercept.
    """
    features = _features(
        spec.n,
        spec.p,
        spec.feature_seed,
        spec.feature_generator,
        spec.feature_decay_exponent,
    )
    target_rng = _cpu_generator(spec.target_seed)
    support_size = max(1, math.ceil(spec.teacher_density * spec.p))
    support = torch.randperm(spec.p, generator=target_rng)[:support_size]
    raw_weights = torch.zeros(spec.p, dtype=torch.float64)
    raw_weights[support] = 0.25 + 0.5 * torch.rand(
        support_size,
        dtype=torch.float64,
        generator=target_rng,
    )
    signal = features @ raw_weights
    signal_scale = signal.square().mean().sqrt()
    noise = torch.randn(spec.n, dtype=torch.float64, generator=target_rng)
    noise -= noise.mean()
    noise /= noise.square().mean().sqrt()
    raw_response = signal + spec.teacher_intercept + spec.noise_ratio * signal_scale * noise
    response_mean = raw_response.mean()
    centered_response = raw_response - response_mean
    response_scale = centered_response.square().mean().sqrt()
    if float(response_scale) == 0:
        raise RuntimeError("generated a constant response")
    response = raw_response / response_scale
    teacher_weights = raw_weights / response_scale
    teacher_intercept = (
        torch.as_tensor(spec.teacher_intercept, dtype=torch.float64) / response_scale
    )

    centered_features = features - features.mean(dim=0)
    centered_response = response - response.mean()
    lambda_max = float(
        torch.linalg.vector_norm(
            centered_features.mT @ centered_response,
            ord=float("inf"),
        )
        / spec.n
    )
    regularization = spec.regularization_fraction * lambda_max
    target = torch.device(device)
    return ElasticNetProblem(
        spec=spec,
        X=features.to(target),
        y=response.to(target),
        teacher_weights=teacher_weights.to(target),
        teacher_intercept=teacher_intercept.to(target),
        lambda_max=lambda_max,
        lambda_l1=regularization,
        lambda_l2=regularization,
        bounded=bounded,
    )

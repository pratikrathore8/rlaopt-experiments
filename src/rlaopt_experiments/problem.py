"""Deterministic prescribed-spectrum problems with SORF singular vectors."""

from __future__ import annotations

from dataclasses import dataclass

import torch


GENERATOR_VERSION = "sorf1"


@dataclass(frozen=True)
class ProblemSpec:
    n: int
    p: int
    alpha: float
    factor_seed: int
    response_seed: int

    @property
    def rank(self) -> int:
        return min(self.n, self.p)

    @property
    def problem_id(self) -> str:
        return (
            f"{GENERATOR_VERSION}-n{self.n}-p{self.p}-a{self.alpha:g}"
            f"-f{self.factor_seed}-y{self.response_seed}"
        )


@dataclass
class RidgeProblem:
    spec: ProblemSpec
    X: torch.Tensor
    y: torch.Tensor
    singular_values: torch.Tensor
    response_coordinates: torch.Tensor
    right_signs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def rhs(self) -> torch.Tensor:
        return self.X.mT @ self.y

    def oracle(self, ridge: float) -> torch.Tensor:
        weights = self.singular_values / (self.singular_values.square() + ridge)
        embedded = torch.zeros(self.spec.p, dtype=self.X.dtype, device=self.X.device)
        embedded[:self.spec.rank] = weights * self.response_coordinates
        _sorf_(embedded, 0, self.right_signs)
        return embedded

    def diagnostics(self, ridge: float) -> dict[str, float]:
        eig = self.singular_values.square()
        d_eff = float((eig / (eig + ridge)).sum())
        smallest_eigenvalue = eig[-1] if self.spec.rank == self.spec.p else 0.0
        full_condition = float((eig[0] + ridge) / (smallest_eigenvalue + ridge))
        return {
            "effective_dimension": d_eff,
            "full_condition_number": full_condition,
            "rank_over_p": self.spec.rank / self.spec.p,
            "rank_over_effective_dimension": self.spec.rank / d_eff,
        }


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _fwht_(tensor: torch.Tensor, dimension: int) -> None:
    """Apply the normalized Walsh-Hadamard transform in place."""
    size = tensor.shape[dimension]
    if not _is_power_of_two(size):
        raise ValueError("Hadamard dimensions must be positive powers of two")
    width = 1
    while width < size:
        if tensor.ndim == 1:
            blocks = tensor.view(size // (2 * width), 2 * width)
            left = blocks[:, :width]
            right = blocks[:, width:]
        elif dimension == 0:
            rows, columns = tensor.shape
            blocks = tensor.view(rows // (2 * width), 2 * width, columns)
            left = blocks[:, :width, :]
            right = blocks[:, width:, :]
        elif dimension == 1:
            rows, columns = tensor.shape
            blocks = tensor.view(rows, columns // (2 * width), 2 * width)
            left = blocks[:, :, :width]
            right = blocks[:, :, width:]
        else:
            raise ValueError("FWHT supports vectors and matrix dimensions 0 or 1")
        saved_left = left.clone()
        left.add_(right)
        right.neg_().add_(saved_left)
        width *= 2
    tensor.div_(size**0.5)


def _random_signs(size: int, generator: torch.Generator) -> torch.Tensor:
    values = torch.randint(0, 2, (size,), dtype=torch.int8, generator=generator)
    return values.mul_(2).sub_(1).to(torch.float64)


def _sorf_(tensor: torch.Tensor, dimension: int,
           signs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    """Apply Q = H D1 H D2 H D3 along one tensor dimension."""
    broadcast = [1] * tensor.ndim
    broadcast[dimension] = tensor.shape[dimension]
    for diagonal in reversed(signs):
        tensor.mul_(diagonal.view(broadcast))
        _fwht_(tensor, dimension)


def generate_problem(spec: ProblemSpec) -> RidgeProblem:
    """Materialize X with independent SORF left and right singular vectors."""
    if not _is_power_of_two(spec.n) or not _is_power_of_two(spec.p):
        raise ValueError("n and p must be positive powers of two")
    dtype = torch.float64
    rank = spec.rank
    factor_rng = torch.Generator(device="cpu").manual_seed(spec.factor_seed)
    left_signs = tuple(_random_signs(spec.n, factor_rng) for _ in range(3))
    right_signs = tuple(_random_signs(spec.p, factor_rng) for _ in range(3))

    indices = torch.arange(1, rank + 1, dtype=dtype)
    singular_values = indices.pow(-spec.alpha / 2)
    x = torch.zeros((spec.n, spec.p), dtype=dtype)
    diagonal = torch.arange(rank)
    x[diagonal, diagonal] = singular_values
    _sorf_(x, 0, left_signs)
    _sorf_(x, 1, right_signs)

    response_rng = torch.Generator(device="cpu").manual_seed(spec.response_seed)
    coordinates = torch.randn(rank, dtype=dtype, generator=response_rng)
    coordinates /= torch.linalg.vector_norm(coordinates)
    y = torch.zeros(spec.n, dtype=dtype)
    y[:rank] = coordinates
    _sorf_(y, 0, left_signs)
    return RidgeProblem(spec, x, y, singular_values, coordinates, right_signs)


def relative_kkt(problem: RidgeProblem, estimate: torch.Tensor, ridge: float) -> float:
    rhs = problem.rhs()
    residual = problem.X.mT @ (problem.X @ estimate) + ridge * estimate - rhs
    return float(torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(rhs))


def relative_solution_error(problem: RidgeProblem, estimate: torch.Tensor, ridge: float) -> float:
    oracle = problem.oracle(ridge)
    return float(torch.linalg.vector_norm(estimate - oracle) / torch.linalg.vector_norm(oracle))

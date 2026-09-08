"""Deterministic prescribed-spectrum problems with SORF singular vectors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rlaopt_experiments.structured_orthogonal import (
    is_power_of_two,
    materialize_sorf_matrix,
    sorf_,
)


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
        embedded[: self.spec.rank] = weights * self.response_coordinates
        sorf_(embedded, 0, self.right_signs)
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


def generate_problem(spec: ProblemSpec, device: torch.device | str = "cpu") -> RidgeProblem:
    """Materialize X on ``device`` with independent SORF singular vectors.

    Random inputs are sampled from CPU generators so a seed identifies the same
    mathematical problem on every backend.  The dense SORF transforms themselves
    run natively on the requested device.
    """
    if not is_power_of_two(spec.n) or not is_power_of_two(spec.p):
        raise ValueError("n and p must be positive powers of two")
    dtype = torch.float64
    target = torch.device(device)
    rank = spec.rank
    indices = torch.arange(1, rank + 1, dtype=dtype)
    singular_values = indices.pow(-spec.alpha / 2).to(target)
    x, left_signs, right_signs = materialize_sorf_matrix(
        spec.n,
        spec.p,
        singular_values,
        seed=spec.factor_seed,
        device=target,
    )

    response_rng = torch.Generator(device="cpu").manual_seed(spec.response_seed)
    coordinates = torch.randn(rank, dtype=dtype, generator=response_rng)
    coordinates /= torch.linalg.vector_norm(coordinates)
    coordinates = coordinates.to(target)
    y = torch.zeros(spec.n, dtype=dtype, device=target)
    y[:rank] = coordinates
    sorf_(y, 0, left_signs)
    return RidgeProblem(spec, x, y, singular_values, coordinates, right_signs)


def relative_kkt(problem: RidgeProblem, estimate: torch.Tensor, ridge: float) -> float:
    rhs = problem.rhs()
    residual = problem.X.mT @ (problem.X @ estimate) + ridge * estimate - rhs
    return float(torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(rhs))


def relative_solution_error(problem: RidgeProblem, estimate: torch.Tensor, ridge: float) -> float:
    oracle = problem.oracle(ridge)
    return float(torch.linalg.vector_norm(estimate - oracle) / torch.linalg.vector_norm(oracle))

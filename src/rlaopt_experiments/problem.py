"""Deterministic prescribed-spectrum synthetic ridge problems."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
        return f"n{self.n}-p{self.p}-a{self.alpha:g}-f{self.factor_seed}-y{self.response_seed}"


@dataclass
class RidgeProblem:
    spec: ProblemSpec
    X: torch.Tensor
    y: torch.Tensor
    U: torch.Tensor
    singular_values: torch.Tensor
    V: torch.Tensor
    response_coordinates: torch.Tensor

    def rhs(self) -> torch.Tensor:
        return self.X.mT @ self.y

    def oracle(self, ridge: float) -> torch.Tensor:
        weights = self.singular_values / (self.singular_values.square() + ridge)
        return self.V @ (weights * self.response_coordinates)

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


def _generator(seed: int, device: torch.device) -> torch.Generator:
    # Generate on CPU for identical CPU/GPU problem instances; move only after generation.
    if device.type != "cpu":
        raise ValueError("problem factors must be generated on CPU before device transfer")
    return torch.Generator(device="cpu").manual_seed(seed)


def _canonical_qr(matrix: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    return q * signs


def generate_problem(spec: ProblemSpec) -> RidgeProblem:
    """Generate X=U diag(k^(-alpha/2)) V^T and a unit response in range(X)."""
    dtype = torch.float64
    cpu = torch.device("cpu")
    r = spec.rank
    factor_rng = _generator(spec.factor_seed, cpu)
    u = _canonical_qr(torch.randn(spec.n, r, dtype=dtype, generator=factor_rng))
    v = _canonical_qr(torch.randn(spec.p, r, dtype=dtype, generator=factor_rng))
    indices = torch.arange(1, r + 1, dtype=dtype)
    singular_values = indices.pow(-spec.alpha / 2)
    x = (u * singular_values) @ v.mT

    response_rng = _generator(spec.response_seed, cpu)
    coordinates = torch.randn(r, dtype=dtype, generator=response_rng)
    coordinates /= torch.linalg.vector_norm(coordinates)
    y = u @ coordinates
    return RidgeProblem(spec, x, y, u, singular_values, v, coordinates)


def relative_kkt(problem: RidgeProblem, estimate: torch.Tensor, ridge: float) -> float:
    rhs = problem.rhs()
    residual = problem.X.mT @ (problem.X @ estimate) + ridge * estimate - rhs
    return float(torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(rhs))


def relative_solution_error(problem: RidgeProblem, estimate: torch.Tensor, ridge: float) -> float:
    oracle = problem.oracle(ridge)
    return float(torch.linalg.vector_norm(estimate - oracle) / torch.linalg.vector_norm(oracle))

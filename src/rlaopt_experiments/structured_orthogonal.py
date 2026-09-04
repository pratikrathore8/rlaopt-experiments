"""Structured orthogonal transforms shared by synthetic problem generators."""

from __future__ import annotations

import torch


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def fwht_(tensor: torch.Tensor, dimension: int) -> None:
    """Apply the normalized Walsh--Hadamard transform in place."""
    size = tensor.shape[dimension]
    if not is_power_of_two(size):
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


def random_signs(size: int, generator: torch.Generator) -> torch.Tensor:
    """Draw a float64 Rademacher vector from ``generator``."""
    values = torch.randint(0, 2, (size,), dtype=torch.int8, generator=generator)
    return values.mul_(2).sub_(1).to(torch.float64)


def sorf_(
    tensor: torch.Tensor,
    dimension: int,
    signs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Apply ``H D1 H D2 H D3`` along one tensor dimension."""
    broadcast = [1] * tensor.ndim
    broadcast[dimension] = tensor.shape[dimension]
    for diagonal in reversed(signs):
        tensor.mul_(diagonal.view(broadcast))
        fwht_(tensor, dimension)


def materialize_sorf_matrix(
    n: int,
    p: int,
    singular_values: torch.Tensor,
    *,
    seed: int,
    device: torch.device | str,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Materialize a dense matrix with SORF singular vectors."""
    if not is_power_of_two(n) or not is_power_of_two(p):
        raise ValueError("SORF dimensions must be positive powers of two")
    rank = min(n, p)
    if singular_values.shape != (rank,):
        raise ValueError(f"singular_values must have shape {(rank,)}")

    target = torch.device(device)
    factor_rng = torch.Generator(device="cpu").manual_seed(seed)
    left_signs = tuple(random_signs(n, factor_rng).to(target) for _ in range(3))
    right_signs = tuple(random_signs(p, factor_rng).to(target) for _ in range(3))
    matrix = torch.zeros((n, p), dtype=singular_values.dtype, device=target)
    diagonal = torch.arange(rank, device=target)
    matrix[diagonal, diagonal] = singular_values.to(target)
    sorf_(matrix, 0, left_signs)
    sorf_(matrix, 1, right_signs)
    return matrix, left_signs, right_signs

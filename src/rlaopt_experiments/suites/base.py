"""Interfaces shared by independently implemented benchmark suites."""

from __future__ import annotations

from typing import Any, Protocol

import torch


class BenchmarkSuite(Protocol):
    """Boundary between generic orchestration and problem-specific code."""

    name: str

    def generate(self, specification: dict[str, Any], device: torch.device) -> Any:
        """Generate and retain one deterministic problem instance."""

    def problem_metadata(self, problem: Any) -> dict[str, Any]:
        """Describe the generated instance and its in-memory representation."""

    def execute(
        self,
        problem: Any,
        command: dict[str, Any],
        backend: str,
    ) -> dict[str, Any]:
        """Run one solver command and return suite-neutral result fields."""

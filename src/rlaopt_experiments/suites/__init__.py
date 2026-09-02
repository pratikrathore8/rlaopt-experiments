"""Problem-suite registry for benchmark-specific behavior."""

from __future__ import annotations

from functools import lru_cache

from rlaopt_experiments.suites.base import BenchmarkSuite


@lru_cache(maxsize=None)
def get_suite(name: str) -> BenchmarkSuite:
    """Return the registered benchmark suite named by a configuration file."""
    if name == "synthetic_ridge":
        from rlaopt_experiments.suites.ridge import RidgeSuite

        return RidgeSuite()
    if name == "synthetic_erm":
        from rlaopt_experiments.suites.synthetic_erm.suite import SyntheticErmSuite

        return SyntheticErmSuite()
    raise ValueError(f"unknown benchmark suite: {name}")


__all__ = ["BenchmarkSuite", "get_suite"]

"""Measure SORF problem generation independently of solver runtime."""

from __future__ import annotations

import json
import os
import resource
import time

import torch

from rlaopt_experiments.problem import ProblemSpec, generate_problem, relative_kkt


def main() -> None:
    threads = int(os.environ.get("BENCHMARK_CPU_THREADS", "64"))
    torch.set_num_threads(threads)
    spec = ProblemSpec(n=2**14, p=2**14, alpha=1.0, factor_seed=11, response_seed=12)
    started = time.perf_counter()
    problem = generate_problem(spec)
    generation_seconds = time.perf_counter() - started
    oracle_kkt = relative_kkt(problem, problem.oracle(1e-4), 1e-4)
    print(
        json.dumps(
            {
                "generation_seconds": generation_seconds,
                "matrix_bytes": problem.X.numel() * problem.X.element_size(),
                "n": spec.n,
                "p": spec.p,
                "oracle_kkt": oracle_kkt,
                "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "threads": torch.get_num_threads(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

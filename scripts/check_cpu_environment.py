"""Time CPU problem generation and spawned-worker startup on a benchmark node."""

from __future__ import annotations

import os
import time

import torch

from rlaopt_experiments.isolation import ProblemWorker
from rlaopt_experiments.problem import ProblemSpec, generate_problem


def main() -> None:
    threads = int(os.environ.get("BENCHMARK_CPU_THREADS", "64"))
    print(f"torch_imported={torch.__version__}", flush=True)
    torch.set_default_dtype(torch.float64)
    started = time.perf_counter()
    torch.set_num_threads(threads)
    print(f"set_num_threads_seconds={time.perf_counter() - started:.6f}", flush=True)
    print(f"threads={torch.get_num_threads()}", flush=True)
    print(f"interop_threads={torch.get_num_interop_threads()}", flush=True)
    print(f"affinity_cpus={len(os.sched_getaffinity(0))}", flush=True)

    specification = {
        "n": 256,
        "p": 256,
        "alpha": 1.0,
        "factor_seed": 1,
        "response_seed": 2,
    }
    started = time.perf_counter()
    generate_problem(ProblemSpec(**specification))
    print(f"direct_generation_seconds={time.perf_counter() - started:.6f}", flush=True)

    started = time.perf_counter()
    worker = ProblemWorker(specification, "cpu")
    try:
        ready = worker.wait_until_ready(timeout_seconds=120)
        print(f"worker_ready_seconds={time.perf_counter() - started:.6f}", flush=True)
        print(f"worker_status={ready['kind']}", flush=True)
    finally:
        worker.close()


if __name__ == "__main__":
    main()

"""Measure one large float64 PyTorch QR factorization."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=2**16)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()

    device = torch.device(args.backend)
    torch.set_default_dtype(torch.float64)
    if args.backend == "cpu":
        torch.set_num_threads(int(os.environ.get("BENCHMARK_CPU_THREADS", "64")))

    allocation_started = time.perf_counter()
    matrix = torch.randn((args.dimension, args.dimension), dtype=torch.float64, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    allocation_seconds = time.perf_counter() - allocation_started

    started = time.perf_counter()
    q, r = torch.linalg.qr(matrix, mode="reduced")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    qr_seconds = time.perf_counter() - started

    # Touch the results so the factorization cannot be discarded.
    checksum = float(q[0, 0] + r[0, 0])
    peak_cuda = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(
        json.dumps(
            {
                "backend": args.backend,
                "dimension": args.dimension,
                "dtype": "float64",
                "allocation_seconds": allocation_seconds,
                "qr_seconds": qr_seconds,
                "checksum": checksum,
                "peak_cuda_allocator_bytes": peak_cuda,
                "peak_process_rss_bytes": peak_rss,
                "torch_version": torch.__version__,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

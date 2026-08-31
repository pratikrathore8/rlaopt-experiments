"""Compare native CPU and CUDA SORF generation for the same seeded problems."""

from __future__ import annotations

import json
import time

import torch

from rlaopt_experiments.problem import ProblemSpec, generate_problem, relative_kkt


def comparison(cpu: torch.Tensor, cuda: torch.Tensor) -> dict[str, float | bool]:
    cuda_cpu = cuda.cpu()
    difference = torch.abs(cpu - cuda_cpu)
    denominator = torch.maximum(torch.abs(cpu), torch.abs(cuda_cpu))
    relative = torch.where(denominator > 0, difference / denominator, 0.0)
    return {
        "equal": torch.equal(cpu, cuda_cpu),
        "max_absolute_difference": float(difference.max()),
        "max_relative_difference": float(relative.max()),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe")
    records = []
    for exponent in (10, 12):
        spec = ProblemSpec(
            n=2**exponent,
            p=2**exponent,
            alpha=1.0,
            factor_seed=11,
            response_seed=12,
        )
        cpu_started = time.perf_counter()
        cpu = generate_problem(spec, device="cpu")
        cpu_seconds = time.perf_counter() - cpu_started

        torch.cuda.synchronize()
        cuda_started = time.perf_counter()
        cuda = generate_problem(spec, device="cuda")
        torch.cuda.synchronize()
        cuda_seconds = time.perf_counter() - cuda_started

        record = {
                "n": spec.n,
                "p": spec.p,
                "cpu_generation_seconds": cpu_seconds,
                "cuda_generation_seconds": cuda_seconds,
                "matrix": comparison(cpu.X, cuda.X),
                "response": comparison(cpu.y, cuda.y),
                "oracle": comparison(cpu.oracle(1e-4), cuda.oracle(1e-4)),
                "cpu_oracle_kkt": relative_kkt(cpu, cpu.oracle(1e-4), 1e-4),
                "cuda_oracle_kkt": relative_kkt(cuda, cuda.oracle(1e-4), 1e-4),
            }
        for key in ("matrix", "response", "oracle"):
            if record[key]["max_absolute_difference"] > 1e-12:
                raise RuntimeError(f"CPU and CUDA {key} disagree for n={spec.n}")
        if record["cpu_oracle_kkt"] > 1e-10 or record["cuda_oracle_kkt"] > 1e-10:
            raise RuntimeError(f"oracle failed the KKT check for n={spec.n}")
        records.append(record)
    print(json.dumps(records, indent=2, sort_keys=True), flush=True)
    print("PROBLEM_BACKEND_EQUIVALENCE=true", flush=True)


if __name__ == "__main__":
    main()

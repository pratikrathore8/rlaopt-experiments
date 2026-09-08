"""Persistent problem worker with a hard process boundary around each solve."""

from __future__ import annotations

import multiprocessing as mp
import os
import platform
import resource
import time
import traceback
from multiprocessing.connection import Connection
from typing import Any


def _worker(
    connection: Connection, suite_name: str, specification: dict[str, Any], backend: str
) -> None:
    """Generate one problem, retain it, and execute commands from the parent."""
    try:
        runtime_name = specification.get("pre_torch_runtime")
        if runtime_name not in {None, "clarabel"}:
            raise ValueError(f"unknown pre-PyTorch runtime: {runtime_name}")
        clarabel_runtime = None
        if runtime_name == "clarabel":
            from rlaopt_experiments.clarabel_bridge import initialize_clarabel_runtime

            clarabel_runtime = initialize_clarabel_runtime(backend)

        import torch

        from rlaopt_experiments.suites import get_suite

        torch.set_default_dtype(torch.float64)
        if os.environ.get("OMP_NUM_THREADS"):
            torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        device = torch.device("cuda" if backend == "cuda" else "cpu")
        suite = get_suite(suite_name)
        preparation_started = time.perf_counter()
        problem = suite.generate(specification, device=device)
        if backend == "cuda":
            torch.cuda.synchronize()
        problem_preparation_seconds = time.perf_counter() - preparation_started
        connection.send(
            {
                "kind": "ready",
                "worker_metadata": {
                    "torch_version": torch.__version__,
                    "pre_torch_runtime": runtime_name,
                    "problem_preparation_seconds": problem_preparation_seconds,
                    **suite.problem_metadata(problem),
                    "torch_num_threads": torch.get_num_threads(),
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version(),
                    "device_name": (
                        torch.cuda.get_device_name(device)
                        if backend == "cuda"
                        else platform.processor()
                    ),
                    "device_capability": (
                        list(torch.cuda.get_device_capability(device))
                        if backend == "cuda"
                        else None
                    ),
                },
            }
        )

        while True:
            command = connection.recv()
            if command["kind"] == "stop":
                return
            if backend == "cuda":
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                if clarabel_runtime is not None:
                    command = command | {"clarabel_runtime": clarabel_runtime}
                result = suite.execute(problem, command, backend)
                connection.send(
                    {
                        "kind": "result",
                        **result,
                        "peak_memory_bytes": (
                            torch.cuda.max_memory_allocated()
                            if backend == "cuda"
                            else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                        ),
                    }
                )
            except BaseException as error:  # Preserve native failures as data.
                connection.send(
                    {
                        "kind": "error",
                        "runtime_seconds": time.perf_counter() - started,
                        "native_status": "exception",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
    except BaseException as error:
        try:
            connection.send(
                {
                    "kind": "startup_error",
                    "native_status": "startup_exception",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        except BaseException:
            pass
    finally:
        connection.close()


class ProblemWorker:
    """Own a generated problem and enforce a timeout on every solve command."""

    def __init__(
        self,
        specification: dict[str, Any],
        backend: str,
        suite: str,
    ):
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(target=_worker, args=(child, suite, specification, backend))
        self._process.start()
        child.close()

    def wait_until_ready(self, timeout_seconds: float | None = None) -> dict[str, Any]:
        if not self._connection.poll(timeout_seconds):
            return {"kind": "startup_timeout", "native_status": "startup_timeout"}
        return self._connection.recv()

    def solve(self, command: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
        self._connection.send(command | {"kind": "solve"})
        if self._connection.poll(timeout_seconds):
            return self._connection.recv()
        self.terminate()
        return {
            "kind": "timeout",
            "runtime_seconds": timeout_seconds,
            "native_status": "timeout",
        }

    @property
    def alive(self) -> bool:
        return self._process.is_alive()

    def close(self) -> None:
        if self.alive:
            self._connection.send({"kind": "stop"})
            self._process.join(timeout=10)
        if self.alive:
            self.terminate()
        self._connection.close()

    def terminate(self) -> None:
        if self.alive:
            self._process.terminate()
            self._process.join(timeout=10)

    def __enter__(self) -> ProblemWorker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

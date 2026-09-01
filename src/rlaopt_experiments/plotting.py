"""Aggregate successful trials and produce paper-oriented figures."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from rlaopt_experiments.config import load_experiment, load_solvers


def _records(path: Path) -> list[dict]:
    return [json.loads(file.read_text()) for file in sorted(path.glob("*.json"))]


def _native_success(row: dict) -> bool:
    if row["metadata"].get("worker_outcome") != "result":
        return False
    status = row["native_status"]
    return status in {"native_converged", "native_complete", "direct", "istop_1", "istop_2"}


def _record_key(row: dict) -> tuple:
    return (
        row["backend"],
        row["solver"],
        row["n"],
        row["p"],
        row["alpha"],
        row["ridge"],
        row["seed"],
        row["repetition"],
    )


def _aggregate(records: list[dict], family: str) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        belongs = {
            "square": row["n"] == row["p"],
            "fixed_p": row["p"] == 16384 and row["n"] >= row["p"],
            "fixed_n": row["n"] == 65536 and row["p"] <= row["n"],
        }
        if belongs[family]:
            groups[
                (row["solver"], row["backend"], row["n"], row["p"], row["alpha"], row["ridge"])
            ].append(row)
    result = []
    for key, values in groups.items():
        valid = [item["runtime_seconds"] for item in values if _native_success(item)]
        result.append(
            dict(zip(("solver", "backend", "n", "p", "alpha", "ridge"), key))
            | {
                "runtime": statistics.median(valid) if valid else None,
                "runtime_min": min(valid) if valid else None,
                "runtime_max": max(valid) if valid else None,
                "native_success_rate": sum(_native_success(item) for item in values) / len(values),
                "external_kkt_success_rate": sum(item["success"] for item in values) / len(values),
            }
        )
    return result


def make_figures(input_dir: Path, output_dir: Path, config_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _records(input_dir)
    if not records:
        raise RuntimeError(f"no JSON records found in {input_dir}")
    for family, scale, label in (
        ("fixed_p", "n", "samples n"),
        ("fixed_n", "p", "features p"),
        ("square", "n", "square dimension n=p"),
    ):
        rows = _aggregate(records, family)
        if not rows:
            continue
        alphas = sorted({row["alpha"] for row in rows})
        ridges = sorted({row["ridge"] for row in rows}, reverse=True)
        figure, axes = plt.subplots(
            len(alphas),
            len(ridges),
            figsize=(4 * len(ridges), 3.2 * len(alphas)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        legend_handles: dict[str, object] = {}
        for alpha_index, alpha in enumerate(alphas):
            for ridge_index, ridge in enumerate(ridges):
                axis = axes[alpha_index][ridge_index]
                series: dict[tuple, list[dict]] = defaultdict(list)
                for row in rows:
                    if (
                        row["runtime"] is not None
                        and row["alpha"] == alpha
                        and row["ridge"] == ridge
                    ):
                        series[(row["solver"], row["backend"])].append(row)
                for (solver, backend), values in sorted(series.items()):
                    values.sort(key=lambda row: row[scale])
                    series_label = f"{solver} ({backend})"
                    handle = axis.scatter(
                        [row[scale] for row in values],
                        [row["runtime"] for row in values],
                        label=series_label,
                        alpha=0.75,
                    )
                    legend_handles.setdefault(series_label, handle)
                    color = handle.get_facecolor()[0]
                    axis.vlines(
                        [row[scale] for row in values],
                        [row["runtime_min"] for row in values],
                        [row["runtime_max"] for row in values],
                        color=color,
                        alpha=0.35,
                    )
                axis.set(
                    xscale="log", yscale="log", title=rf"$\alpha={alpha:g}$, $\lambda={ridge:g}$"
                )
                axis.grid(True, which="both", alpha=0.2)
                if alpha_index == len(alphas) - 1:
                    axis.set_xlabel(label)
                if ridge_index == 0:
                    axis.set_ylabel("runtime (seconds)")
        if legend_handles:
            figure.legend(
                legend_handles.values(),
                legend_handles.keys(),
                loc="outside lower center",
                ncol=min(4, len(legend_handles)),
                fontsize=8,
            )
            figure.subplots_adjust(bottom=0.12)
        figure.tight_layout()
        figure.savefig(output_dir / f"runtime_{family}.pdf")
        figure.savefig(output_dir / f"runtime_{family}.png", dpi=200)
        plt.close(figure)

    config = load_experiment(config_path)
    expected = {
        (backend, solver, shape.n, shape.p, alpha, ridge, seed, repetition)
        for backend in ("cpu", "cuda")
        for solver in load_solvers(config_path, backend)
        for shape in config.shapes
        for alpha in config.alphas
        for ridge in config.lambdas
        for seed in config.seeds
        for repetition in range(config.repetitions)
    }
    missing = expected - {_record_key(row) for row in records}
    summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "native_success": 0,
            "external_kkt_pass": 0,
            "external_kkt_miss": 0,
            "timeout": 0,
            "error": 0,
            "missing": 0,
        }
    )
    for row in records:
        bucket = summary[f"{row['solver']}:{row['backend']}"]
        bucket["native_success"] += int(_native_success(row))
        bucket["external_kkt_pass"] += int(row["success"])
        bucket["external_kkt_miss"] += int(
            row["metadata"].get("worker_outcome") == "result" and not row["success"]
        )
        bucket["timeout"] += int(row["timed_out"])
        bucket["error"] += int(row["metadata"].get("worker_outcome") == "error")
    for backend, solver, *_ in missing:
        summary[f"{solver}:{backend}"]["missing"] += 1
    (output_dir / "failure_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    missing_rows = [
        dict(zip(("backend", "solver", "n", "p", "alpha", "ridge", "seed", "repetition"), key))
        for key in sorted(missing)
    ]
    (output_dir / "missing_records.json").write_text(
        json.dumps(missing_rows, indent=2, sort_keys=True) + "\n"
    )

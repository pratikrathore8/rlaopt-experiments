"""Aggregate successful trials and produce paper-oriented figures."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _records(path: Path) -> list[dict]:
    return [json.loads(file.read_text()) for file in sorted(path.glob("*.json"))]


def _aggregate(records: list[dict], family: str) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        belongs = {
            "square": row["n"] == row["p"],
            "fixed_p": row["p"] == 16384 and row["n"] >= row["p"],
            "fixed_n": row["n"] == 65536 and row["p"] <= row["n"],
        }
        if belongs[family]:
            groups[(row["solver"], row["backend"], row["n"], row["p"], row["alpha"],
                    row["ridge"])].append(row)
    result = []
    for key, values in groups.items():
        valid = [item["runtime_seconds"] for item in values if item["success"]]
        result.append(dict(zip(("solver", "backend", "n", "p", "alpha", "ridge"), key)) | {
            "runtime": statistics.median(valid) if valid else None,
            "success_rate": sum(item["success"] for item in values) / len(values),
        })
    return result


def make_figures(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _records(input_dir)
    if not records:
        raise RuntimeError(f"no JSON records found in {input_dir}")
    for family, scale, label in (("fixed_p", "n", "samples n"),
                                  ("fixed_n", "p", "features p"),
                                  ("square", "n", "square dimension n=p")):
        rows = _aggregate(records, family)
        if not rows:
            continue
        alphas = sorted({row["alpha"] for row in rows})
        ridges = sorted({row["ridge"] for row in rows}, reverse=True)
        figure, axes = plt.subplots(
            len(alphas), len(ridges), figsize=(4 * len(ridges), 3.2 * len(alphas)),
            sharex=True, sharey=True, squeeze=False,
        )
        legend_handles: dict[str, object] = {}
        for alpha_index, alpha in enumerate(alphas):
            for ridge_index, ridge in enumerate(ridges):
                axis = axes[alpha_index][ridge_index]
                series: dict[tuple, list[dict]] = defaultdict(list)
                for row in rows:
                    if row["runtime"] is not None and row["alpha"] == alpha and row["ridge"] == ridge:
                        series[(row["solver"], row["backend"])].append(row)
                for (solver, backend), values in sorted(series.items()):
                    values.sort(key=lambda row: row[scale])
                    series_label = f"{solver} ({backend})"
                    handle = axis.scatter(
                        [row[scale] for row in values], [row["runtime"] for row in values],
                        label=series_label, alpha=0.75,
                    )
                    legend_handles.setdefault(series_label, handle)
                axis.set(xscale="log", yscale="log",
                         title=rf"$\alpha={alpha:g}$, $\lambda={ridge:g}$")
                axis.grid(True, which="both", alpha=0.2)
                if alpha_index == len(alphas) - 1:
                    axis.set_xlabel(label)
                if ridge_index == 0:
                    axis.set_ylabel("runtime (seconds)")
        if legend_handles:
            figure.legend(legend_handles.values(), legend_handles.keys(), loc="outside lower center",
                          ncol=min(4, len(legend_handles)), fontsize=8)
            figure.subplots_adjust(bottom=0.12)
        figure.tight_layout()
        figure.savefig(output_dir / f"runtime_{family}.pdf")
        figure.savefig(output_dir / f"runtime_{family}.png", dpi=200)
        plt.close(figure)

    failures = output_dir / "failure_summary.json"
    summary: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0, "timeout": 0})
    for row in records:
        bucket = summary[f"{row['solver']}:{row['backend']}"]
        bucket["success" if row["success"] else "failure"] += 1
        bucket["timeout"] += int(row["timed_out"])
    failures.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

"""Render the production paper figures without importing any solver runtimes.

Run from the repository root with `.venv/bin/python scripts/make_paper_figures.py`.
All eligibility decisions and missing trials are exported alongside the figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics as stats
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from rlaopt_experiments.records import read_record
from paper_refinement import apply_real_refinement, apply_ridge_refinement


SOLVER_COLORS = {
    "rlaopt_nystrom_pcg": "#163e64",
    "rlaopt_cg": "#e69f00",
    "scipy_lsqr": "#009e73",
    "cuml_lsmr": "#56b4e9",
    "torch_qr": "#cc79a7",
    "rlaopt_sapphire": "#6b4595",
    "rlaopt_admm": "#a54520",
    "projected_gradient": "#0072b2",
    "projected_gradient_no_jit": "#0072b2",
    "jaxopt_lbfgsb": "#009e73",
    "jaxopt_lbfgsb_no_jit": "#009e73",
    "scs_cpu_indirect": "#b47a00",
    "scs_cuda_direct": "#e69f00",
    "scs": "#e69f00",
    "scs_cuda": "#b47a00",
    "clarabel_qdldl": "#56b4e9",
    "cuclarabel_cudss": "#56b4e9",
}
LABELS = {
    "rlaopt_nystrom_pcg": r"Nystr\"om PCG",
    "rlaopt_cg": "CG",
    "scipy_lsqr": "SciPy LSQR",
    "cuml_lsmr": "cuML LSMR",
    "torch_qr": "PyTorch QR",
    "rlaopt_sapphire": "SAPPHIRE",
    "rlaopt_admm": "NysADMM",
    "projected_gradient_no_jit": "JAXopt APG (JIT off)",
    "jaxopt_lbfgsb_no_jit": "JAXopt L-BFGS-B (JIT off)",
    "projected_gradient": "JAXopt APG (JIT on)",
    "jaxopt_lbfgsb": "JAXopt L-BFGS-B (JIT on)",
    "scs_cpu_indirect": "SCS indirect",
    "scs_cuda_direct": "SCS direct",
    "scs": "SCS direct",
    "scs_cuda": "SCS indirect",
    "clarabel_qdldl": "Clarabel",
    "cuclarabel_cudss": "cuClarabel",
}
STATUS_COLORS = {
    "ACC": "#ead4c0",
    "OK": "#d5e9e1",
    "TO": "#f7e3ba",
    "NC": "#ecdbe9",
    "ITER": "#ecdbe9",
    "HOST": "#efc9c9",
    "GPU": "#efc9c9",
    "IDX": "#dfd1e8",
    "ERR": "#e1e1e1",
    "MISS": "#eeeeee",
    "WARMUP_OOM": "#efc9c9",
}
STATUS_LABELS = {
    "ACC": "Accuracy miss",
    "OK": "Success",
    "TO": "Timeout",
    "ITER": "Iteration limit",
    "NC": "Not converged",
    "HOST": "Host OOM",
    "GPU": "GPU OOM",
    "IDX": "Index overflow",
    "ERR": "Worker error",
    "MISS": "No record",
    "WARMUP_OOM": "Warmup OOM",
}


def tex(text):
    # Mathtext supplies Computer Modern and LaTeX without an external TeX install.
    text = text.replace(r"Nystr\"om", r"Nystr\ddot{o}m")
    return (
        r"$\mathrm{" + text.replace(" ", r"\ ").replace("_", r"\_").replace("-", r"\text{-}") + "}$"
    )


def label(solver):
    return tex(LABELS[solver])


def read_json(path):
    return json.loads(path.read_text())


def success(row):
    if row is None:
        return False
    if "plot_eligible" in row:
        return bool(row["plot_eligible"])
    if "runtime_eligible" in row:
        return bool(row["runtime_eligible"])
    return row["metadata"].get("worker_outcome") == "result" and row["native_status"] in {
        "native_converged",
        "native_complete",
        "direct",
        "istop_1",
        "istop_2",
    }


def runtime(row):
    value = row.get("runtime_seconds", row.get("timings", {}).get("runtime_seconds"))
    if success(row):
        assert value is not None and math.isfinite(value) and value > 0
    return value


def status(row):
    if row is None:
        return "MISS"
    if success(row):
        return "OK"
    if row.get("native_success") and row.get("external_success") is False:
        return "ACC"
    if row.get("timed_out") or row["native_status"] == "timeout":
        return "TO"
    error = row["metadata"].get("error_message", "")
    if "InexactError: trunc(Int32" in error:
        return "IDX"
    if "out of memory" in error.lower() or "OutOfMemory" in error:
        return "GPU" if "cuda" in error.lower() else "HOST"
    if row["native_status"] in {"not_converged", "max_iterations"}:
        if row.get("iterations", 0) >= row["metadata"].get("max_iterations", math.inf):
            return "ITER"
        return "NC"
    return "ERR"


def real_key(row):
    return (row["backend"], row["solver"], row["problem_id"], row["seed"], row.get("repetition", 0))


def ridge_key(row):
    return tuple(
        row[k] for k in ("backend", "solver", "n", "p", "alpha", "ridge", "seed", "repetition")
    )


def save(fig, output, name):
    for extension in ("pdf", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def grid(ax):
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def power_ticks(ax, values):
    ax.set_xscale("log", base=2)
    ax.set_xticks(values, [rf"$2^{{{int(math.log2(v))}}}$" for v in values])
    ax.minorticks_off()


def timeout_ticks(ax, seconds, lower):
    ticks = [
        10.0**e for e in range(math.ceil(math.log10(lower)), math.floor(math.log10(seconds)) + 1)
    ]
    ticks.append(seconds)
    labels = [rf"$10^{{{int(math.log10(t))}}}$" for t in ticks[:-1]]
    labels.append(tex(f"{seconds:,} ({'1 hr' if seconds == 3600 else '15 min'} limit)"))
    ax.set_yticks(ticks, labels)


def dataset_label(row):
    return tex(row["display"]) + "\n" + rf"${row['n']:,}\times {row['p']:,}$"


def ridge_figures(rows, output, main_only=False):
    solvers = {
        "cpu": ["rlaopt_nystrom_pcg", "rlaopt_cg", "scipy_lsqr", "torch_qr"],
        "cuda": ["rlaopt_nystrom_pcg", "rlaopt_cg", "cuml_lsmr", "torch_qr"],
    }
    alphas = sorted({r["alpha"] for r in rows})
    lambdas = sorted({r["ridge"] for r in rows})
    families = {
        "fixed_n": (lambda r: r["n"] == 65536, "p", r"$p\quad(n=2^{16})$"),
        "fixed_p": (lambda r: r["p"] == 16384 and r["n"] >= 16384, "n", r"$n\quad(p=2^{14})$"),
        "square": (lambda r: r["n"] == r["p"], "n", r"$n=p$"),
    }
    for family, (belongs, scale, xlabel) in families.items():
        for ridge in lambdas:
            if main_only and (family != "fixed_n" or ridge != min(lambdas)):
                continue
            selected = [r for r in rows if belongs(r) and r["ridge"] == ridge]
            sizes = sorted({r[scale] for r in selected})
            lower = min(runtime(r) for r in selected if success(r)) * 0.65
            fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), sharey=True, layout="constrained")
            for i, backend in enumerate(solvers):
                for j, alpha in enumerate(alphas):
                    ax = axes[i, j]
                    failure_line = 0
                    for si, solver in enumerate(solvers[backend]):
                        xs, ys, lo, hi = [], [], [], []
                        for size in sizes:
                            group = [
                                r
                                for r in selected
                                if r["backend"] == backend
                                and r["alpha"] == alpha
                                and r["solver"] == solver
                                and r[scale] == size
                            ]
                            valid = [runtime(r) for r in group if success(r)]
                            if valid:
                                median = stats.median(valid)
                                xs.append(size)
                                ys.append(median)
                                lo.append(median - min(valid))
                                hi.append(max(valid) - median)
                                if any(r.get("refined") for r in group):
                                    ax.scatter(
                                        size,
                                        median,
                                        marker="D",
                                        s=27,
                                        color=SOLVER_COLORS[solver],
                                        zorder=4,
                                    )
                                if len(valid) < len(group):
                                    ax.annotate(
                                        tex(f"{len(valid)}/{len(group)}"),
                                        (size, median),
                                        xytext=(3, 5),
                                        textcoords="offset points",
                                        fontsize=7,
                                    )
                            failed = Counter(r["plot_status"] for r in group if not success(r))
                            if failed:
                                # A separate axes-coordinate strip has no runtime meaning.
                                annotation = "; ".join(
                                    f"{STATUS_LABELS[s]} ({c}/{len(group)} seeds)"
                                    for s, c in sorted(failed.items())
                                )
                                ax.text(
                                    size,
                                    1.025 + failure_line * 0.065,
                                    tex(f"{LABELS[solver]}: {annotation}"),
                                    transform=ax.get_xaxis_transform(),
                                    ha="right",
                                    fontsize=7.5,
                                    color=SOLVER_COLORS[solver],
                                )
                                failure_line += 1
                        ax.errorbar(
                            xs,
                            ys,
                            yerr=[lo, hi],
                            fmt="o",
                            ms=4,
                            capsize=2,
                            color=SOLVER_COLORS[solver],
                            label=label(solver),
                        )
                    power_ticks(ax, sizes)
                    ax.set_yscale("log")
                    ax.set_ylim(lower, 1500)
                    ax.axhline(900, color=".6", ls=":", lw=0.8)
                    timeout_ticks(ax, 900, lower)
                    ax.set_title(
                        tex("CPU" if backend == "cpu" else "GPU") + rf"$: \alpha={alpha:g}$", pad=42
                    )
                    ax.set_xlabel(xlabel)
                    if j == 0:
                        ax.set_ylabel(tex("Solver Runtime (s)"))
                    grid(ax)
            handles = [
                Line2D([], [], color=SOLVER_COLORS[solver], marker="o", ls="", label=label(solver))
                for solver in [
                    "rlaopt_nystrom_pcg",
                    "rlaopt_cg",
                    "scipy_lsqr",
                    "cuml_lsmr",
                    "torch_qr",
                ]
            ]
            fig.legend(handles=handles, loc="outside lower center", ncols=5, fontsize=9)
            fig.suptitle(
                tex("Ridge regression") + rf"$: \lambda=10^{{{int(math.log10(ridge))}}}$",
                fontsize=14,
            )
            name = (
                "ridge_scaling"
                if family == "fixed_n" and ridge == min(lambdas)
                else f"appendix_ridge_{family}_lambda_1e{int(math.log10(ridge))}"
            )
            save(fig, output, name)

    if not main_only:
        preconditioning_figures(rows, output, alphas, lambdas)


def preconditioning_figures(rows, output, alphas, lambdas):
    lookup = {ridge_key(r): r for r in rows}
    shapes = sorted({(r["n"], r["p"]) for r in rows})
    cells, export = {}, []
    for n, p in shapes:
        for backend in ("cpu", "cuda"):
            for i, alpha in enumerate(alphas):
                for j, ridge in enumerate(lambdas):
                    group = [
                        r
                        for r in rows
                        if (r["backend"], r["solver"], r["n"], r["p"], r["alpha"], r["ridge"])
                        == (backend, "rlaopt_nystrom_pcg", n, p, alpha, ridge)
                    ]
                    ratios, censored = [], False
                    for r in group:
                        cg = lookup.get(ridge_key(r)[:1] + ("rlaopt_cg",) + ridge_key(r)[2:])
                        if success(r) and cg and (success(cg) or cg["plot_status"] == "TO"):
                            ratios.append(
                                (900 if cg["plot_status"] == "TO" else runtime(cg)) / runtime(r)
                            )
                            censored |= cg["plot_status"] == "TO"
                    value = (
                        stats.median(ratios) if len(ratios) == len(group) and ratios else math.nan
                    )
                    cells[n, p, backend, i, j] = value, censored
                    export.append(
                        {
                            "n": n,
                            "p": p,
                            "backend": backend,
                            "alpha": alpha,
                            "ridge": ridge,
                            "ratio": value,
                            "lower_bound": censored,
                            "matched_seeds": len(ratios),
                        }
                    )
    norm = Normalize(0, max(v for v, c in cells.values() if math.isfinite(v) and not c))

    def draw(ax, n, p, backend):
        values = np.array(
            [
                [
                    cells[n, p, backend, i, j][0] if not cells[n, p, backend, i, j][1] else np.nan
                    for j in range(3)
                ]
                for i in range(3)
            ]
        )
        im = ax.imshow(values, cmap="Blues", norm=norm)
        for i in range(3):
            for j in range(3):
                v, censored = cells[n, p, backend, i, j]
                shown = math.floor(v * 100) / 100 if censored and math.isfinite(v) else v
                prefix = r"\geq " if censored else ""
                annotation = (
                    rf"${prefix}{shown:.2f}\times$" if math.isfinite(v) else tex("Unavailable")
                )
                ax.text(
                    j,
                    i,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if not censored and v > 0.65 * norm.vmax else "black",
                )
                if censored:
                    ax.add_patch(
                        Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", edgecolor=".8", lw=0
                        )
                    )
        ax.set_xticks(range(3), [rf"$10^{{{int(math.log10(v))}}}$" for v in lambdas])
        ax.set_yticks(range(3), [rf"${a:g}$" for a in alphas])
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\alpha$")
        ax.set_title(
            tex("CPU" if backend == "cpu" else "GPU")
            + rf"$:\ n=2^{{{int(math.log2(n))}}},\ p=2^{{{int(math.log2(p))}}}$",
            fontsize=10,
        )
        return im

    fig, axes = plt.subplots(
        math.ceil(len(shapes) / 2),
        4,
        figsize=(13, 3 * math.ceil(len(shapes) / 2)),
        layout="constrained",
        squeeze=False,
    )
    for index, (n, p) in enumerate(shapes):
        for b, backend in enumerate(("cpu", "cuda")):
            im = draw(axes[index // 2, 2 * (index % 2) + b], n, p, backend)
        single, pair_axes = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained")
        for ax, backend in zip(pair_axes, ("cpu", "cuda")):
            subimage = draw(ax, n, p, backend)
        single.colorbar(
            subimage, ax=pair_axes, label=tex("Speedup") + r"$\ (T_{\mathrm{CG}}/T_{\mathrm{PCG}})$"
        )
        save(single, output, f"appendix_ridge_preconditioning_n{n}_p{p}")
    if len(shapes) % 2:
        for ax in axes[-1, 2:]:
            ax.set_visible(False)
    fig.colorbar(
        im, ax=axes, label=tex("Speedup") + r"$\ (T_{\mathrm{CG}}/T_{\mathrm{PCG}})$", shrink=0.6
    )
    fig.suptitle(tex("Preconditioning Speedup Across All Matrix Shapes"), fontsize=14)
    save(fig, output, "ridge_preconditioning")
    write_csv(output / "preconditioning_values.csv", export)


def real_solver_sets(entries, kind, jit=False):
    if kind == "multinomial":
        suffix = "" if jit else "_no_jit"
        solver_sets = {
            b: ["rlaopt_sapphire", "projected_gradient" + suffix, "jaxopt_lbfgsb" + suffix]
            for b in ("cpu", "cuda")
        }
    else:
        solver_sets = {
            "cpu": ["rlaopt_admm", "scs", "clarabel_qdldl"],
            "cuda": ["rlaopt_admm", "scs_cuda", "cuclarabel_cudss"],
        }
    if kind != "multinomial":
        for backend, extra, position in (
            ("cpu", "scs_cpu_indirect", 1),
            ("cuda", "scs_cuda_direct", 2),
        ):
            if any(r["solver"] == extra for r in entries):
                solver_sets[backend].insert(position, extra)
    return solver_sets


def real_figures(entries, output, datasets, kind, jit=False):
    solver_sets = real_solver_sets(entries, kind, jit)
    fig, axes = plt.subplots(
        2, 2, figsize=(12.5, 7.4), gridspec_kw={"height_ratios": [3.4, 1.1]}, layout="constrained"
    )
    lookup = {(r["backend"], r["solver"], r["dataset"]): r for r in entries if r["kind"] == kind}
    for col, (backend, solvers) in enumerate(solver_sets.items()):
        ax, table = axes[:, col]
        for si, solver in enumerate(solvers):
            for di, dataset in enumerate(datasets):
                r = lookup[backend, solver, dataset]
                x = di + (si - (len(solvers) - 1) / 2) * 0.19
                if r["plot_status"] == "OK":
                    ax.scatter(
                        x,
                        r["runtime"],
                        color=SOLVER_COLORS[solver],
                        s=40,
                        marker="D" if r.get("refined") else "o",
                        zorder=3,
                    )
                elif r["plot_status"] == "TO":
                    ax.annotate(
                        "",
                        xy=(x, 4200),
                        xytext=(x, 3300),
                        arrowprops={"arrowstyle": "->", "color": SOLVER_COLORS[solver], "lw": 1.5},
                    )
                s = r["plot_status"]
                table.add_patch(
                    Rectangle(
                        (di - 0.48, si - 0.44),
                        0.96,
                        0.88,
                        facecolor=STATUS_COLORS[s],
                        edgecolor="white",
                    )
                )
                table.text(di, si, tex(STATUS_LABELS[s]), ha="center", va="center", fontsize=7)
        ax.axhline(3600, color=".5", ls=":", lw=0.8)
        ax.set(yscale="log", ylim=(1, 4700), xlim=(-0.6, len(datasets) - 0.4))
        timeout_ticks(ax, 3600, 1)
        ax.set_xticks(
            range(len(datasets)),
            [dataset_label(lookup[backend, solvers[0], d]) for d in datasets],
            rotation=23,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel(tex("Solver Runtime (s)"))
        ax.set_title(tex("CPU: 64 cores" if backend == "cpu" else "GPU: H200"))
        handles = [
            Line2D([], [], color=SOLVER_COLORS[s], marker="o", ls="", label=label(s))
            for i, s in enumerate(solvers)
        ]
        ax.legend(
            handles=handles,
            fontsize=8,
            loc="upper right" if jit and backend == "cuda" else "lower right",
        )
        grid(ax)
        table.set(xlim=(-0.6, len(datasets) - 0.4), ylim=(len(solvers) - 0.5, -0.5))
        table.set_yticks(range(len(solvers)), [label(s) for s in solvers], fontsize=8)
        table.set_xticks([])
        table.spines[:].set_visible(False)
        table.tick_params(length=0)
        if kind == "bounded_elastic_net":
            ax.axvline(2.5, color=".75", lw=0.8)
            ax.text(1, 1.02, tex("dense"), transform=ax.get_xaxis_transform(), ha="center")
            ax.text(3.5, 1.02, tex("sparse"), transform=ax.get_xaxis_transform(), ha="center")
    fig.suptitle(
        tex(
            "Bounded multinomial logistic regression"
            if kind == "multinomial"
            else "Bounded elastic net"
        ),
        fontsize=14,
    )
    name = "bounded_multinomial" if kind == "multinomial" else "bounded_elastic_net"
    save(fig, output, ("appendix_" + name + "_jit") if jit else name)


def speedups(ridge, real, output):
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.2, 1, 1]}, layout="constrained"
    )
    lookup = {ridge_key(r): r for r in ridge}
    ax = axes[0]
    export = []
    for ai, alpha in enumerate((0.5, 1.0, 2.0)):
        values = defaultdict(list)
        for r in ridge:
            if (r["backend"], r["solver"], r["n"], r["alpha"], r["ridge"]) != (
                "cuda",
                "rlaopt_nystrom_pcg",
                65536,
                alpha,
                1e-6,
            ):
                continue
            cpu = lookup.get(("cpu",) + ridge_key(r)[1:])
            if success(cpu) and success(r):
                ratio = runtime(cpu) / runtime(r)
                values[r["p"]].append(ratio)
                export.append(
                    {
                        "method": "nystrom_pcg",
                        "case": str(r["p"]),
                        "alpha": alpha,
                        "seed": r["seed"],
                        "ratio": ratio,
                        "lower_bound": False,
                    }
                )
        xs = sorted(values)
        ys = [stats.median(values[x]) for x in xs]
        ax.errorbar(
            xs,
            ys,
            yerr=[
                [y - min(values[x]) for x, y in zip(xs, ys)],
                [max(values[x]) - y for x, y in zip(xs, ys)],
            ],
            fmt="o",
            color=SOLVER_COLORS["rlaopt_nystrom_pcg"],
            linestyle=["-", "--", ":"][ai],
            label=rf"$\alpha={alpha:g}$",
            capsize=3,
        )
    power_ticks(ax, sorted({r["p"] for r in ridge if r["n"] == 65536}))
    ax.set_yscale("log")
    ax.axhline(1, color=".5", ls=":")
    ax.set_xlabel(r"$p\quad(n=2^{16},\ \lambda=10^{-6})$")
    ax.set_ylabel(r"$T_{\mathrm{CPU}}/T_{\mathrm{GPU}}$")
    ax.set_title(label("rlaopt_nystrom_pcg"))
    ax.legend(fontsize=8)
    grid(ax)
    for ax, solver in zip(axes[1:], ("rlaopt_sapphire", "rlaopt_admm")):
        lookup_real = {(r["backend"], r["dataset"]): r for r in real if r["solver"] == solver}
        pairs = []
        for (backend, dataset), gpu in lookup_real.items():
            if backend != "cuda" or gpu["plot_status"] != "OK":
                continue
            cpu = lookup_real["cpu", dataset]
            if cpu["plot_status"] not in {"OK", "TO"}:
                continue
            bound = cpu["plot_status"] == "TO"
            ratio = (3600 if bound else cpu["runtime"]) / gpu["runtime"]
            pairs.append((dataset_label(gpu), ratio, bound))
            export.append(
                {
                    "method": solver,
                    "case": dataset,
                    "alpha": "",
                    "seed": gpu["seed"],
                    "ratio": ratio,
                    "lower_bound": bound,
                }
            )
        pairs.sort()
        for i, (_, ratio, bound) in enumerate(pairs):
            if bound:
                ax.annotate(
                    "",
                    xy=(ratio * 1.55, i),
                    xytext=(ratio, i),
                    arrowprops={"arrowstyle": "->", "color": SOLVER_COLORS[solver], "lw": 1.8},
                )
                ax.plot(ratio, i, "|", color=SOLVER_COLORS[solver])
            else:
                ax.scatter(ratio, i, color=SOLVER_COLORS[solver], s=40)
            ax.annotate(
                rf"${'>' if bound else ''}{ratio:.2f}\times$",
                (ratio * math.sqrt(1.55) if bound else ratio, i),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.set_yticks(range(len(pairs)), [p[0] for p in pairs], fontsize=8)
        ax.set(xscale="log", xlim=(0.8, 16), ylim=(len(pairs) - 0.5, -0.65))
        ax.set_xticks([1, 2, 4, 8, 16], [rf"${v}\times$" for v in (1, 2, 4, 8, 16)])
        ax.axvline(1, color=".5", ls=":")
        ax.set_title(label(solver))
        ax.set_xlabel(r"$T_{\mathrm{CPU}}/T_{\mathrm{GPU}}$")
        grid(ax)
    save(fig, output, "cpu_gpu_speedup")
    write_csv(output / "speedup_values.csv", export)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ridge", type=Path, default=Path("artifacts/production-20260831"))
    parser.add_argument("--real", type=Path, default=Path("artifacts/real-erm-production-20260905"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/paper-figures"))
    parser.add_argument("--accuracy-refinement", action="store_true")
    parser.add_argument("--ridge-refinement", type=Path)
    parser.add_argument("--real-supplement", type=Path, action="append", default=[])
    parser.add_argument("--real-refinement", type=Path, action="append", default=[])
    args = parser.parse_args()
    if (args.ridge_refinement or args.real_refinement) and not args.accuracy_refinement:
        parser.error("Refinement inputs require --accuracy-refinement")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    config = tomllib.loads(Path("configs/synthetic.toml").read_text())["experiment"]
    ridge = {}
    sources = []
    for path in sorted((args.ridge / "records").glob("*.json")):
        r = read_record(path) | {"_source": str(path)}
        r["plot_status"] = status(r)
        assert ridge_key(r) not in ridge, f"Duplicate trial: {path}"
        ridge[ridge_key(r)] = r
        sources.append(path)
    expected = set()
    for backend in ("cpu", "cuda"):
        path = args.ridge / "manifests" / f"{backend}.jsonl"
        sources.append(path)
        for line in path.read_text().splitlines():
            job = json.loads(line)
            for lam in config["lambdas"]:
                for repetition in range(config["repetitions"]):
                    r = job | {"ridge": lam, "repetition": repetition}
                    key = ridge_key(r)
                    expected.add(key)
                    if key not in ridge:
                        ridge[key] = r | {
                            "runtime_eligible": False,
                            "plot_status": "MISS",
                            "native_status": "missing",
                            "metadata": {},
                        }
    assert set(ridge) == expected, "Ridge records do not match manifests/configuration"
    # Missing measurements can be traced to failed full-QR warmups. Only assign
    # this cause when all chunk failures are QR warmups and the OOM count agrees.
    for log in sorted((args.ridge / "logs").glob("cpu*.out")):
        content = log.read_text()
        failures = re.findall(r"CHUNK_FAILURE chunk=\d+ manifest_index=(\d+) status=1", content)
        oom_count = sum(map(int, re.findall(r"Detected (\d+) oom_kill", content)))
        if (
            not failures
            or oom_count != len(failures)
            or content.count("warmup = worker.solve(") != len(failures)
        ):
            continue
        host = (
            "soal-8"
            if log.name.startswith("cpu8-")
            else "soal-9"
            if log.name.startswith("cpu9-")
            else None
        )
        if host is None:
            continue
        manifest = args.ridge / "manifests" / f"cpu-{host}.jsonl"
        jobs = [json.loads(line) for line in manifest.read_text().splitlines()]
        failed_jobs = [jobs[int(index)] for index in failures]
        if any(job["solver"] != "torch_qr" for job in failed_jobs):
            continue
        for job in failed_jobs:
            for lam in config["lambdas"]:
                for repetition in range(config["repetitions"]):
                    r = ridge[ridge_key(job | {"ridge": lam, "repetition": repetition})]
                    if r["plot_status"] == "MISS":
                        r["plot_status"] = "WARMUP_OOM"
                        r["failure_evidence"] = str(log)
        sources.extend([log, manifest])
    refinement_audit = []
    if args.accuracy_refinement:
        refinement_audit.extend(
            apply_ridge_refinement(ridge, args.ridge_refinement, ridge_key, sources)
        )
        for r in ridge.values():
            if r.get("plot_status") not in {"MISS", "WARMUP_OOM"}:
                r["plot_status"] = status(r)
    real_roots = [args.real, *args.real_supplement]
    real_records = {}
    for path in sorted(p for root in real_roots for p in (root / "results/records").glob("*.json")):
        r = read_record(path) | {"_source": str(path)}
        assert real_key(r) not in real_records, f"Duplicate trial: {path}"
        real_records[real_key(r)] = r
        sources.append(path)
    if args.accuracy_refinement:
        refinement_audit.extend(
            apply_real_refinement(real_records, args.real_refinement, real_key, sources)
        )
    # Associate scheduler OOM evidence only when a batch has one failed command.
    failure_keys = {}
    for path in sorted(p for root in real_roots for p in (root / "logs").glob("wave-*.out")):
        text = path.read_text()
        failures = re.findall(r"BATCH_FAILURE batch=(\d+) manifest_index=(\d+) status=\d+", text)
        host = re.search(r"(cpu|cuda)-(soal-\d+)-", path.name)
        if host:
            for batch, index in failures:
                manifest = (
                    path.parent.parent
                    / "batches"
                    / f"{host[1]}-{host[2]}"
                    / f"batch-{int(batch):03}.jsonl"
                )
                job = json.loads(manifest.read_text().splitlines()[int(index)])
                classification = (
                    "HOST" if "oom_kill event" in text and len(failures) == 1 else "ERR"
                )
                failure_keys[real_key(job)] = (classification, str(path))
                sources.extend([path, manifest])
    entries = []
    seen = set()
    for root, backend in ((root, b) for root in real_roots for b in ("cpu", "cuda")):
        path = root / "manifests" / f"{backend}.jsonl"
        sources.append(path)
        for line in path.read_text().splitlines():
            job = json.loads(line)
            key = real_key(job)
            assert key not in seen
            seen.add(key)
            r = real_records.get(key)
            dataset = job["problem_spec"]["dataset"]
            display = dataset.replace("_", "-")
            if dataset in {"acsincome", "yearpredictionmsd", "yolanda"}:
                display += "-rf"
            s = status(r)
            evidence = "record" if r else "manifest only"
            if r is None and key in failure_keys:
                s, evidence = failure_keys[key]
            diagnostic = args.real / "logs/diag-cuclar-year-17282628_36.out"
            if (
                dataset == "yearpredictionmsd"
                and job["solver"] == "cuclarabel_cudss"
                and s == "ERR"
                and diagnostic.exists()
                and "alloc device buffer failed" in diagnostic.read_text()
            ):
                s = "GPU"
                evidence = str(diagnostic)
                sources.append(diagnostic)
            entries.append(
                {
                    "dataset": dataset,
                    "n": next(
                        v["problem"]["n"]
                        for v in real_records.values()
                        if v["metadata"]["dataset"] == dataset
                    ),
                    "p": next(
                        v["problem"]["p"]
                        for v in real_records.values()
                        if v["metadata"]["dataset"] == dataset
                    ),
                    "display": display,
                    "kind": job["problem_type"],
                    "backend": backend,
                    "solver": job["solver"],
                    "seed": job["seed"],
                    "plot_status": s,
                    "runtime": runtime(r) if r else None,
                    "native_status": r["native_status"] if r else "missing",
                    "external_success": r.get("external_success") if r else None,
                    "evidence": str(r.get("_source", evidence))
                    if r and evidence == "record"
                    else evidence,
                    "original_native_status": r.get("original_native_status") if r else None,
                    "original_external_success": r.get("original_external_success") if r else None,
                    "original_runtime_seconds": r.get("original_runtime_seconds") if r else None,
                    "displayed_native_tolerance": r.get("displayed_native_tolerance")
                    if r
                    else None,
                    "refined": r.get("refined", False) if r else False,
                    "selected_native_tolerance": r.get("selected_native_tolerance") if r else None,
                    "refinement_attempts": r.get("refinement_attempts", 0) if r else 0,
                    "refinement_last_status": r.get("refinement_last_status") if r else None,
                    "total_measured_solver_seconds": r.get("total_measured_solver_seconds")
                    if r
                    else None,
                }
            )
    assert set(real_records) <= seen, "Unexpected real-data records"
    write_csv(args.output / "real_outcomes.csv", entries)
    write_csv(
        args.output / "ridge_outcomes.csv",
        [
            {
                k: r.get(k)
                for k in (
                    "backend",
                    "solver",
                    "n",
                    "p",
                    "alpha",
                    "ridge",
                    "seed",
                    "repetition",
                    "plot_status",
                    "runtime_seconds",
                    "relative_kkt",
                    "failure_evidence",
                    "refined",
                    "selected_native_tolerance",
                    "refinement_attempts",
                    "refinement_last_status",
                    "original_runtime_seconds",
                    "original_native_status",
                    "original_external_success",
                    "displayed_native_tolerance",
                    "total_measured_solver_seconds",
                )
            }
            for r in ridge.values()
        ],
    )
    if refinement_audit:
        write_csv(args.output / "refinement_attempts.csv", refinement_audit)
        (args.output / "refinement_attempts.json").write_text(
            json.dumps(refinement_audit, indent=2) + "\n"
        )
    ridge_figures(list(ridge.values()), args.output, main_only=args.accuracy_refinement)
    multi = ["cifar10", "fashion_mnist", "news20", "rcv1", "svhn"]
    if not args.accuracy_refinement:
        real_figures(entries, args.output, multi, "multinomial")
        real_figures(entries, args.output, multi, "multinomial", jit=True)
    real_figures(
        entries,
        args.output,
        ["acsincome", "yearpredictionmsd", "yolanda", "e2006", "realsim"],
        "bounded_elastic_net",
    )
    if not args.accuracy_refinement:
        speedups(list(ridge.values()), entries, args.output)
    dataset_rows = []
    for dataset in sorted({r["dataset"] for r in entries}):
        record = next(r for r in real_records.values() if r["metadata"]["dataset"] == dataset)
        n, p = record["problem"]["n"], record["problem"]["p"]
        dataset_rows.append(
            {
                "dataset": next(r["display"] for r in entries if r["dataset"] == dataset),
                "n": n,
                "p": p,
                "structure": "sparse"
                if dataset in {"e2006", "realsim", "news20", "rcv1"}
                else "dense",
                "geometry": "tall" if n > p else "wide" if p > n else "square",
                "random_features": record["metadata"].get("random_features"),
            }
        )
    write_csv(args.output / "dataset_regimes.csv", dataset_rows)
    sources.extend([Path(__file__), Path("configs/synthetic.toml"), args.real / "config.toml"])
    sources.extend(root / "config.toml" for root in args.real_supplement)
    sources.append(Path(__file__).with_name("paper_refinement.py"))
    audit = {
        "ridge_trials": len(ridge),
        "ridge_statuses": dict(Counter(r["plot_status"] for r in ridge.values())),
        "real_trials": len(entries),
        "real_statuses": dict(Counter(r["plot_status"] for r in entries)),
        "success_rule": (
            "first native-and-external passing tolerance; cumulative attempt cost reported separately"
            if args.accuracy_refinement
            else "frozen calibrated native criterion; external diagnostics do not filter runtime points"
        ),
        "failure_rule": "final completed attempt if no native-and-external passing attempt exists",
        "refined_trials": sum(r.get("refined", False) for r in ridge.values())
        + sum(r["refined"] for r in entries),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(sources))
        },
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "source_sha256"}, indent=2))
    print(f"Updated {2 if args.accuracy_refinement else 22} PDF/PNG figure pairs in {args.output}")


if __name__ == "__main__":
    main()

"""Command-line interface for generation, execution, and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlaopt_experiments.config import load_experiment, load_tolerances
from rlaopt_experiments.runner import run_job


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlaopt-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--config", type=Path, default=Path("configs/synthetic.toml"))
    manifest.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    manifest.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run-job")
    for name, kind in (("n", int), ("p", int), ("alpha", float), ("seed", int)):
        run.add_argument(f"--{name}", type=kind, required=True)
    run.add_argument("--solver", required=True)
    run.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    run.add_argument("--config", type=Path, default=Path("configs/synthetic.toml"))
    run.add_argument("--tolerances", type=Path, default=Path("configs/tolerances.toml"))
    run.add_argument("--output", type=Path, default=Path("artifacts"))
    plot = subparsers.add_parser("plot")
    plot.add_argument("--input", type=Path, default=Path("artifacts/records"))
    plot.add_argument("--output", type=Path, default=Path("artifacts/figures"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_experiment(args.config) if hasattr(args, "config") else None
    if args.command == "manifest":
        raw = __import__("tomllib").loads(args.config.read_text())
        solvers = raw["backends"][args.backend]["solvers"]
        jobs = [{"n": shape.n, "p": shape.p, "family": shape.family, "alpha": alpha,
                 "seed": seed, "solver": solver, "backend": args.backend}
                for shape in config.shapes for alpha in config.alphas for seed in config.seeds
                for solver in solvers]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(json.dumps(job, sort_keys=True) for job in jobs) + "\n")
        print(f"wrote {len(jobs)} jobs to {args.output}")
    elif args.command == "run-job":
        tolerance = load_tolerances(args.tolerances, args.backend)[args.solver]
        run_job(n=args.n, p=args.p, alpha=args.alpha, seed=args.seed, solver=args.solver,
                backend=args.backend, ridges=list(config.lambdas), native_tolerance=tolerance,
                kkt_tolerance=config.kkt_tolerance, timeout_seconds=config.timeout_seconds,
                rank=config.nystrom_rank, warmups=config.warmups,
                repetitions=config.repetitions, output_dir=args.output)
    else:
        from rlaopt_experiments.plotting import make_figures
        make_figures(args.input, args.output)

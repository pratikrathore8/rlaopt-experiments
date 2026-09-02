"""Command-line interface for generation, execution, and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlaopt_experiments.config import load_experiment, load_tolerances
from rlaopt_experiments.execution import read_manifest_job, run_manifest_job
from rlaopt_experiments.manifests import build_manifest
from rlaopt_experiments.runner import run_job


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlaopt-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--config", type=Path, default=Path("configs/synthetic.toml"))
    manifest.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest_run = subparsers.add_parser("run-manifest-job")
    manifest_run.add_argument("--manifest", type=Path, required=True)
    manifest_run.add_argument("--index", type=int, required=True)
    manifest_run.add_argument(
        "--config", type=Path, default=Path("configs/synthetic_erm_smoke.toml")
    )
    manifest_run.add_argument("--output", type=Path, default=Path("artifacts"))
    run = subparsers.add_parser("run-job")
    for name, kind in (("n", int), ("p", int), ("alpha", float), ("seed", int)):
        run.add_argument(f"--{name}", type=kind, required=True)
    run.add_argument("--solver", required=True)
    run.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    run.add_argument("--config", type=Path, default=Path("configs/synthetic.toml"))
    run.add_argument("--tolerances", type=Path, default=Path("configs/tolerances.toml"))
    run.add_argument("--output", type=Path, default=Path("artifacts"))
    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--solver", required=True)
    calibration.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    calibration.add_argument(
        "--candidates", type=float, nargs="+", default=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    )
    calibration.add_argument("--config", type=Path, default=Path("configs/synthetic.toml"))
    calibration.add_argument("--output", type=Path, default=Path("artifacts/calibration"))
    plot = subparsers.add_parser("plot")
    plot.add_argument("--input", type=Path, default=Path("artifacts/records"))
    plot.add_argument("--config", type=Path, default=Path("configs/synthetic.toml"))
    plot.add_argument("--output", type=Path, default=Path("artifacts/figures"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "manifest":
        jobs = build_manifest(args.config, args.backend)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(json.dumps(job, sort_keys=True) for job in jobs) + "\n")
        print(f"wrote {len(jobs)} jobs to {args.output}")
        return

    if args.command == "run-manifest-job":
        job = read_manifest_job(args.manifest, args.index)
        run_manifest_job(job, args.config, args.output)
        return

    config = load_experiment(args.config) if hasattr(args, "config") else None
    if args.command == "run-job":
        tolerance = load_tolerances(args.tolerances, args.backend)[args.solver]
        run_job(
            n=args.n,
            p=args.p,
            alpha=args.alpha,
            seed=args.seed,
            solver=args.solver,
            backend=args.backend,
            ridges=list(config.lambdas),
            native_tolerance=tolerance,
            kkt_tolerance=config.kkt_tolerance,
            timeout_seconds=config.timeout_seconds,
            startup_timeout_seconds=config.startup_timeout_seconds,
            rank=config.nystrom_rank,
            warmups=config.warmups,
            repetitions=config.repetitions,
            output_dir=args.output,
            suite=config.suite,
        )
    elif args.command == "calibrate":
        from rlaopt_experiments.calibration import calibrate

        selected = calibrate(
            solver=args.solver,
            backend=args.backend,
            candidates=args.candidates,
            config=config,
            output_dir=args.output / args.backend / args.solver,
        )
        print(f"selected native tolerance for {args.backend}/{args.solver}: {selected:g}")
    else:
        from rlaopt_experiments.plotting import make_figures

        make_figures(args.input, args.output, args.config)

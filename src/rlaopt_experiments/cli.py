"""Command-line interface for generation, execution, and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
        "--problem-type",
        choices=("multinomial", "vanilla_elastic_net", "bounded_elastic_net"),
    )
    calibration.add_argument(
        "--candidates",
        type=float,
        nargs="+",
    )
    calibration.add_argument("--config", type=Path, default=Path("configs/synthetic.toml"))
    calibration.add_argument("--output", type=Path, default=Path("artifacts/calibration"))
    plot = subparsers.add_parser("plot")
    plot.add_argument("--input", type=Path, default=Path("artifacts/records"))
    plot.add_argument("--config", type=Path, default=Path("configs/synthetic.toml"))
    plot.add_argument("--output", type=Path, default=Path("artifacts/figures"))
    real_data = subparsers.add_parser("prepare-real-data")
    selection = real_data.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--dataset", action="append", dest="datasets")
    real_data.add_argument("--data-root", type=Path, required=True)
    real_data.add_argument("--redownload", action="store_true")
    real_data.add_argument("--reprocess", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare-real-data":
        from rlaopt_experiments.suites.real_erm.data import DATASETS, prepare_datasets

        names = list(DATASETS) if args.all else args.datasets
        prepare_datasets(
            names,
            args.data_root,
            redownload=args.redownload,
            reprocess=args.reprocess,
        )
        return

    if args.command == "manifest":
        from rlaopt_experiments.manifests import build_manifest

        jobs = build_manifest(args.config, args.backend)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(json.dumps(job, sort_keys=True) for job in jobs) + "\n")
        print(f"wrote {len(jobs)} jobs to {args.output}")
        return

    if args.command == "run-manifest-job":
        from rlaopt_experiments.execution import read_manifest_job, run_manifest_job

        job = read_manifest_job(args.manifest, args.index)
        run_manifest_job(job, args.config, args.output)
        return

    if args.command == "calibrate":
        from rlaopt_experiments.manifests import configured_suite

        suite = configured_suite(args.config)
        if suite == "synthetic_erm":
            if args.problem_type is None:
                raise ValueError("--problem-type is required when calibrating synthetic_erm")
            if args.candidates is not None:
                raise ValueError(
                    "synthetic_erm calibration candidates must be defined in its TOML file"
                )
            from rlaopt_experiments.suites.synthetic_erm.calibration import (
                calibrate_synthetic_erm,
            )
            from rlaopt_experiments.suites.synthetic_erm.config import (
                load_synthetic_erm_calibration_config,
            )

            calibration_config = load_synthetic_erm_calibration_config(args.config)
            selected = calibrate_synthetic_erm(
                problem_type=args.problem_type,
                solver=args.solver,
                backend=args.backend,
                candidates=list(calibration_config.candidates),
                config=calibration_config.experiment,
                output_dir=(args.output / args.backend / args.problem_type / args.solver),
            )
        elif suite == "synthetic_ridge":
            if args.problem_type is not None:
                raise ValueError("--problem-type is only valid when calibrating synthetic_erm")
            from rlaopt_experiments.calibration import calibrate
            from rlaopt_experiments.config import load_experiment

            selected = calibrate(
                solver=args.solver,
                backend=args.backend,
                candidates=args.candidates or [1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                config=load_experiment(args.config),
                output_dir=args.output / args.backend / args.solver,
            )
        else:
            raise ValueError(f"calibration is not implemented for suite: {suite}")
        print(
            f"selected native tolerance for "
            f"{args.backend}/{args.problem_type or 'ridge'}/{args.solver}: {selected:g}"
        )
        return

    from rlaopt_experiments.config import load_experiment

    config = load_experiment(args.config) if hasattr(args, "config") else None
    if args.command == "run-job":
        from rlaopt_experiments.config import load_tolerances
        from rlaopt_experiments.runner import run_job

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
    else:
        from rlaopt_experiments.plotting import make_figures

        make_figures(args.input, args.output, args.config)

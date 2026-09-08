"""Freeze native-success ridge accuracy misses for conditional tolerance refinement."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from rlaopt_experiments.config import load_experiment
from rlaopt_experiments.records import read_record


def prepare(records: Path, config_path: Path, output: Path, image_sha256: str) -> dict:
    config = load_experiment(config_path)
    if config.repetitions != 1:
        raise ValueError("This follow-up requires the original one-repetition protocol")
    jobs = []
    for path in sorted(records.glob("*.json")):
        record = read_record(path)
        if not (record["native_success"] and not record["external_success"]):
            continue
        if (record["backend"], record["solver"]) not in {
            ("cpu", "scipy_lsqr"), ("cuda", "cuml_lsmr")
        }:
            raise ValueError(f"Unexpected refinement candidate: {path}")
        if record.get("repetition", 0) != 0:
            raise ValueError("Expected one original measurement per trial")
        if record["metadata"]["native_tolerance"] != 1e-9:
            raise ValueError("The predefined refinement ladder starts from native tolerance 1e-9")
        node = record["metadata"]["hostname"].split(".")[0]
        if node not in ({"soal-8", "soal-9"} if record["backend"] == "cpu" else {"soal-12"}):
            raise ValueError(f"Unexpected original node: {node}")
        jobs.append({
            "trial_id": path.stem,
            "original_record": str(path.resolve()),
            "original_record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "original_runtime_seconds": record["runtime_seconds"],
            "original_relative_kkt": record["relative_kkt"],
            "original_image_sha256": record["metadata"].get("cuda_image_sha256"),
            "image_sha256": image_sha256,
            "node": node, "cpu_threads": 64,
            "native_tolerances": [1e-10, 1e-11],
            "controls": {key: getattr(config, key) for key in (
                "suite", "kkt_tolerance", "timeout_seconds", "startup_timeout_seconds",
                "warmups", "repetitions",
            )} | {"rank": config.nystrom_rank},
            "problem": {key: record[key] for key in (
                "n", "p", "alpha", "ridge", "seed", "solver", "backend",
            )},
        })
    if not jobs:
        raise ValueError("No native-success accuracy misses found")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.toml").write_bytes(config_path.read_bytes())
    (output / "original-records").mkdir()
    for job in jobs:
        source = Path(job["original_record"])
        frozen = output / "original-records" / source.name
        frozen.write_bytes(source.read_bytes())
        job["original_record"] = str(frozen.resolve())
    (output / "candidates.json").write_text(json.dumps(jobs, indent=2) + "\n")
    manifests = []
    for node in ("soal-8", "soal-9", "soal-12"):
        selected = [job for job in jobs if job["node"] == node]
        for start in range(0, len(selected), 6):
            wave = start // 6
            path = output / f"wave-{wave:03d}-{node}.jsonl"
            chunk = selected[start:start + 6]
            path.write_text("".join(json.dumps(job, sort_keys=True) + "\n" for job in chunk))
            path.with_suffix(".jsonl.sha256").write_text(
                hashlib.sha256(path.read_bytes()).hexdigest() + "  " + str(path.resolve()) + "\n"
            )
            manifests.append({"wave": wave, "node": node, "tasks": len(chunk),
                              "concurrency": 2 if node == "soal-12" else 1,
                              "path": str(path.resolve())})
    plan = {"phase": "post-production accuracy refinement", "submitted": False,
            "candidate_count": len(jobs), "native_tolerances": [1e-10, 1e-11],
            "second_attempt_rule": "only if first attempt has native success and external failure",
            "config": asdict(config), "image_sha256": image_sha256, "manifests": manifests}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("artifacts/production-20260831/records"))
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.toml"))
    parser.add_argument("--image-checksum", type=Path,
                        default=Path("containers/rlaopt-cuda-13.0.2.sif.sha256"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.records, args.config, args.output,
                             args.image_checksum.read_text().split()[0]), indent=2))


if __name__ == "__main__":
    main()

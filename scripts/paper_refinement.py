"""Auditable selection of accuracy-qualified post-production measurements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rlaopt_experiments.records import read_record


def qualified(row):
    return bool(row.get("native_success") and row.get("external_success"))


def select_attempt(original, attempts):
    """Select the first passing setting, never the fastest; retain misses if none pass."""
    if attempts and not (original["native_success"] and not original["external_success"]):
        raise ValueError("Refinement requires an original native-success accuracy miss")
    previous = original.get("metadata", {}).get("native_tolerance")
    for row in attempts:
        tolerance = row["metadata"]["native_tolerance"]
        if previous is None or not 0 < tolerance < previous:
            raise ValueError("Refinement tolerances must be strictly decreasing")
        previous = tolerance
    rows = [original, *attempts]
    chosen = next((i for i, row in enumerate(rows) if qualified(row)), None)
    selected = dict(rows[chosen] if chosen is not None else original)
    selected["plot_eligible"] = chosen is not None
    selected["refined"] = chosen is not None and chosen > 0
    selected["selected_native_tolerance"] = (
        rows[chosen].get("metadata", {}).get("native_tolerance") if chosen is not None else None
    )
    selected["refinement_attempts"] = len(attempts)
    selected["total_measured_solver_seconds"] = sum(r.get("runtime_seconds", 0.0) for r in rows)
    selected["original_native_status"] = original["native_status"]
    selected["original_external_success"] = original["external_success"]
    selected["original_runtime_seconds"] = original.get("runtime_seconds")
    selected["refinement_last_status"] = attempts[-1]["native_status"] if attempts else None
    audit = []
    for i, row in enumerate(rows):
        audit.append(
            {
                "problem_id": original.get("problem_id"),
                "backend": original["backend"],
                "solver": original["solver"],
                "seed": original["seed"],
                "n": original.get("n"),
                "p": original.get("p"),
                "alpha": original.get("alpha"),
                "ridge": original.get("ridge"),
                "repetition": original.get("repetition", 0),
                "attempt": i,
                "source": row.get("_source"),
                "native_tolerance": row.get("metadata", {}).get("native_tolerance"),
                "native_status": row["native_status"],
                "native_success": row["native_success"],
                "external_success": row["external_success"],
                "runtime_seconds": row.get("runtime_seconds"),
                "relative_kkt": row.get("relative_kkt"),
                "stationarity": row.get("stationarity"),
                "feasibility": row.get("feasibility"),
                "selected": i == chosen,
                "total_measured_solver_seconds": selected["total_measured_solver_seconds"],
            }
        )
    return selected, audit


def load_attempt(path, sources):
    row = read_record(path) | {"_source": str(path)}
    sources.append(path)
    return row


def apply_ridge_refinement(records, root, key, sources):
    attempts_by_key = {}
    if root is not None:
        candidates_path = root / "candidates.json"
        sources.append(candidates_path)
        for job in json.loads(candidates_path.read_text()):
            original_path = Path(job["original_record"])
            if (
                hashlib.sha256(original_path.read_bytes()).hexdigest()
                != job["original_record_sha256"]
            ):
                raise ValueError("Frozen refinement original checksum mismatch")
            original = load_attempt(original_path, sources)
            k = key(original)
            if k not in records:
                raise ValueError("Refinement trial absent from production inputs")
            if (
                hashlib.sha256(Path(records[k]["_source"]).read_bytes()).hexdigest()
                != job["original_record_sha256"]
            ):
                raise ValueError("Refinement original differs from plotted production record")
            summary_path = root / "results" / job["trial_id"] / "summary.json"
            attempts = []
            # A completed summary is the commit point; workers may still be writing records.
            if summary_path.exists():
                sources.append(summary_path)
                summary = json.loads(summary_path.read_text())
                for index, attempt in enumerate(summary["attempts"]):
                    if attempt["native_tolerance"] != job["native_tolerances"][index]:
                        raise ValueError("Attempt does not match predefined ladder")
                    row = load_attempt(Path(attempt["record"]), sources)
                    if (
                        key(row) != k
                        or row["metadata"]["native_tolerance"] != attempt["native_tolerance"]
                    ):
                        raise ValueError("Refinement record identity or tolerance mismatch")
                    attempts.append(row)
            if k in attempts_by_key:
                raise ValueError("Duplicate refinement candidate")
            attempts_by_key[k] = attempts
    audit = []
    for k, row in records.items():
        if row.get("native_status") == "missing":
            row["plot_eligible"] = False
            continue
        selected, trial_audit = select_attempt(row, attempts_by_key.get(k, []))
        records[k] = selected
        if k in attempts_by_key:
            audit.extend(trial_audit)
    return audit


def apply_real_refinement(records, roots, key, sources):
    attempts = {}
    for root in roots:
        sources.append(root / "config.toml")
        for path in sorted((root / "results/records").glob("*.json")):
            row = load_attempt(path, sources)
            k = key(row)
            if k not in records:
                raise ValueError("Real refinement has no matching original; include its supplement")
            attempts.setdefault(k, []).append(row)
    audit = []
    for k, row in records.items():
        ordered = sorted(
            attempts.get(k, []), key=lambda r: r["metadata"]["native_tolerance"], reverse=True
        )
        records[k], trial_audit = select_attempt(row, ordered)
        if ordered:
            audit.extend(trial_audit)
    return audit

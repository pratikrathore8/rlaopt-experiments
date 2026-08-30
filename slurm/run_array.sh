#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --array=0-287
set -euo pipefail

BACKEND="${BACKEND:?set BACKEND to cpu or cuda}"
MANIFEST="${MANIFEST:?set MANIFEST to a JSONL manifest}"
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
read -r N P ALPHA SEED SOLVER < <(python -c \
  'import json,sys; j=json.loads(sys.argv[1]); print(j["n"],j["p"],j["alpha"],j["seed"],j["solver"])' "$LINE")
timeout --signal=TERM 59m uv run rlaopt-bench run-job \
  --n "$N" --p "$P" --alpha "$ALPHA" --seed "$SEED" --solver "$SOLVER" --backend "$BACKEND"

#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
set -euo pipefail

export OMP_NUM_THREADS="${BENCHMARK_CPU_THREADS:-64}"
export MKL_NUM_THREADS="${BENCHMARK_CPU_THREADS:-64}"
export OPENBLAS_NUM_THREADS="${BENCHMARK_CPU_THREADS:-64}"
if [[ -f "${SLURM_SUBMIT_DIR:-.}/containers/rapids.env" ]]; then
  set -a
  source "${SLURM_SUBMIT_DIR:-.}/containers/rapids.env"
  set +a
fi

BACKEND="${BACKEND:?set BACKEND to cpu or cuda}"
MANIFEST="${MANIFEST:?set MANIFEST to a JSONL manifest}"
if [[ "$BACKEND" == "cuda" && ! "${RAPIDS_IMAGE_DIGEST:-}" =~ ^sha256:[0-9a-f]{64}$ \
      && "${ALLOW_MUTABLE_RAPIDS_TAG:-0}" != "1" ]]; then
  echo "Refusing CUDA production run without an immutable RAPIDS digest." >&2
  echo "Set ALLOW_MUTABLE_RAPIDS_TAG=1 only for smoke tests." >&2
  exit 2
fi
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
read -r N P ALPHA SEED SOLVER < <(python -c \
  'import json,sys; j=json.loads(sys.argv[1]); print(j["n"],j["p"],j["alpha"],j["seed"],j["solver"])' "$LINE")
timeout --signal=TERM 59m uv run rlaopt-bench run-job \
  --n "$N" --p "$P" --alpha "$ALPHA" --seed "$SEED" --solver "$SOLVER" --backend "$BACKEND"

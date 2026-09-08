#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
REPOSITORY="${SLURM_SUBMIT_DIR:?submit from repository root}"
MANIFEST="${MANIFEST:?set the frozen refinement manifest}"
OUTPUT_DIR="${OUTPUT_DIR:?set the separate refinement results directory}"
INDEX="${SLURM_ARRAY_TASK_ID:?submit as an array}"
cd "$REPOSITORY"
sha256sum --check --status "$MANIFEST.sha256"
source containers/cuda.env
IMAGE="$REPOSITORY/$RLAOPT_CUDA_IMAGE"
export RLAOPT_CUDA_IMAGE_SHA256="$(cut -d' ' -f1 "$IMAGE.sha256")"
export APPTAINERENV_RLAOPT_CUDA_IMAGE_SHA256="$RLAOPT_CUDA_IMAGE_SHA256"
export APPTAINERENV_RLAOPT_GIT_COMMIT="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'Refusing ridge refinement from a dirty checkout.' >&2
  exit 2
fi
export APPTAINERENV_RLAOPT_GIT_DIRTY=0
export BENCHMARK_CPU_THREADS=64 OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64
read -r BACKEND EXPECTED_NODE < <(python3 - "$MANIFEST" "$INDEX" <<'PY'
import json, sys
job = json.loads(open(sys.argv[1]).read().splitlines()[int(sys.argv[2])])
print(job['problem']['backend'], job['node'])
PY
)
if [[ "$(hostname -s)" != "$EXPECTED_NODE" ]]; then
  echo "Expected original node $EXPECTED_NODE" >&2
  exit 2
fi
ARGS=(--bind "$REPOSITORY:$REPOSITORY" --pwd "$REPOSITORY")
if [[ "$BACKEND" == cuda ]]; then ARGS=(--nv "${ARGS[@]}"); fi
# Two attempts each allow 15 min startup, 15 min warmup, and 15 min solve.
timeout --signal=TERM 115m apptainer exec "${ARGS[@]}" "$IMAGE" \
  /opt/rlaopt-experiments/.venv/bin/python scripts/run_ridge_refinement.py \
  --manifest "$MANIFEST" --index "$INDEX" --output "$OUTPUT_DIR"

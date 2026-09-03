#!/usr/bin/env bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
set -euo pipefail

export OMP_NUM_THREADS="${BENCHMARK_CPU_THREADS:-64}"
export MKL_NUM_THREADS="${BENCHMARK_CPU_THREADS:-64}"
export OPENBLAS_NUM_THREADS="${BENCHMARK_CPU_THREADS:-64}"
if [[ -f "${SLURM_SUBMIT_DIR:-.}/containers/cuda.env" ]]; then
  set -a
  source "${SLURM_SUBMIT_DIR:-.}/containers/cuda.env"
  set +a
fi

BACKEND="${BACKEND:?set BACKEND to cpu or cuda}"
MANIFEST="${MANIFEST:?set MANIFEST to a JSONL manifest}"
CONFIG="${CONFIG:-configs/synthetic.toml}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts}"
MANIFEST_INDEX="${MANIFEST_INDEX:-${SLURM_ARRAY_TASK_ID:?set MANIFEST_INDEX or submit as an array}}"
if [[ "$BACKEND" == "cuda" && ! "${CUDA_BASE_IMAGE_DIGEST:-}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Refusing CUDA run without an immutable base-image digest." >&2
  exit 2
fi
LINE="$(sed -n "$((MANIFEST_INDEX + 1))p" "$MANIFEST")"
if [[ -z "$LINE" ]]; then
  echo "No manifest record at zero-based index $MANIFEST_INDEX in $MANIFEST" >&2
  exit 2
fi
REPOSITORY="${SLURM_SUBMIT_DIR:?submit jobs from the repository root}"
RLAOPT_GIT_COMMIT="$(git -C "$REPOSITORY" rev-parse HEAD)"
if [[ -n "$(git -C "$REPOSITORY" status --porcelain)" ]]; then
  RLAOPT_GIT_DIRTY=1
else
  RLAOPT_GIT_DIRTY=0
fi
export APPTAINERENV_RLAOPT_GIT_COMMIT="$RLAOPT_GIT_COMMIT"
export APPTAINERENV_RLAOPT_GIT_DIRTY="$RLAOPT_GIT_DIRTY"
echo "RLAOPT_GIT_COMMIT=$RLAOPT_GIT_COMMIT"
echo "RLAOPT_GIT_DIRTY=$RLAOPT_GIT_DIRTY"

DERIVED_IMAGE="${RLAOPT_CUDA_IMAGE:-containers/rlaopt-cuda-${CUDA_IMAGE_VERSION}.sif}"
if [[ "$DERIVED_IMAGE" != /* ]]; then
  DERIVED_IMAGE="$REPOSITORY/$DERIVED_IMAGE"
fi
if [[ ! -f "$DERIVED_IMAGE" ]]; then
  echo "Benchmark image not found: $DERIVED_IMAGE" >&2
  echo "Submit slurm/build_cuda.sh first." >&2
  exit 2
fi
RLAOPT_CUDA_IMAGE_SHA256="$(cut -d' ' -f1 "$DERIVED_IMAGE.sha256")"
export RLAOPT_CUDA_IMAGE_SHA256
export APPTAINERENV_RLAOPT_CUDA_IMAGE_SHA256="$RLAOPT_CUDA_IMAGE_SHA256"
echo "RLAOPT_CUDA_IMAGE_SHA256=$RLAOPT_CUDA_IMAGE_SHA256"

read -r SUITE JOB_BACKEND < <(apptainer exec "$DERIVED_IMAGE" \
  /opt/rlaopt-experiments/.venv/bin/python -c \
  'import json,sys; j=json.loads(sys.argv[1]); print(j.get("suite"), j.get("backend"))' "$LINE")
if [[ "$JOB_BACKEND" != "$BACKEND" ]]; then
  echo "Manifest backend $JOB_BACKEND does not match requested backend $BACKEND" >&2
  exit 2
fi

APPTAINER_ARGS=(--bind "$REPOSITORY:$REPOSITORY" --pwd "$REPOSITORY")
if [[ "$BACKEND" == "cuda" ]]; then
  APPTAINER_ARGS=(--nv "${APPTAINER_ARGS[@]}")
fi

if [[ "$SUITE" == "synthetic_erm" ]]; then
  timeout --signal=TERM 179m apptainer exec "${APPTAINER_ARGS[@]}" \
    "$DERIVED_IMAGE" /opt/rlaopt-experiments/.venv/bin/rlaopt-bench run-manifest-job \
    --manifest "$MANIFEST" --index "$MANIFEST_INDEX" \
    --config "$CONFIG" --output "$OUTPUT_DIR"
  exit
fi
if [[ "$SUITE" != "synthetic_ridge" ]]; then
  echo "Unsupported manifest suite: $SUITE" >&2
  exit 2
fi

read -r N P ALPHA SEED SOLVER < <(apptainer exec "$DERIVED_IMAGE" \
  /opt/rlaopt-experiments/.venv/bin/python -c \
  'import json,sys; j=json.loads(sys.argv[1]); print(j["n"],j["p"],j["alpha"],j["seed"],j["solver"])' "$LINE")
timeout --signal=TERM 179m apptainer exec "${APPTAINER_ARGS[@]}" \
  "$DERIVED_IMAGE" /opt/rlaopt-experiments/.venv/bin/rlaopt-bench run-job \
  --config "$CONFIG" --output "$OUTPUT_DIR" \
  --n "$N" --p "$P" --alpha "$ALPHA" --seed "$SEED" --solver "$SOLVER" --backend "$BACKEND"

#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
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
if [[ "$BACKEND" == "cuda" && ! "${CUDA_BASE_IMAGE_DIGEST:-}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Refusing CUDA run without an immutable base-image digest." >&2
  exit 2
fi
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"

if [[ "$BACKEND" == "cuda" ]]; then
  REPOSITORY="${SLURM_SUBMIT_DIR:?submit CUDA jobs from the repository root}"
  DERIVED_IMAGE="${RLAOPT_CUDA_IMAGE:-containers/rlaopt-cuda-${CUDA_IMAGE_VERSION}.sif}"
  if [[ "$DERIVED_IMAGE" != /* ]]; then
    DERIVED_IMAGE="$REPOSITORY/$DERIVED_IMAGE"
  fi
  if [[ ! -f "$DERIVED_IMAGE" ]]; then
    echo "Derived image not found: $DERIVED_IMAGE" >&2
    echo "Submit slurm/build_cuda.sh first." >&2
    exit 2
  fi
  RLAOPT_CUDA_IMAGE_SHA256="$(sha256sum "$DERIVED_IMAGE" | cut -d' ' -f1)"
  export RLAOPT_CUDA_IMAGE_SHA256
  export APPTAINERENV_RLAOPT_CUDA_IMAGE_SHA256="$RLAOPT_CUDA_IMAGE_SHA256"
  echo "RLAOPT_CUDA_IMAGE_SHA256=$RLAOPT_CUDA_IMAGE_SHA256"
  read -r N P ALPHA SEED SOLVER < <(apptainer exec "$DERIVED_IMAGE" \
    /opt/rlaopt-experiments/.venv/bin/python -c \
    'import json,sys; j=json.loads(sys.argv[1]); print(j["n"],j["p"],j["alpha"],j["seed"],j["solver"])' "$LINE")
  timeout --signal=TERM 59m apptainer exec --nv \
    --bind "$REPOSITORY:$REPOSITORY" --pwd "$REPOSITORY" \
    "$DERIVED_IMAGE" /opt/rlaopt-experiments/.venv/bin/rlaopt-bench run-job \
    --config "$CONFIG" \
    --n "$N" --p "$P" --alpha "$ALPHA" --seed "$SEED" --solver "$SOLVER" --backend "$BACKEND"
else
  read -r N P ALPHA SEED SOLVER < <(uv run --frozen python -c \
    'import json,sys; j=json.loads(sys.argv[1]); print(j["n"],j["p"],j["alpha"],j["seed"],j["solver"])' "$LINE")
  timeout --signal=TERM 59m uv run --frozen rlaopt-bench run-job \
    --config "$CONFIG" \
    --n "$N" --p "$P" --alpha "$ALPHA" --seed "$SEED" --solver "$SOLVER" --backend "$BACKEND"
fi

#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=stage-real-data
#SBATCH --output=stage-real-data-%N-%j.out
set -euo pipefail

REPOSITORY="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
DATA_ROOT="${DATA_ROOT:-/scr/pratikr/rlaopt-real-data}"
source "$REPOSITORY/containers/cuda.env"
DERIVED_IMAGE="${RLAOPT_CUDA_IMAGE:-containers/rlaopt-cuda-${CUDA_IMAGE_VERSION}.sif}"
if [[ "$DERIVED_IMAGE" != /* ]]; then
  DERIVED_IMAGE="$REPOSITORY/$DERIVED_IMAGE"
fi
if [[ ! -f "$DERIVED_IMAGE" ]]; then
  echo "Benchmark image not found: $DERIVED_IMAGE" >&2
  echo "Submit slurm/build_cuda.sh first." >&2
  exit 2
fi

mkdir -p "$DATA_ROOT"
APPTAINER_ARGS=(
  --bind "$REPOSITORY:$REPOSITORY"
  --bind "$DATA_ROOT:$DATA_ROOT"
  --pwd "$REPOSITORY"
)
PREPARE_ARGS=(--all --data-root "$DATA_ROOT")
if [[ "${REDOWNLOAD:-0}" == "1" ]]; then
  PREPARE_ARGS+=(--redownload)
elif [[ "${REPROCESS:-0}" == "1" ]]; then
  PREPARE_ARGS+=(--reprocess)
fi

apptainer exec "${APPTAINER_ARGS[@]}" "$DERIVED_IMAGE" \
  /opt/rlaopt-experiments/.venv/bin/rlaopt-bench prepare-real-data "${PREPARE_ARGS[@]}"
apptainer exec "${APPTAINER_ARGS[@]}" "$DERIVED_IMAGE" \
  /opt/rlaopt-experiments/.venv/bin/rlaopt-bench verify-real-data \
  --all --data-root "$DATA_ROOT"

echo "REAL_DATA_CACHE_READY node=$(hostname) data_root=$DATA_ROOT"

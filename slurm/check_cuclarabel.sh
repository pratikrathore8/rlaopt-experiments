#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodelist=soal-12
#SBATCH --gres=gpu:h200nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=rlaopt-cuclarabel-check
#SBATCH --output=cuclarabel-check-%j.out

set -euo pipefail

REPOSITORY="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
source "$REPOSITORY/containers/cuda.env"

DERIVED_IMAGE="$RLAOPT_CUDA_IMAGE"
if [[ "$DERIVED_IMAGE" != /* ]]; then
  DERIVED_IMAGE="$REPOSITORY/$DERIVED_IMAGE"
fi
if [[ ! -f "$DERIVED_IMAGE" ]]; then
  echo "Derived image not found: $DERIVED_IMAGE" >&2
  echo "Submit slurm/build_cuda.sh first." >&2
  exit 2
fi
if [[ ! -f "$DERIVED_IMAGE.sha256" ]]; then
  echo "Image checksum not found: $DERIVED_IMAGE.sha256" >&2
  exit 2
fi
(
  cd "$(dirname "$DERIVED_IMAGE")"
  sha256sum --check "$(basename "$DERIVED_IMAGE").sha256"
)

hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi
apptainer inspect "$DERIVED_IMAGE"
apptainer exec --nv --bind "$REPOSITORY:$REPOSITORY" --pwd "$REPOSITORY" \
  "$DERIVED_IMAGE" /opt/rlaopt-experiments/.venv/bin/python \
  scripts/check_cuclarabel.py --backend both

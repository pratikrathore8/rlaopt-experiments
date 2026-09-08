#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodelist=soal-12
#SBATCH --gres=gpu:h200nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=rlaopt-cuda-check
#SBATCH --output=cuda-check-%j.out

set -euo pipefail

REPOSITORY="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
source "$REPOSITORY/containers/cuda.env"

if [[ ! "$CUDA_BASE_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "containers/cuda.env does not contain an immutable digest" >&2
  exit 2
fi

DERIVED_IMAGE="$RLAOPT_CUDA_IMAGE"
if [[ "$DERIVED_IMAGE" != /* ]]; then
  DERIVED_IMAGE="$REPOSITORY/$DERIVED_IMAGE"
fi
if [[ ! -f "$DERIVED_IMAGE" ]]; then
  echo "Derived image not found: $DERIVED_IMAGE" >&2
  echo "Submit slurm/build_cuda.sh first." >&2
  exit 2
fi

hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi
echo "CUDA_IMAGE_SHA256=$(cut -d' ' -f1 "$DERIVED_IMAGE.sha256")"
apptainer inspect "$DERIVED_IMAGE"
apptainer exec --nv --bind "$REPOSITORY:$REPOSITORY" --pwd "$REPOSITORY" \
  "$DERIVED_IMAGE" /opt/rlaopt-experiments/.venv/bin/python scripts/check_gpu_environment.py
apptainer exec --nv --bind "$REPOSITORY:$REPOSITORY" --pwd "$REPOSITORY" \
  "$DERIVED_IMAGE" /opt/rlaopt-experiments/.venv/bin/python scripts/check_scs_backends.py --backend cuda

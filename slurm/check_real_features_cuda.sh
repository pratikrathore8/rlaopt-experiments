#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodelist=soal-12
#SBATCH --gres=gpu:h200nvl:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --job-name=real-features-cuda
#SBATCH --output=real-features-cuda-%j.out

set -euo pipefail

REPOSITORY="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
source "$REPOSITORY/containers/cuda.env"
DERIVED_IMAGE="$RLAOPT_CUDA_IMAGE"
if [[ "$DERIVED_IMAGE" != /* ]]; then
  DERIVED_IMAGE="$REPOSITORY/$DERIVED_IMAGE"
fi

apptainer exec --nv --bind "$REPOSITORY:$REPOSITORY" --pwd "$REPOSITORY" \
  --env "PYTHONPATH=$REPOSITORY/src" \
  "$DERIVED_IMAGE" /opt/rlaopt-experiments/.venv/bin/python \
  scripts/check_real_features_cuda.py

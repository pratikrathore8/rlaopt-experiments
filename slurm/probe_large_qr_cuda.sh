#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodelist=soal-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=probe-qr-cuda
#SBATCH --output=probe-qr-cuda-%j.out

set -euo pipefail
REPOSITORY="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
source "$REPOSITORY/containers/cuda.env"
DERIVED_IMAGE="$REPOSITORY/$RLAOPT_CUDA_IMAGE"
cd "$REPOSITORY"
/usr/bin/time -v apptainer exec --nv --bind "$REPOSITORY:$REPOSITORY" --pwd "$REPOSITORY" \
  "$DERIVED_IMAGE" /opt/rlaopt-experiments/.venv/bin/python \
  scripts/probe_large_qr.py --backend cuda --dimension 65536

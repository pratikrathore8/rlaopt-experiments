#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodelist=soal-8,soal-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=torch-qr-drivers
#SBATCH --output=torch-qr-drivers-%j.out

set -euo pipefail

REPOSITORY="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
source "$REPOSITORY/containers/cuda.env"
DERIVED_IMAGE="$RLAOPT_CUDA_IMAGE"
if [[ "$DERIVED_IMAGE" != /* ]]; then
  DERIVED_IMAGE="$REPOSITORY/$DERIVED_IMAGE"
fi
export BENCHMARK_CPU_THREADS=64
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64

cd "$REPOSITORY"
apptainer exec --bind "$REPOSITORY:$REPOSITORY" --pwd "$REPOSITORY" \
  "$DERIVED_IMAGE" /opt/rlaopt-experiments/.venv/bin/python \
  scripts/compare_torch_qr_drivers.py

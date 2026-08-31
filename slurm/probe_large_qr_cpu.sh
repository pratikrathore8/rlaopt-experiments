#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodelist=soal-8,soal-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem=384G
#SBATCH --time=00:30:00
#SBATCH --job-name=probe-qr-cpu
#SBATCH --output=probe-qr-cpu-%j.out

set -euo pipefail
REPOSITORY="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
source "$REPOSITORY/containers/cuda.env"
DERIVED_IMAGE="$REPOSITORY/$RLAOPT_CUDA_IMAGE"
export BENCHMARK_CPU_THREADS=64 OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64
cd "$REPOSITORY"
/usr/bin/time -v apptainer exec --bind "$REPOSITORY:$REPOSITORY" --pwd "$REPOSITORY" \
  "$DERIVED_IMAGE" /opt/rlaopt-experiments/.venv/bin/python \
  scripts/probe_large_qr.py --backend cpu --dimension 65536

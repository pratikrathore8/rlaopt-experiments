#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodelist=soal-8,soal-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=rlaopt-cpu-calibrate
#SBATCH --output=cpu-calibrate-%j.out

set -euo pipefail

REPOSITORY="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
source "$REPOSITORY/containers/cuda.env"
SOLVER="${SOLVER:?set SOLVER to an iterative CPU solver}"
CONFIG="${CONFIG:-configs/calibration.toml}"
CANDIDATES="${CANDIDATES:-1e-5 3e-6 1e-6 3e-7 1e-7 3e-8 1e-8}"

DERIVED_IMAGE="$RLAOPT_CUDA_IMAGE"
if [[ "$DERIVED_IMAGE" != /* ]]; then
  DERIVED_IMAGE="$REPOSITORY/$DERIVED_IMAGE"
fi
if [[ ! -f "$DERIVED_IMAGE" ]]; then
  echo "Benchmark image not found: $DERIVED_IMAGE" >&2
  exit 2
fi

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
read -r -a CANDIDATE_ARRAY <<< "$CANDIDATES"
echo "HOST=$(hostname)"
echo "SOLVER=$SOLVER"
echo "CONFIG=$CONFIG"
echo "CANDIDATES=${CANDIDATE_ARRAY[*]}"
RLAOPT_CUDA_IMAGE_SHA256="$(cut -d' ' -f1 "$DERIVED_IMAGE.sha256")"
export APPTAINERENV_RLAOPT_CUDA_IMAGE_SHA256="$RLAOPT_CUDA_IMAGE_SHA256"
echo "RLAOPT_CUDA_IMAGE_SHA256=$RLAOPT_CUDA_IMAGE_SHA256"

cd "$REPOSITORY"
apptainer exec --bind "$REPOSITORY:$REPOSITORY" --pwd "$REPOSITORY" \
  "$DERIVED_IMAGE" /opt/rlaopt-experiments/.venv/bin/rlaopt-bench calibrate \
  --backend cpu --solver "$SOLVER" --config "$CONFIG" \
  --output artifacts/calibration --candidates "${CANDIDATE_ARRAY[@]}"

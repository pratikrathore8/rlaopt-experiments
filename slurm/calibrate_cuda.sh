#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodelist=soal-12
#SBATCH --gres=gpu:h200nvl:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=rlaopt-cuda-calibrate
#SBATCH --output=cuda-calibrate-%j.out

set -euo pipefail

REPOSITORY="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
source "$REPOSITORY/containers/cuda.env"
SOLVER="${SOLVER:?set SOLVER to an iterative CUDA solver}"
CONFIG="${CONFIG:-configs/calibration.toml}"
CANDIDATES="${CANDIDATES:-1e-5 3e-6 1e-6 3e-7 1e-7 3e-8 1e-8}"

DERIVED_IMAGE="$RLAOPT_CUDA_IMAGE"
if [[ "$DERIVED_IMAGE" != /* ]]; then
  DERIVED_IMAGE="$REPOSITORY/$DERIVED_IMAGE"
fi
if [[ ! -f "$DERIVED_IMAGE" ]]; then
  echo "Derived image not found: $DERIVED_IMAGE" >&2
  exit 2
fi

RLAOPT_CUDA_IMAGE_SHA256="$(cut -d' ' -f1 "$DERIVED_IMAGE.sha256")"
export APPTAINERENV_RLAOPT_CUDA_IMAGE_SHA256="$RLAOPT_CUDA_IMAGE_SHA256"
read -r -a CANDIDATE_ARRAY <<< "$CANDIDATES"
echo "SOLVER=$SOLVER"
echo "CONFIG=$CONFIG"
echo "CANDIDATES=${CANDIDATE_ARRAY[*]}"
echo "RLAOPT_CUDA_IMAGE_SHA256=$RLAOPT_CUDA_IMAGE_SHA256"

apptainer exec --nv --bind "$REPOSITORY:$REPOSITORY" --pwd "$REPOSITORY" \
  "$DERIVED_IMAGE" /opt/rlaopt-experiments/.venv/bin/rlaopt-bench calibrate \
  --backend cuda --solver "$SOLVER" --config "$CONFIG" \
  --output artifacts/calibration --candidates "${CANDIDATE_ARRAY[@]}"

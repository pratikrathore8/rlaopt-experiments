#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodelist=soal-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5%4
#SBATCH --job-name=rlaopt-erm-cuda-calibrate
#SBATCH --output=cuda-erm-calibrate-%A_%a.out

set -euo pipefail

REPOSITORY="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
CONFIG="${CONFIG:-configs/synthetic_erm_calibration.toml}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/synthetic-erm-calibration}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?submit this script as an array}"

PROBLEM_TYPES=(
  multinomial
  multinomial
  multinomial
  bounded_elastic_net
  bounded_elastic_net
  bounded_elastic_net
)
SOLVERS=(
  rlaopt_sapphire
  projected_gradient
  jaxopt_lbfgsb
  rlaopt_admm
  scs_cuda
  cuclarabel_cudss
)

if [[ ! "$TASK_INDEX" =~ ^[0-9]+$ ]] || (( TASK_INDEX >= ${#SOLVERS[@]} )); then
  echo "Invalid calibration task index: $TASK_INDEX" >&2
  exit 2
fi
if [[ ! -f "$REPOSITORY/$CONFIG" && "$CONFIG" != /* ]]; then
  echo "Calibration configuration not found: $REPOSITORY/$CONFIG" >&2
  exit 2
fi
if [[ "$CONFIG" == /* && ! -f "$CONFIG" ]]; then
  echo "Calibration configuration not found: $CONFIG" >&2
  exit 2
fi

set -a
source "$REPOSITORY/containers/cuda.env"
set +a
DERIVED_IMAGE="$RLAOPT_CUDA_IMAGE"
if [[ "$DERIVED_IMAGE" != /* ]]; then
  DERIVED_IMAGE="$REPOSITORY/$DERIVED_IMAGE"
fi
if [[ ! -f "$DERIVED_IMAGE" ]]; then
  echo "Benchmark image not found: $DERIVED_IMAGE" >&2
  echo "Rebuild the image after committing calibration support." >&2
  exit 2
fi
if [[ ! -f "$DERIVED_IMAGE.sha256" ]]; then
  echo "Benchmark image checksum not found: $DERIVED_IMAGE.sha256" >&2
  exit 2
fi

PROBLEM_TYPE="${PROBLEM_TYPES[$TASK_INDEX]}"
SOLVER="${SOLVERS[$TASK_INDEX]}"
RLAOPT_GIT_COMMIT="$(git -C "$REPOSITORY" rev-parse HEAD)"
if [[ -n "$(git -C "$REPOSITORY" status --porcelain)" ]]; then
  RLAOPT_GIT_DIRTY=1
else
  RLAOPT_GIT_DIRTY=0
fi
RLAOPT_CUDA_IMAGE_SHA256="$(cut -d' ' -f1 "$DERIVED_IMAGE.sha256")"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"
export APPTAINERENV_MKL_NUM_THREADS="$MKL_NUM_THREADS"
export APPTAINERENV_OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export APPTAINERENV_RLAOPT_GIT_COMMIT="$RLAOPT_GIT_COMMIT"
export APPTAINERENV_RLAOPT_GIT_DIRTY="$RLAOPT_GIT_DIRTY"
export APPTAINERENV_RLAOPT_CUDA_IMAGE_SHA256="$RLAOPT_CUDA_IMAGE_SHA256"

echo "HOST=$(hostname)"
echo "TASK_INDEX=$TASK_INDEX"
echo "PROBLEM_TYPE=$PROBLEM_TYPE"
echo "SOLVER=$SOLVER"
echo "CONFIG=$CONFIG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "RLAOPT_GIT_COMMIT=$RLAOPT_GIT_COMMIT"
echo "RLAOPT_GIT_DIRTY=$RLAOPT_GIT_DIRTY"
echo "RLAOPT_CUDA_IMAGE_SHA256=$RLAOPT_CUDA_IMAGE_SHA256"

cd "$REPOSITORY"
apptainer exec --nv --bind "$REPOSITORY:$REPOSITORY" --pwd "$REPOSITORY" \
  "$DERIVED_IMAGE" /opt/rlaopt-experiments/.venv/bin/rlaopt-bench calibrate \
  --config "$CONFIG" \
  --backend cuda \
  --problem-type "$PROBLEM_TYPE" \
  --solver "$SOLVER" \
  --output "$OUTPUT_DIR"

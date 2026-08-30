#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodelist=soal-12
#SBATCH --gres=gpu:h200nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=rlaopt-rapids-check
#SBATCH --output=rapids-check-%j.out

set -euo pipefail

REPOSITORY="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
source "$REPOSITORY/containers/rapids.env"

if [[ ! "$RAPIDS_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "containers/rapids.env does not contain an immutable digest" >&2
  exit 2
fi

export APPTAINER_CACHEDIR="/tmp/$USER/apptainer-cache-$SLURM_JOB_ID"
export APPTAINER_TMPDIR="/tmp/$USER/apptainer-tmp-$SLURM_JOB_ID"
GPU_ENVIRONMENT="/tmp/$USER/rlaopt-gpu-env-$SLURM_JOB_ID"
UVX_BIN="$(command -v uvx)"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
trap 'rm -rf "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$GPU_ENVIRONMENT"' EXIT

RAPIDS_REFERENCE="docker://${RAPIDS_IMAGE%:*}@${RAPIDS_IMAGE_DIGEST}"

echo "=== allocation ==="
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi

echo "=== immutable image ==="
echo "$RAPIDS_REFERENCE"

echo "=== RAPIDS base environment ==="
apptainer exec --nv "$RAPIDS_REFERENCE" python -c '
import platform
import cupy
import cuml

properties = cupy.cuda.runtime.getDeviceProperties(0)
name = properties["name"]
if isinstance(name, bytes):
    name = name.decode()
print("Python:", platform.python_version())
print("cuML:", cuml.__version__)
print("CuPy:", cupy.__version__)
print("CUDA runtime:", cupy.cuda.runtime.runtimeGetVersion())
print("CUDA devices:", cupy.cuda.runtime.getDeviceCount())
print("GPU:", name)
'

echo "=== project uv environment interoperability ==="
apptainer exec --nv --bind "$REPOSITORY:$REPOSITORY" --bind /tmp:/tmp \
  "$RAPIDS_REFERENCE" bash -s -- "$REPOSITORY" "$GPU_ENVIRONMENT" "$UVX_BIN" <<'ENVIRONMENT_SETUP'
set -euo pipefail
REPOSITORY="$1"
GPU_ENVIRONMENT="$2"
UVX_BIN="$3"
cd "$REPOSITORY"

"$UVX_BIN" uv@0.12.7 venv \
  --clear \
  --system-site-packages \
  --python /opt/conda/bin/python \
  "$GPU_ENVIRONMENT"

UV_PROJECT_ENVIRONMENT="$GPU_ENVIRONMENT" \
  "$UVX_BIN" uv@0.12.7 sync --frozen --inexact

"$GPU_ENVIRONMENT/bin/python" scripts/check_gpu_environment.py
ENVIRONMENT_SETUP

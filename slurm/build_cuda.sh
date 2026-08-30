#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --nodelist=soal-12
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=rlaopt-cuda-build
#SBATCH --output=cuda-build-%j.out

set -euo pipefail

REPOSITORY="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
export APPTAINER_CACHEDIR="/tmp/$USER/apptainer-cache-$SLURM_JOB_ID"
export APPTAINER_TMPDIR="/tmp/$USER/apptainer-tmp-$SLURM_JOB_ID"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
trap 'rm -rf "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"' EXIT

"$REPOSITORY/scripts/build_cuda_image.sh"

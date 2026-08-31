#!/usr/bin/env bash
#SBATCH --account=soal
#SBATCH --partition=soal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
set -euo pipefail

REPOSITORY="${SLURM_SUBMIT_DIR:?submit jobs from the repository root}"
MANIFEST="${MANIFEST:?set MANIFEST to a JSONL manifest}"
CHUNK_COUNT="${CHUNK_COUNT:?set CHUNK_COUNT to the submitted array size}"
CHUNK_INDEX="${SLURM_ARRAY_TASK_ID:?submit this script as an array}"

if (( CHUNK_COUNT <= 0 || CHUNK_INDEX < 0 || CHUNK_INDEX >= CHUNK_COUNT )); then
  echo "Invalid chunk $CHUNK_INDEX of $CHUNK_COUNT" >&2
  exit 2
fi

LINE_COUNT="$(wc -l < "$MANIFEST")"
failures=0
attempted=0
for ((manifest_index = CHUNK_INDEX; manifest_index < LINE_COUNT; manifest_index += CHUNK_COUNT)); do
  attempted=$((attempted + 1))
  echo "CHUNK_PROGRESS chunk=$CHUNK_INDEX manifest_index=$manifest_index attempted=$attempted"
  if MANIFEST_INDEX="$manifest_index" bash "$REPOSITORY/slurm/run_array.sh"; then
    :
  else
    status=$?
    failures=$((failures + 1))
    echo "CHUNK_FAILURE chunk=$CHUNK_INDEX manifest_index=$manifest_index status=$status" >&2
  fi
done

echo "CHUNK_SUMMARY chunk=$CHUNK_INDEX attempted=$attempted failures=$failures"
if (( failures > 0 )); then
  exit 1
fi

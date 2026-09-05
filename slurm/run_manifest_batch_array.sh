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
BATCH_LIST="${BATCH_LIST:?set BATCH_LIST to a wave target list}"
BATCH_INDEX="${SLURM_ARRAY_TASK_ID:?submit this script as an array}"
BATCH_MANIFEST="$(sed -n "$((BATCH_INDEX + 1))p" "$BATCH_LIST")"
if [[ -z "$BATCH_MANIFEST" || ! -f "$BATCH_MANIFEST" ]]; then
  echo "No batch manifest at zero-based index $BATCH_INDEX in $BATCH_LIST" >&2
  exit 2
fi
if [[ ! -f "$BATCH_MANIFEST.sha256" ]]; then
  echo "Batch checksum is missing: $BATCH_MANIFEST.sha256" >&2
  exit 2
fi
sha256sum --check --status "$BATCH_MANIFEST.sha256"

LINE_COUNT="$(wc -l < "$BATCH_MANIFEST")"
if (( LINE_COUNT < 1 || LINE_COUNT > 7 )); then
  echo "Refusing batch with $LINE_COUNT jobs; the safe maximum is 7" >&2
  exit 2
fi

failures=0
for ((manifest_index = 0; manifest_index < LINE_COUNT; manifest_index += 1)); do
  echo "BATCH_PROGRESS batch=$BATCH_INDEX manifest_index=$manifest_index total=$LINE_COUNT"
  if MANIFEST="$BATCH_MANIFEST" MANIFEST_INDEX="$manifest_index" \
    bash "$REPOSITORY/slurm/run_array.sh"; then
    :
  else
    status=$?
    failures=$((failures + 1))
    echo "BATCH_FAILURE batch=$BATCH_INDEX manifest_index=$manifest_index status=$status" >&2
  fi
done

echo "BATCH_SUMMARY batch=$BATCH_INDEX attempted=$LINE_COUNT failures=$failures"
if (( failures > 0 )); then
  exit 1
fi

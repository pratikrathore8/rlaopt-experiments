#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 PLAN_DIRECTORY WAVE_INDEX" >&2
  exit 2
fi

REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN_DIRECTORY="$(realpath "$1")"
WAVE_INDEX="$2"
if [[ ! "$WAVE_INDEX" =~ ^[0-9]+$ ]]; then
  echo "WAVE_INDEX must be a nonnegative integer" >&2
  exit 2
fi
printf -v WAVE_NAME 'wave-%03d' "$WAVE_INDEX"
WAVE_DIRECTORY="$PLAN_DIRECTORY/waves/$WAVE_NAME"
CONFIG="$PLAN_DIRECTORY/config.toml"
OUTPUT_DIR="$PLAN_DIRECTORY/results"
MAX_SUBMITTED_JOBS="${MAX_SUBMITTED_JOBS:-20}"

if [[ ! -d "$WAVE_DIRECTORY" || ! -f "$CONFIG" ]]; then
  echo "Production plan or wave is missing: $WAVE_DIRECTORY" >&2
  exit 2
fi
if [[ ! -f "$CONFIG.sha256" ]] || ! sha256sum --check --status "$CONFIG.sha256"; then
  echo "Frozen production configuration checksum validation failed." >&2
  exit 2
fi
if [[ -n "$(git -C "$REPOSITORY" status --porcelain)" ]]; then
  echo "Refusing production submission from a dirty repository." >&2
  exit 2
fi

required=0
for list in "$WAVE_DIRECTORY"/*.txt; do
  count="$(wc -l < "$list")"
  if (( count < 1 || count > 6 )); then
    echo "Invalid task count $count in $list" >&2
    exit 2
  fi
  required=$((required + count))
done
if (( required > 18 )); then
  echo "Refusing wave with $required tasks; the plan maximum is 18" >&2
  exit 2
fi

active_wave_jobs="$(squeue -h -r -u "$USER" -o '%j' | awk '$1 ~ /^real-erm-prod-/ {count++} END {print count+0}')"
if (( active_wave_jobs > 0 )); then
  echo "Refusing submission while $active_wave_jobs real-data production tasks are active." >&2
  exit 2
fi
current_jobs="$(squeue -h -r -u "$USER" -o '%i' | wc -l)"
if (( current_jobs + required > MAX_SUBMITTED_JOBS )); then
  echo "Need $required slots but only $((MAX_SUBMITTED_JOBS - current_jobs)) are available." >&2
  exit 2
fi

submit_target() {
  local target="$1"
  local backend="$2"
  local node="$3"
  local list="$WAVE_DIRECTORY/$target.txt"
  [[ -f "$list" ]] || return 0
  local count
  count="$(wc -l < "$list")"
  local args=(
    --array="0-$((count - 1))%1"
    --nodelist="$node"
    --job-name="real-erm-prod-$WAVE_NAME-$target"
    --output="$PLAN_DIRECTORY/logs/$WAVE_NAME-$target-%A_%a.out"
    --export="ALL,BACKEND=$backend,BATCH_LIST=$list,CONFIG=$CONFIG,OUTPUT_DIR=$OUTPUT_DIR"
  )
  if [[ "$backend" == "cuda" ]]; then
    args+=(--gres=gpu:h200nvl:1)
  fi
  sbatch "${args[@]}" "$REPOSITORY/slurm/run_manifest_batch_array.sh"
}

mkdir -p "$PLAN_DIRECTORY/logs" "$OUTPUT_DIR"
submit_target cpu-soal-8 cpu soal-8
submit_target cpu-soal-9 cpu soal-9
submit_target cuda-soal-12 cuda soal-12

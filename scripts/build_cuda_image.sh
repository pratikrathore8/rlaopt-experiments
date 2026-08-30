#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPOSITORY/containers/cuda.env"

if [[ ! "$CUDA_BASE_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "containers/cuda.env does not contain an immutable digest" >&2
  exit 2
fi

OUTPUT="${1:-$REPOSITORY/$RLAOPT_CUDA_IMAGE}"
BUILD_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/rlaopt-apptainer-build.XXXXXX")"
trap 'rm -rf "$BUILD_DIRECTORY"' EXIT

CUDA_IMAGE_REFERENCE="${CUDA_BASE_IMAGE%:*}@${CUDA_BASE_IMAGE_DIGEST}"
sed \
  -e "s|@CUDA_IMAGE_REFERENCE@|$CUDA_IMAGE_REFERENCE|g" \
  -e "s|@CUDA_BASE_IMAGE_DIGEST@|$CUDA_BASE_IMAGE_DIGEST|g" \
  -e "s|@CUDA_IMAGE_VERSION@|$CUDA_IMAGE_VERSION|g" \
  "$REPOSITORY/containers/cuda.def.in" > "$BUILD_DIRECTORY/cuda.def"

mkdir -p "$(dirname "$OUTPUT")"
cd "$REPOSITORY"
apptainer build --force --fakeroot "$OUTPUT" "$BUILD_DIRECTORY/cuda.def"
apptainer inspect "$OUTPUT"
IMAGE_SHA256="$(sha256sum "$OUTPUT" | cut -d' ' -f1)"
printf '%s  %s\n' "$IMAGE_SHA256" "$(basename "$OUTPUT")" > "$OUTPUT.sha256"
cat "$OUTPUT.sha256"
echo "Built $OUTPUT"

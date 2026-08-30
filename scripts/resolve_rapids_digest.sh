#!/usr/bin/env bash
set -euo pipefail

source "${1:-containers/rapids.env}"
if ! command -v skopeo >/dev/null 2>&1; then
  echo "skopeo is required to query the registry without pulling the image" >&2
  exit 1
fi
DIGEST="$(skopeo inspect --format '{{.Digest}}' "docker://${RAPIDS_IMAGE}")"
echo "RAPIDS_IMAGE_DIGEST=${DIGEST}"
echo "Immutable reference: ${RAPIDS_IMAGE%:*}@${DIGEST}"

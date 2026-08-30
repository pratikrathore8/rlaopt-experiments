#!/usr/bin/env bash
set -euo pipefail

source "${1:-containers/rapids.env}"
if command -v skopeo >/dev/null 2>&1; then
  DIGEST="$(skopeo inspect --override-os linux --override-arch amd64 \
    --format '{{.Digest}}' "docker://${RAPIDS_IMAGE}")"
elif command -v docker >/dev/null 2>&1; then
  DIGEST="$(docker manifest inspect --verbose "$RAPIDS_IMAGE" | \
    python3 -c 'import json,sys; data=json.load(sys.stdin); print(next(item["Descriptor"]["digest"] for item in data if item["Descriptor"]["platform"] == {"architecture": "amd64", "os": "linux"}))')"
else
  echo "skopeo or docker is required for a read-only registry query" >&2
  exit 1
fi
echo "RAPIDS_IMAGE_DIGEST=${DIGEST}"
echo "Immutable reference: ${RAPIDS_IMAGE%:*}@${DIGEST}"

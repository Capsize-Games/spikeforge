#!/usr/bin/env bash
# Build dependencies on the deploy runner, outside the production host.
set -euo pipefail
cd "$(dirname "$0")/.."
revision="${GITHUB_SHA:?GITHUB_SHA is required}"
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]] || \
   [[ "$(git rev-parse HEAD)" != "$revision" ]]; then
  echo "Deployment revision does not match the checked-out source." >&2
  exit 1
fi
docker build \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  --label "org.opencontainers.image.revision=$revision" \
  --tag "spikeforge-dashboard:$revision" .
image_id="$(docker image inspect --format '{{.Id}}' "spikeforge-dashboard:$revision")"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf 'image_id=%s\n' "$image_id" >> "$GITHUB_OUTPUT"
fi

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
  --build-arg LOCKFILE=requirements/server.cpu.lock \
  --label "org.opencontainers.image.revision=$revision" \
  --tag "spikeforge-dashboard:$revision" .
config_digest="$(docker image save "spikeforge-dashboard:$revision" | \
  python3 scripts/dashboard-image-digest.py)"
echo "Built dashboard configuration $config_digest for $revision."
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf 'config_digest=%s\n' "$config_digest" >> "$GITHUB_OUTPUT"
fi

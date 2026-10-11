#!/usr/bin/env bash
# Start only a preloaded dashboard image; no production dependency build.
set -euo pipefail
revision="${1:?Deployment revision is required}"
expected="${2:?Builder image ID is required}"
directory="${3:-/opt/spikeforge}"
project="${4:-spikeforge}"
helper="$(cd "$(dirname "$0")" && pwd)/check-dashboard-revision.sh"
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]] || \
   [[ ! "$expected" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Invalid deployment revision or builder image ID." >&2
  exit 1
fi
export SPIKEFORGE_DASHBOARD_IMAGE="spikeforge-dashboard:$revision"
if ! actual="$(docker image inspect --format \
  '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
  "$SPIKEFORGE_DASHBOARD_IMAGE" 2>/dev/null)"; then
  echo "Required dashboard image is not loaded: $SPIKEFORGE_DASHBOARD_IMAGE" >&2
  exit 1
fi
if [[ "$actual" != "$revision" ]]; then
  echo "Dashboard image revision mismatch: expected $revision, found $actual." >&2
  exit 1
fi
loaded="$(docker image inspect --format '{{.Id}}' "$SPIKEFORGE_DASHBOARD_IMAGE")"
if [[ "$loaded" != "$expected" ]]; then
  echo "Loaded dashboard image ID mismatch: expected $expected, found $loaded." >&2
  exit 1
fi
cd "$directory"
compose=(docker compose -p "$project" --env-file .env \
  -f docker-compose.hetzner.yml)
"${compose[@]}" up -d --no-build --pull never --wait --wait-timeout 180 \
  --remove-orphans spikeforge-dashboard
container="$("${compose[@]}" ps -q spikeforge-dashboard)"
test -n "$container"
bash "$helper" "$revision" "$container" "$expected"

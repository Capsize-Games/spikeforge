#!/usr/bin/env bash
# Verify the running container rather than trusting an image tag alone.
set -euo pipefail
revision="${1:?Deployment revision is required}"
container="${2:?Container ID is required}"
expected="${3:?Verified local image ID is required}"
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]] || \
   [[ ! "$expected" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Invalid deployment revision or verified local image ID." >&2
  exit 1
fi
label='{{index .Config.Labels "org.opencontainers.image.revision"}}'
actual="$(docker inspect --format "$label" "$container")"
if [[ "$actual" != "$revision" ]]; then
  echo "Dashboard revision mismatch: expected $revision, found $actual." >&2
  exit 1
fi
running="$(docker inspect --format '{{.Image}}' "$container")"
if [[ "$running" != "$expected" ]]; then
  echo "Dashboard image ID mismatch: expected $expected, found $running." >&2
  exit 1
fi
echo "Dashboard revision $revision is running image $running."

#!/usr/bin/env bash
# A queued old revision must not replace the current main deployment.
set -euo pipefail
revision="${GITHUB_SHA:?GITHUB_SHA is required}"
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]] || \
   [[ "$(git rev-parse HEAD)" != "$revision" ]]; then
  echo "Deployment revision does not match the checked-out source." >&2
  exit 1
fi
current="$(git ls-remote --exit-code origin refs/heads/main | cut -f1)"
if [[ "$revision" != "$current" ]]; then
  echo "Refusing superseded dashboard deployment revision $revision." >&2
  exit 1
fi

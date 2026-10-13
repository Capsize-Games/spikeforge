#!/usr/bin/env bash
#
# Deploy the dashboard to dash.spikeforge.net over SSH, without waiting on CI.
#
# `.github/workflows/deploy-hetzner.yml` does the same three things, but a push
# to main also starts the seventeen-job CI suite (Python matrices, extras
# builds that pull Torch), and the deploy queues behind it on the same three
# runners. For a client-only change that is minutes of work behind hours of
# queue — the mirror refresh took about two hours that way and about fifteen
# minutes this way.
#
# The steps are the workflow's own, so it needs the same access: an SSH host
# that can reach the deployment machine as a user in the docker group, or root.
# Point it with HETZNER_HOST (default: the `hetzner-airunner` ssh alias).
#
#   scripts/deploy_dashboard_local.sh
#
set -euo pipefail

HOST="${HETZNER_HOST:-hetzner-airunner}"
REMOTE_DIR="${SPIKEFORGE_REMOTE_DIR:-/opt/spikeforge}"
COMPOSE="docker compose -p spikeforge --env-file .env -f docker-compose.hetzner.yml"

cd "$(dirname "$0")/.."

# `build/` is generated documentation output: megabytes of it, gigabytes with
# its datasets, and the deployment never reads it. Shipping it once filled the
# deployment host's disk and failed the sync halfway.
echo "==> syncing to ${HOST}:${REMOTE_DIR}"
rsync -rl --delete \
  --exclude='.git' \
  --exclude='.env' \
  --exclude='venv' \
  --exclude='build' \
  --exclude='client/node_modules' \
  --exclude='client/dist' \
  --exclude='**/__pycache__' \
  --exclude='*.egg-info' \
  -e "ssh -o BatchMode=yes" \
  ./ "${HOST}:${REMOTE_DIR}/"

# The image build writes several gigabytes of layers, and the host runs other
# services, so the disk is the thing that fails first. Reclaim only what no
# container uses.
echo "==> reclaiming unused images and build cache"
ssh -o BatchMode=yes "$HOST" \
  "docker image prune -a -f >/dev/null; docker builder prune -a -f >/dev/null; df -h / | tail -1"

echo "==> building and restarting"
ssh -o BatchMode=yes "$HOST" \
  "cd ${REMOTE_DIR} && ${COMPOSE} up -d --build --remove-orphans"

echo "==> health"
ssh -o BatchMode=yes "$HOST" \
  "docker inspect --format '{{.State.Health.Status}}' spikeforge-dashboard"

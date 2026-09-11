#!/usr/bin/env bash
#
# Build and run the snn-interpreter server from the Docker container.
#
# This is the one-shot "get the dashboard up" entry point: it selects the
# CPU/GPU image profile, builds it, starts the container, waits for the
# /health endpoint, and (by default) follows the logs. The container serves
# the React dashboard, the FastAPI/WebSocket API, and everything else on a
# single port (8877 by default).
#
# Usage:
#   scripts/docker_server.sh [options]
#
# Options:
#   --cpu             use the CPU-only torch image (`--profile cpu`)
#   --gpu             use the explicit CUDA torch image (`--profile gpu`)
#   --default         use the default service (CUDA build); this is the default
#   --port N          host port to publish (default: 8877 / $SNN_HOST_PORT)
#   --no-build        start without rebuilding the image
#   -d, --detach      run the container in the background and return
#   --follow          with -d, tail the logs after the health check
#   --no-wait         skip waiting for the /health endpoint
#   -h, --help        show this message
#
# Examples:
#   scripts/docker_server.sh                 # build + run CUDA image, follow logs
#   scripts/docker_server.sh --cpu           # build + run the CPU-only image
#   scripts/docker_server.sh -d --follow     # run detached, then tail logs
#   scripts/docker_server.sh --no-build      # restart the existing image
#   scripts/docker_server.sh --port 9000     # serve the dashboard on :9000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SERVICE="snn-interpreter"
PROFILE=""
BUILD=1
DETACH=0
FOLLOW=0
WAIT=1
HOST_PORT="${SNN_HOST_PORT:-8877}"
HEALTH_TIMEOUT="${SNN_HEALTH_TIMEOUT:-180}"

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

docker_compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    die "docker compose is not available"
  fi
}

usage() {
  cat <<'EOF'
Usage: scripts/docker_server.sh [options]

Build and run the snn-interpreter server from the Docker container and serve
the dashboard on a single port (default 8877).

Options:
  --cpu             use the CPU-only torch image (--profile cpu)
  --gpu             use the explicit CUDA torch image (--profile gpu)
  --default         use the default service (CUDA build); this is the default
  --port N          host port to publish (default: 8877 / $SNN_HOST_PORT)
  --no-build        start without rebuilding the image
  -d, --detach      run the container in the background and return
  --follow          with -d, tail the logs after the health check
  --no-wait         skip waiting for the /health endpoint
  -h, --help        show this message

Examples:
  scripts/docker_server.sh                 # build + run CUDA image, follow logs
  scripts/docker_server.sh --cpu           # build + run the CPU-only image
  scripts/docker_server.sh -d --follow     # run detached, then tail logs
  scripts/docker_server.sh --no-build      # restart the existing image
  scripts/docker_server.sh --port 9000     # serve the dashboard on :9000
EOF
}

# --- argument parsing --------------------------------------------------------

while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) PROFILE="cpu"; SERVICE="snn-interpreter-cpu" ;;
    --gpu) PROFILE="gpu"; SERVICE="snn-interpreter-gpu" ;;
    --default) PROFILE=""; SERVICE="snn-interpreter" ;;
    --port)
      [ $# -ge 2 ] || die "--port requires a value"
      HOST_PORT="$2"; shift
      ;;
    --port=*) HOST_PORT="${1#*=}" ;;
    --no-build) BUILD=0 ;;
    -d | --detach) DETACH=1 ;;
    --follow) FOLLOW=1 ;;
    --no-wait) WAIT=0 ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"

case "$HOST_PORT" in
  '' | *[!0-9]*) die "--port must be a number, got '$HOST_PORT'" ;;
esac

# --- compose argument list ---------------------------------------------------

COMPOSE_ARGS=()
if [ -n "$PROFILE" ]; then
  COMPOSE_ARGS+=(--profile "$PROFILE")
fi

# The compose files publish ${SNN_HOST_PORT}:8877; export the chosen port.
export SNN_HOST_PORT="$HOST_PORT"

# --- run ---------------------------------------------------------------------

URL="http://localhost:${HOST_PORT}"

if [ "$BUILD" -eq 1 ]; then
  info "Building the server image (service: $SERVICE)"
  docker_compose "${COMPOSE_ARGS[@]}" build "$SERVICE"
else
  info "Skipping build (reusing the existing image)"
fi

if [ "$DETACH" -eq 1 ]; then
  info "Starting the server container in the background"
  docker_compose "${COMPOSE_ARGS[@]}" up -d "$SERVICE"
else
  info "Starting the server container (Ctrl-C stops it)"
fi

if [ "$WAIT" -eq 1 ]; then
  info "Waiting for the server to become healthy at ${URL}/health"
  waited=0
  until curl -fsS "${URL}/health" >/dev/null 2>&1; do
    if [ "$waited" -ge "$HEALTH_TIMEOUT" ]; then
      warn "server did not report healthy within ${HEALTH_TIMEOUT}s"
      warn "recent logs:"
      docker_compose "${COMPOSE_ARGS[@]}" logs --tail=40 "$SERVICE" || true
      exit 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  info "Server is healthy: ${URL}"
fi

if [ "$DETACH" -eq 1 ]; then
  info "Dashboard: ${URL}"
  info "Stop with: docker compose down   (or: docker compose stop $SERVICE)"
  if [ "$FOLLOW" -eq 1 ]; then
    info "Following logs (Ctrl-C stops following; the container keeps running)"
    docker_compose "${COMPOSE_ARGS[@]}" logs -f "$SERVICE"
  fi
else
  info "Dashboard: ${URL} (streaming logs; Ctrl-C stops the container)"
fi

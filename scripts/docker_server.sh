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

# Always start detached. This returns as soon as the container is created, so
# the health wait always applies to a real, running container. (A blocking
# foreground `up` relative to the health wait is what caused the hang: the
# wait could run while no container was actually attached.)
info "Starting the server container"
docker_compose "${COMPOSE_ARGS[@]}" up -d "$SERVICE"

container_state() {
  docker inspect -f '{{.State.Status}}' "$SERVICE" 2>/dev/null || echo "missing"
}

wait_for_health() {
  info "Waiting for the server to become healthy at ${URL}/health"
  local waited=0 state
  while :; do
    if curl -fsS "${URL}/health" >/dev/null 2>&1; then
      info "Server is healthy: ${URL}"
      return 0
    fi
    state="$(container_state)"
    if [ "$state" != "running" ]; then
      warn "the '$SERVICE' container is not running (state: $state)"
      warn "recent logs:"
      docker_compose "${COMPOSE_ARGS[@]}" logs --tail=50 "$SERVICE" || true
      return 1
    fi
    if [ "$waited" -ge "$HEALTH_TIMEOUT" ]; then
      warn "server did not report healthy within ${HEALTH_TIMEOUT}s"
      docker_compose "${COMPOSE_ARGS[@]}" ps || true
      warn "recent logs:"
      docker_compose "${COMPOSE_ARGS[@]}" logs --tail=50 "$SERVICE" || true
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

if [ "$WAIT" -eq 1 ]; then
  wait_for_health || exit 1
fi

info "Dashboard: ${URL}"

if [ "$DETACH" -eq 1 ]; then
  info "Stop with: docker compose down   (or: docker compose stop $SERVICE)"
  if [ "$FOLLOW" -eq 1 ]; then
    info "Following logs (Ctrl-C stops following; the container keeps running)"
    docker_compose "${COMPOSE_ARGS[@]}" logs -f "$SERVICE"
  fi
  exit 0
fi

# Foreground: stream logs and stop the container when the user hits Ctrl-C.
info "Streaming logs; Ctrl-C stops the container"
stop_container() {
  info "Stopping the server container"
  docker_compose "${COMPOSE_ARGS[@]}" stop "$SERVICE" >/dev/null 2>&1 || true
}
trap stop_container INT TERM EXIT
docker_compose "${COMPOSE_ARGS[@]}" logs -f "$SERVICE"

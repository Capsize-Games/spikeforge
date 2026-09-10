#!/usr/bin/env bash
#
# Developer utility for the snn-interpreter project.
#
# One entry point for the tasks an engineer or AI agent reaches for most:
# environment setup, linting/tests, running the servers, inspecting and
# clearing the dataset cache, and Docker. Run `scripts/dev.sh help` for the
# full command list.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLIENT="$ROOT/client"
VENV="$ROOT/venv"
DATA_DIR="${SNN_DATA_DIR:-$ROOT/build}"
SERVICE="snn-interpreter"
DATASETS=(
  MNIST FashionMNIST KMNIST QMNIST USPS EMNIST cifar-10-batches-py mnist
)

# --- helpers -----------------------------------------------------------------

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# Prefer the project venv, falling back to whatever is on PATH.
venv_tool() {
  local name="$1"; shift
  if [ -x "$VENV/bin/$name" ]; then "$VENV/bin/$name" "$@"; else "$name" "$@"; fi
}

py() {
  if [ -x "$VENV/bin/python" ]; then "$VENV/bin/python" "$@"; else python "$@"; fi
}

docker_compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    die "docker compose is not available"
  fi
}

require_client_deps() {
  [ -d "$CLIENT/node_modules" ] || die "run 'scripts/dev.sh client-install' first"
}

# --- setup -------------------------------------------------------------------

cmd_setup() {
  info "Installing Python package with dev + web extras"
  py -m pip install -e ".[dev,web]"
  info "Installing client dependencies"
  (cd "$CLIENT" && npm install)
}

# --- python quality ----------------------------------------------------------

cmd_lint() {
  info "ruff check"
  venv_tool ruff check snn_interpreter server main.py main_encodings.py tests
}

cmd_test() {
  info "pytest"
  venv_tool pytest "$@"
}

# --- client ------------------------------------------------------------------

cmd_client_install() {
  info "Installing client dependencies"
  (cd "$CLIENT" && npm install)
}

cmd_client_typecheck() {
  require_client_deps
  info "tsc -b"
  (cd "$CLIENT" && node_modules/.bin/tsc -b --pretty false)
}

cmd_client_build() {
  require_client_deps
  info "vite build"
  (cd "$CLIENT" && npm run build)
}

cmd_client_dev() {
  require_client_deps
  info "Starting Vite dev server on :5173"
  (cd "$CLIENT" && npm run dev)
}

# --- servers -----------------------------------------------------------------

cmd_server() {
  info "Starting FastAPI server on :8877"
  py -m server "$@"
}

cmd_health() {
  curl -fsS http://127.0.0.1:8877/health && echo
}

cmd_dev() {
  info "Starting FastAPI (:8877) and Vite (:5173) — Ctrl-C stops both"
  py -m server &
  local server_pid=$!
  (cd "$CLIENT" && npm run dev) &
  local client_pid=$!
  # shellcheck disable=SC2064
  trap "kill $server_pid $client_pid 2>/dev/null || true" INT TERM EXIT
  wait || true
}

# --- data cache --------------------------------------------------------------

cmd_data() {
  info "Data directory: $DATA_DIR"
  if [ -d "$DATA_DIR" ]; then
    du -sh "$DATA_DIR"/* 2>/dev/null | sort -h || true
  else
    echo "(not created yet)"
  fi
}

cmd_data_dir() {
  echo "$DATA_DIR"
}

# Remove dataset caches but keep models/ and exported artifacts.
cmd_data_clear() {
  local target="${1:-all}"
  if [ "$target" = "all" ]; then
    info "Removing dataset caches under $DATA_DIR"
    local name
    for name in "${DATASETS[@]}"; do rm -rf "${DATA_DIR:?}/$name"; done
  else
    info "Removing dataset cache: $target"
    rm -rf "${DATA_DIR:?}/$target"
  fi
  cmd_data
}

cmd_data_nuke() {
  warn "Removing the entire data directory (datasets AND saved models): $DATA_DIR"
  rm -rf "${DATA_DIR:?}"
}

# --- docker ------------------------------------------------------------------

cmd_docker_up() { docker_compose up --build; }
cmd_docker_down() { docker_compose down; }
cmd_docker_build() { docker_compose build "$@"; }
cmd_docker_logs() { docker_compose logs -f "$@"; }
cmd_docker_shell() { docker_compose exec "$SERVICE" bash; }

cmd_docker_data_clear() {
  local target="${1:-all}"
  if [ "$target" = "all" ]; then
    info "Removing dataset caches inside the container"
    docker_compose exec "$SERVICE" sh -c \
      "rm -rf /data/MNIST /data/FashionMNIST /data/KMNIST /data/QMNIST \
       /data/USPS /data/EMNIST /data/cifar-10-batches-py /data/mnist"
  else
    info "Removing dataset cache in container: $target"
    docker_compose exec "$SERVICE" rm -rf "/data/$target"
  fi
}

cmd_docker_reset() {
  warn "This deletes the snn-data volume (datasets AND saved models)"
  docker_compose down -v
  docker_compose up --build
}

# --- aggregate ---------------------------------------------------------------

cmd_check() {
  cmd_lint
  cmd_test
  cmd_client_typecheck
  cmd_client_build
}

cmd_help() {
  cat <<'EOF'
Usage: scripts/dev.sh <command> [args]

Setup
  setup                     install Python (dev,web extras) + client deps

Quality
  lint                      ruff check the Python package and server
  test [pytest args]        run the test suite
  check                     lint + test + client type-check + client build

Servers
  server [args]             run the FastAPI server (:8877)
  client-dev                run the Vite dev server (:5173)
  dev                       run the API and Vite dev server together
  health                    curl the server /health endpoint

Client
  client-install            npm install in client/
  client-typecheck          tsc -b (no emit)
  client-build              npm run build

Data cache
  data                      show the data dir and per-dataset sizes
  data-dir                  print the resolved data directory
  data-clear [name]         remove all dataset caches, or one by name
  data-nuke                 remove the whole data dir (datasets + models)

Docker
  docker-up                 docker compose up --build
  docker-down               docker compose down
  docker-build [args]       docker compose build
  docker-logs [args]        docker compose logs -f
  docker-shell              bash inside the running container
  docker-data-clear [name]  clear dataset caches inside the container
  docker-reset              down -v then up --build (wipes the volume)

  help                      show this message

Dataset names: MNIST FashionMNIST KMNIST QMNIST USPS EMNIST \
cifar-10-batches-py mnist
EOF
}

# --- dispatch ----------------------------------------------------------------

main() {
  local command="${1:-help}"
  shift || true
  case "$command" in
    setup) cmd_setup ;;
    lint) cmd_lint ;;
    test) cmd_test "$@" ;;
    check) cmd_check ;;
    server) cmd_server "$@" ;;
    client-dev) cmd_client_dev ;;
    dev) cmd_dev ;;
    health) cmd_health ;;
    client-install) cmd_client_install ;;
    client-typecheck) cmd_client_typecheck ;;
    client-build) cmd_client_build ;;
    data) cmd_data ;;
    data-dir) cmd_data_dir ;;
    data-clear) cmd_data_clear "$@" ;;
    data-nuke) cmd_data_nuke ;;
    docker-up) cmd_docker_up ;;
    docker-down) cmd_docker_down ;;
    docker-build) cmd_docker_build "$@" ;;
    docker-logs) cmd_docker_logs "$@" ;;
    docker-shell) cmd_docker_shell ;;
    docker-data-clear) cmd_docker_data_clear "$@" ;;
    docker-reset) cmd_docker_reset ;;
    help | --help | -h) cmd_help ;;
    *) warn "unknown command: $command"; echo; cmd_help; exit 1 ;;
  esac
}

main "$@"

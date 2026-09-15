#!/usr/bin/env bash
#
# One-command editable install of the spikeforge workspace.
#
# Installs all four distributions from this checkout in editable mode:
#   spikeforge, spikeforge-targets, spikeforge-hub, spikeforge-server
#
# Usage:
#   ./install.sh [--no-server] [--dev] [--user]
#
#   --no-server   skip the spikeforge-server distribution
#   --dev         add the [dev] extra to every distribution that declares one
#   --user        pass --user through to pip (install into the user site)
#   -h, --help    show this help
#
# Environment:
#   PYTHON    interpreter to use (default: python)
#   PIP_ARGS  extra arguments passed through to `pip install`, word-split, for
#             example: PIP_ARGS="--no-deps --no-build-isolation" ./install.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

WITH_SERVER=1
WITH_DEV=0
PIP_USER=0

usage() {
  cat <<'EOF'
Install spikeforge (editable) from this checkout.

Usage:
  ./install.sh [--no-server] [--dev] [--user]

Options:
  --no-server   skip the spikeforge-server distribution
  --dev         add the [dev] extra to every distribution that declares one
  --user        pass --user through to pip (install into the user site)
  -h, --help    show this help

Environment:
  PYTHON        interpreter to use (default: python)
  PIP_ARGS      extra arguments passed through to `pip install`, word-split
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-server) WITH_SERVER=0 ;;
    --dev) WITH_DEV=1 ;;
    --user) PIP_USER=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      printf 'install.sh: unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

# Assemble the pip command once. General options (including the word-split
# PIP_ARGS passthrough) go *before* `-e`, so pip never mistakes them for the
# editable requirement: PIP_ARGS="--no-deps" ./install.sh
PIP_CMD=("$PYTHON" -m pip install)
if [ "$PIP_USER" = "1" ]; then
  PIP_CMD+=("--user")
fi
if [ -n "${PIP_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  PIP_CMD+=(${PIP_ARGS})
fi
PIP_CMD+=(-e)

spec() {
  # Echo the editable target for a package, adding [dev] where supported so
  # `--dev` never points pip at an extra a distribution does not declare.
  local dir="$1"
  if [ "$WITH_DEV" = "1" ] \
    && grep -qE '^dev[[:space:]]*=' "$dir/pyproject.toml"; then
    printf '%s[dev]' "$dir"
  else
    printf '%s' "$dir"
  fi
}

targets=()
targets+=("$(spec ./packages/spikeforge)")
targets+=("$(spec ./packages/spikeforge-targets)")
targets+=("$(spec ./packages/spikeforge-hub)")
if [ "$WITH_SERVER" = "1" ]; then
  # spikeforge-serve comes first and is not optional here: the server pins it,
  # and leaving it out makes pip resolve that pin from PyPI rather than this
  # checkout -- quietly, with no error, mixing published code into a local
  # install.
  targets+=("$(spec ./packages/spikeforge-serve)")
  targets+=("$(spec ./packages/spikeforge-server)")
fi

printf '\033[1;34m==>\033[0m Installing spikeforge from %s\n' "$ROOT"
printf '    %s\n' "${targets[@]}"
printf '\033[1;34m==>\033[0m %s %s\n' "${PIP_CMD[*]}" "${targets[*]}"
"${PIP_CMD[@]}" "${targets[@]}"

printf '\033[1;32m==>\033[0m installed; run '\''spikeforge --help'\'' or '\''spikeforge-server'\''\n'

#!/usr/bin/env bash
#
# Build the documentation site from the authoritative design documents.
#
# plans/ (plus README.md) is the single source of truth: the docs/ tree is
# regenerated on every build and is never edited by hand. With --check the
# build runs `mkdocs build --strict`, so a broken documentation link fails.
#
# Usage:
#   scripts/build_docs.sh          build the site into build/docs
#   scripts/build_docs.sh --check  same, but fail on broken documentation links
#
# Requires the docs extra:  pip install -e ".[docs]"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DOCS="$ROOT/docs"
CHECK=0
if [ "${1:-}" = "--check" ]; then
  CHECK=1
fi

# Prefer the project venv (matching scripts/dev.sh), else whatever is on PATH.
PY="$ROOT/venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python"
fi

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

info "Generating docs/ from plans/ and README.md"
rm -rf "$DOCS"
mkdir -p "$DOCS"
cp plans/*.md "$DOCS"/
# Lowercase so it does not collide with MkDocs' README.md -> index.md alias.
cp README.md "$DOCS/readme.md"
# Root documents that plan pages link to as sibling pages, plus the
# user-facing cookbook, the examples index, and the readiness checklist.
cp rules.md INTEGRATION_PLAN.md COOKBOOK.md OPEN_SOURCE_CHECKLIST.md "$DOCS"/
cp examples/README.md "$DOCS/examples.md"
# Plan files reference one another as plans/<name>.md and other pages as
# ../<file>, which is correct from the repo root; inside the flat docs/ tree
# those prefixes are dropped, the examples index becomes a sibling page, and
# README's self-references follow its lowercased file name.
sed -i \
  -e 's|](../examples/README.md)|](examples.md)|g' \
  -e 's|](examples/README.md)|](examples.md)|g' \
  -e 's|](examples/)|](examples.md)|g' \
  -e 's|](../|](|g' \
  -e 's|](plans/|](|g' \
  -e 's|](README.md|](readme.md|g' \
  "$DOCS"/*.md

if [ "$CHECK" = "1" ]; then
  info "mkdocs build --strict"
  "$PY" -m mkdocs build --strict
  # MkDocs itself tolerates the plans' repo-relative source references;
  # this check still fails on a broken link between documentation pages.
  info "checking documentation links"
  "$PY" "$ROOT/scripts/check_docs_links.py" "$DOCS"
else
  info "mkdocs build"
  "$PY" -m mkdocs build
fi

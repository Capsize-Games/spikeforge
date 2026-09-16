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

info "Generating docs/ from plans/, documentation/ and README.md"
rm -rf "$DOCS"
mkdir -p "$DOCS"
cp plans/*.md "$DOCS"/
# Lowercase so it does not collide with MkDocs' README.md -> index.md alias.
cp README.md "$DOCS/readme.md"
# Root documents that plan pages link to as sibling pages, plus the
# user-facing cookbook, the examples index, and the readiness checklist.
cp rules.md COPY_POLICY.md INTEGRATION_PLAN.md COOKBOOK.md \
   OPEN_SOURCE_CHECKLIST.md "$DOCS"/
cp examples/README.md "$DOCS/examples.md"
# Long-form reference pages (see documentation/README.md). The documentation
# index is copied as documentation.md so it cannot collide with the site home
# (index.md) or the lowercased README alias (readme.md).
cp documentation/*.md "$DOCS"/
mv "$DOCS/README.md" "$DOCS/documentation.md"
# Plan files reference one another as plans/<name>.md and other pages as
# ../<file>, which is correct from the repo root; inside the flat docs/ tree
# those prefixes are dropped, the examples index becomes a sibling page, the
# documentation index gets its non-colliding name, and README's self-references
# follow its lowercased file name.
sed -i \
  -e 's|](../examples/README.md)|](examples.md)|g' \
  -e 's|](examples/README.md)|](examples.md)|g' \
  -e 's|](examples/)|](examples.md)|g' \
  -e 's|](documentation/README.md)|](documentation.md)|g' \
  -e 's|](documentation/|](|g' \
  -e 's|](../|](|g' \
  -e 's|](plans/|](|g' \
  -e 's|](README.md|](readme.md|g' \
  "$DOCS"/*.md

# The API reference is generated into docs/ only, never into documentation/:
# mkdocstrings' `:::` directives are meaningless anywhere but a MkDocs build,
# and documentation/ is also the source for the published GitHub Wiki, where
# they would render as literal text.
info "Generating the API reference page"
cat > "$DOCS/api-reference.md" <<'MARKDOWN'
# API reference

Generated from the shipped docstrings and type hints, so the signatures here
cannot drift from the code. This covers the **public** surface of each
distribution -- the names exported from each import root, plus the modules a
caller integrating spikeforge into their own code reaches for directly.
Private names (leading underscore) are omitted.

For narrative documentation start at [Quickstart](quickstart.md) and
[Architecture](architecture.md); this page is the lookup table.

## spikeforge

`TrainingEngine` is the entry point for new code: it owns the cancellable
training loop, topology selection, encoding, checkpointing, and evaluation.
`SNNTrainer` is the older, narrower MNIST rate-coding helper behind the
tutorial demo.

::: spikeforge
    options:
      members:
        - TrainingEngine
        - SNNTrainer
        - SNNTrainerLogger
        - LatencyTrainer
        - DeltaTrainer
        - RandomSpikeGenerator
        - convert_to_time

### Version and compatibility reporting

::: spikeforge.version

### Topologies

::: spikeforge.topology.registry

### Encoding

::: spikeforge.encoding.spike_encoder

### Datasets

::: spikeforge.data.datasets

::: spikeforge.data.data_loader

### NIR bridge

::: spikeforge.nir_bridge

## spikeforge_targets

::: spikeforge_targets

## spikeforge_hub

::: spikeforge_hub.catalog

::: spikeforge_hub.entry
    options:
      members:
        - HubEntry

::: spikeforge_hub.inspect

## spikeforge_io

::: spikeforge_io
MARKDOWN

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

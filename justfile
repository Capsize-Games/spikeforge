set shell := ["bash", "-euo", "pipefail", "-c"]

default: ci

setup:
    bash scripts/dev.sh setup

lint:
    bash scripts/dev.sh lint

test:
    bash scripts/dev.sh test

typecheck:
    bash scripts/dev.sh client-typecheck

build:
    bash scripts/dev.sh client-build

run:
    bash scripts/dev.sh dev

docs:
    @echo "See AGENTS.md and documentation/development.md."

ci:
    bash scripts/dev.sh check

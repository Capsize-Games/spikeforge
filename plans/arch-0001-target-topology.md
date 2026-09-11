# ARCH-0001 target topology

**Status: accepted — implemented.**
**Date:** 2026-09-11 · **Issue:** ARCH-0001 *Phased repo split: core library, deploy targets, dashboard*
**Owner:** Capsize Games (maintainer) · **Depends on:** [`plans/arch-0001-adr-repo-topology.md`](plans/arch-0001-adr-repo-topology.md)

## Decision

Adopt the following distribution-to-import-root mapping. **No two distributions
ship into the same top-level namespace.** The plan's PEP 420 namespace-package
escape hatch (`plans/repo_topology_plan.md` §4) is deliberately **not** used.

| Distribution | Import root | Source of truth today | Owner phase |
|---|---|---|---|
| `spikeforge` | `spikeforge` | `spikeforge/` (minus any extracted subpackages) | Phase 1 |
| `spikeforge-server` | `server` (kept) | `server/` | Phase 1 |
| `spikeforge-targets` | `spikeforge_targets` | `spikeforge_targets`, `spikeforge_targets.backends`, `spikeforge_targets.energy`, `spikeforge_targets.event_runtime` | Phase 3 |
| `spikeforge-hub` | `spikeforge_hub` | `spikeforge_hub` | Phase 4 (go) |
| `spikeforge-dashboard` (npm `spikeforge-dashboard`, private) | n/a — npm | `client/` | Phase 2 |

### Core import root

The core distribution owns the import root `spikeforge`. The project rename
**landed**: the bring-up project was renamed to `spikeforge` and moved under the
[`capsize-games`](https://github.com/capsize-games) organization, so the core
distribution, its import root, and its repository all read `spikeforge`. No
alias, shim, or namespace package is used for core.

### Server import root

The `spikeforge-server` distribution keeps the existing top-level package
`server` as its import root (end state: distribution `spikeforge-server`, import
root `server`). Rationale: `server/` is already a top-level package and every one
of its library-importing files imports `spikeforge`, so keeping `server` avoids
churn in CI, Docker, and the WS entry point. The `spikeforge-server` repository
is a deliberate no-go (trigger T4 did not fire), so the server stays in this
repository as a separate distribution; no `spikeforge_server/` rename is planned
and no alias package is used.

### Targets, hub, and dashboard import roots

- `spikeforge-targets` owns `spikeforge_targets` and absorbs `targets`, `targets/backends`,
  `energy`, and `event_runtime`. These four subpackages move together because
  `targets/backends` depends on `energy`/`event_runtime` and all carry the
  volatile backend SDK surface.
- `spikeforge-hub` owns `spikeforge_hub` and absorbs `hub`.
- The dashboard keeps the npm package name `spikeforge-dashboard` even though
  its GitHub repository and product name become `spikeforge-dashboard`. The npm package
  stays `private: true`.

## GitHub repository names

| Repository | Phase | Status |
|---|---|---|
| `capsize-games/spikeforge-dashboard` | Phase 2 | **extracted** (T1 fired) |
| `capsize-games/spikeforge-targets` | Phase 3 | **extracted** (T2 fired) |
| `capsize-games/spikeforge-hub` | Phase 4 | **extracted** (T3 fired) |
| `capsize-games/spikeforge-server` | Phase 4 | **no-go** — T4 did not fire |
| `capsize-games/spikeforge` | — | the core repository (renamed to `spikeforge`) |

All repositories live under the [`capsize-games`](https://github.com/capsize-games)
organization and use hyphenated `spikeforge*` names; the core repository is
[`capsize-games/spikeforge`](https://github.com/capsize-games/spikeforge).

## Dependency direction

The invariant is **dependency arrows point from a component to what it
requires**, and there is no cycle. A satellite may depend on `spikeforge`;
`spikeforge` never depends on a satellite.

```mermaid
flowchart TB
    dashboard[spikeforge-dashboard React plus Vite]
    server[spikeforge-server server package]
    hub[spikeforge-hub spikeforge_hub]
    targets[spikeforge-targets spikeforge_targets]
    core[spikeforge spikeforge]
    protocol[protocol JSON Schema contract]

    targets --> core
    hub --> core
    server --> core
    server --> targets
    server --> hub
    server --> protocol
    dashboard --> protocol
    core --> protocol
```

Reading the edges as "depends on": `spikeforge-targets`, `spikeforge-hub`, and
`spikeforge-server` each depend on core; the server additionally depends on
`spikeforge_targets` and `spikeforge_hub` because its files import those
capability roots; and both the server and the dashboard depend on the
`protocol/` contract, never on each other's code.

### Why this direction is stable

- **Core is the floor.** It has no incoming packaging dependency on a
  satellite, which is what makes a headless `pip install spikeforge`
  provably free of the forbidden list in
  [`plans/arch-0001-core-boundary.md`](plans/arch-0001-core-boundary.md).
- **The protocol is a leaf, not a library.** Because `protocol/` is data (JSON
  Schema), the dashboard and the server can both depend on it without depending
  on each other — which is what makes the Phase 2 dashboard extraction safe.
- **Capabilities are siblings.** `spikeforge-targets` and `spikeforge-hub` do not depend on
  each other, so either can be extracted without ordering constraints beyond
  core.

## Layout mechanism

Each distribution is defined by its own PEP 621
[`pyproject.toml`](plans/arch-0001-packaging-versioning.md) under a top-level
`packages/` workspace directory. The distribution's build config selects its
import root via `package-dir`; the extraction is **done**, so each import root
now lives at its own top-level path in this repository and in its satellite.

```text
packages/
  spikeforge/
    pyproject.toml            # distribution spikeforge, import root spikeforge
  spikeforge-server/
    pyproject.toml            # distribution spikeforge-server, import root server
  spikeforge-targets/
    pyproject.toml            # distribution spikeforge-targets, import root spikeforge_targets
  spikeforge-hub/
    pyproject.toml            # distribution spikeforge-hub, import root spikeforge_hub
```

The core distribution's `find` configuration explicitly excludes the roots it
does not own, so `server/` can no longer leak into the core wheel:

```toml
[tool.setuptools.packages.find]
where = ["../.."]
include = ["spikeforge*"]
exclude = ["server*", "tests*", "spikeforge_targets*", "spikeforge_hub*"]
```

This is the direct fix for the verified fact that
[`setup.py`](setup.py:37) currently packages `server/` into `spikeforge`.

## Related decisions

- [`plans/arch-0001-adr-repo-topology.md`](plans/arch-0001-adr-repo-topology.md) — why one repo, multiple distributions.
- [`plans/arch-0001-core-boundary.md`](plans/arch-0001-core-boundary.md) — the forbidden-import contract for core.
- [`plans/arch-0001-packaging-versioning.md`](plans/arch-0001-packaging-versioning.md) — names, extras, scripts, pinning.
- [`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md) — the reversible move sequence.
- [`plans/repo_topology_plan.md`](plans/repo_topology_plan.md) — the source analysis.

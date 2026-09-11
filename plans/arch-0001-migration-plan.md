# ARCH-0001 migration plan

**Status: accepted (proposed for maintainer sign-off).**
**Date:** 2026-09-11 · **Issue:** ARCH-0001 *Phased repo split: core library, deploy targets, dashboard*
**Owner:** Capsize Games (maintainer) · **Depends on:** [`plans/arch-0001-adr-repo-topology.md`](plans/arch-0001-adr-repo-topology.md)

## Decision

Execute the split in four ordered phases. **Phase 1 moves no code and is the
only mandatory phase**; Phases 2–4 each fire only when their trigger
([`plans/arch-0001-decision-metrics.md`](plans/arch-0001-decision-metrics.md))
is met. Every phase is **additive before subtractive** and **reversible**: the
new repository is created and pushed from git history before anything is deleted
here, and the pre-phase commit is the rollback point.

## Phase 1 — monorepo, multiple distributions (no repo moves)

Goal: make the boundaries real without moving a file.

1. Add the `protocol/` contract authority, the `protocol_version` envelope
   field, the TS codegen/validation step, and
   `tests/test_protocol_schema_parity.py`
   ([`plans/arch-0001-protocol-contract.md`](plans/arch-0001-protocol-contract.md)).
2. Add the `packages/` workspace with `packages/spikeforge/pyproject.toml`
   (excluding `server/`) and `packages/spikeforge-server/pyproject.toml`
   ([`plans/arch-0001-packaging-versioning.md`](plans/arch-0001-packaging-versioning.md)).
3. Extend `scripts/blocked_deps/sitecustomize.py` with `fastapi`, `pydantic`,
   `uvicorn`; add `scripts/check_core_boundary.py`; add the `headless` CI step
   ([`plans/arch-0001-core-boundary.md`](plans/arch-0001-core-boundary.md)).
4. Remove the `web` extra from core; make it the server distribution's base
   dependencies. Update [`docker-compose.yml`](docker-compose.yml) and
   [`Dockerfile`](Dockerfile) to install `spikeforge-server`.
5. Port `tests/test_packaging_profiles.py` to read the new
   `packages/*/pyproject.toml` files instead of [`setup.py`](setup.py).

**No test moves.** Tests stay where they are because nothing has moved.

**Rollback:** revert the Phase 1 PR. Core packaging returns to `setup.py`; the
protocol work is additive and can remain.

## Phase 2 — extract the dashboard (trigger T1)

1. On a throwaway clone, `git subtree split --prefix=client -b spikeforge-dashboard-split`
   and push to a new `capsize-games/spikeforge-dashboard`.
2. Move the dashboard CI job (`npm ci && npm run build`) and the
   `client/package-lock.json` cache to the new repo.
3. In core, pin the dashboard bundle version in `compatibility.json`; the server
   serves a pinned prebuilt bundle instead of building `client/` in-repo.
4. Keep `client/` in core for one release as a read-only mirror, then delete it.

**Test relocation:** the parity tests
([`tests/test_client_animation_payload.py`](tests/test_client_animation_payload.py)
and the schema parity test) **stay in core**, because core owns `protocol/` and
is the cross-contract authority. Their inputs become the pinned generated TS
artifact rather than a live `client/` checkout.

**Rollback:** restore `client/` from the mirror (or from the pre-phase tag); the
server build reverts to in-repo `client/`.

## Phase 3 — extract `spikeforge-targets` (trigger T2)

1. On a throwaway clone, run the `git filter-repo` multi-prefix extraction from
   [`plans/arch-0001-packaging-versioning.md`](plans/arch-0001-packaging-versioning.md)
   and push to a new `capsize-games/spikeforge-targets` with import root `spikeforge_targets`.
2. Delete `spikeforge/{targets,energy,event_runtime}` from core; add the
   `spikeforge-targets` pinned dev dependency and the lazy re-export shim.
3. Move console-script ownership of `spikeforge-energy` and `spikeforge-targets` to the new
   distribution.
4. Move the tests for the moved code into `spikeforge-targets`; the server updates its
   imports from `spikeforge.energy` / `.event_runtime` / `.targets` to
   `spikeforge_targets.*`.

**Test relocation rule:** tests move with their subject. Tests that reference
`targets`, `energy`, or `event_runtime` relocate to `capsize-games/spikeforge-targets`. Tests
that exercise the *seam* (the server driving a target through the protocol) stay
in the cross-package integration suite described below.

**Rollback:** restore the four prefixes from the pre-phase tag; drop the pin and
shim.

## Phase 4 — extract `spikeforge-hub` (go) and `spikeforge-server` (conditional)

- **`spikeforge-hub` (go, trigger T3).** Single-directory `git subtree split` of
  `spikeforge/hub` to `capsize-games/spikeforge-hub` with import root `spikeforge_hub`; move the
  `spikeforge-hub` console script and hub tests; keep `spikeforge.hub` as a lazy
  shim.
- **`spikeforge-server` (conditional, trigger T4).** Only when the server must release
  on its own cadence: `git subtree split --prefix=server` to
  `capsize-games/spikeforge-server`. The server keeps import root `server`
  ([`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md)).
  Its 17 server-touching test files and the `server/schemas/*` modules move with
  it.

**Protocol ownership after Phase 4 (if it fires):** `protocol/` remains in
`capsize-games/spikeforge` as the contract authority. The server and dashboard
repos consume it as a pinned artifact or git submodule; they never fork it.

## Cross-package integration suite

Regardless of how far extraction goes, one **thin** suite stays co-located with
the protocol authority (core) and installs the pinned satellites. It guards the
seams rather than the internals:

- `tests/test_client_animation_payload.py` — the client-facing payload contract.
- `tests/test_protocol_schema_parity.py` — pydantic models vs JSON Schema vs
  `protocol_version.txt`.
- one server-through-core smoke test that drives a real request over the
  protocol using the pinned `spikeforge-server`.

Everything else migrates to the owning repository.

## Docs strategy

`plans/` stays **authoritative and in the core repository**
([`plans/index.md`](plans/index.md)); `scripts/build_docs.sh` keeps globbing
`plans/*.md` and this migration does not move the docs pipeline. Satellite
repositories carry only a minimal `README.md` and link back to the core docs
site for shared design. If extraction reaches Phase 4 and the site must build
per repo, the core site remains the hub page; no `plans/` content is duplicated.

## Deprecation and shim policy for moved import paths

| Old path | New path | Phase | Shim lifetime |
|---|---|---|---|
| `spikeforge.targets.*` | `spikeforge_targets.*` | 3 | 1 minor release after extraction |
| `spikeforge.targets.backends.*` | `spikeforge_targets.backends.*` | 3 | 1 minor release |
| `spikeforge.energy.*` | `spikeforge_targets.energy.*` | 3 | 1 minor release |
| `spikeforge.event_runtime.*` | `spikeforge_targets.event_runtime.*` | 3 | 1 minor release |
| `spikeforge.hub.*` | `spikeforge_hub.*` | 4 | 1 minor release |

Shims emit `DeprecationWarning` naming the new path, import the satellite
lazily, and raise a clear `ImportError` when it is not installed. Shims are
removed at the next MAJOR bump (or at 1.0), whichever comes first. Legacy
WebSocket payload keys, `main.py`, `main_encodings.py`, and the legacy checkpoint
keys `_fc1/_lif1/_fc2/_lif2` are **not** affected and remain stable throughout —
this migration is about packaging topology, not the public runtime surface.

## Reversibility summary

Every phase is guarded by its trigger
([`plans/arch-0001-decision-metrics.md`](plans/arch-0001-decision-metrics.md)),
landed as one reviewable PR, and tagged beforehand. Because extraction pushes a
new repository from history *before* deleting anything, a failed phase is
reverted by checking out the pre-phase tag and deleting the pin — no data is
lost. Risks and their mitigations are tracked in
[`plans/arch-0001-risk-register.md`](plans/arch-0001-risk-register.md).

## Related decisions

- [`plans/arch-0001-adr-repo-topology.md`](plans/arch-0001-adr-repo-topology.md) — the topology decision.
- [`plans/arch-0001-packaging-versioning.md`](plans/arch-0001-packaging-versioning.md) — pinning and the extraction commands.
- [`plans/arch-0001-core-boundary.md`](plans/arch-0001-core-boundary.md) — what Phase 1 enforces.
- [`plans/arch-0001-decision-metrics.md`](plans/arch-0001-decision-metrics.md) — when each phase fires.
- [`plans/repo_topology_plan.md`](plans/repo_topology_plan.md) — the source analysis.

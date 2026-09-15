# ARCH-0001 migration plan

**Status: accepted — implemented.**
**Date:** 2026-09-11 · **Issue:** ARCH-0001 *Phased repo split: core library, deploy targets, dashboard*
**Owner:** Capsize Games (maintainer) · **Depends on:** [`plans/arch-0001-adr-repo-topology.md`](plans/arch-0001-adr-repo-topology.md)

## Decision

The split was executed in four ordered phases. Phase 1 (multiple distributions in
one repository) was mandatory and landed; Phases 2–4 fired only when their
trigger
([`plans/arch-0001-decision-metrics.md`](plans/arch-0001-decision-metrics.md))
was met. Phase 2 (dashboard), Phase 3 (`spikeforge-targets`), and Phase 4
(`spikeforge-hub`) shipped; the Phase 4 `spikeforge-server` extraction did not
fire (T4 no-go). Every phase was **additive before subtractive** and
**reversible**: the new repository was created and pushed from git history before
anything was deleted here, and the pre-phase commit is the rollback point.

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
   `packages/*/pyproject.toml` files instead of `setup.py`.

**No test moves.** Tests stay where they are because nothing has moved.

**Rollback:** revert the Phase 1 PR. Core packaging returns to `setup.py`; the
protocol work is additive and can remain.

## Phase 2 — extracted the dashboard (trigger T1)

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

## Phase 3 — extracted `spikeforge-targets` (trigger T2)

1. On a throwaway clone, ran the `git filter-repo` multi-prefix extraction from
   [`plans/arch-0001-packaging-versioning.md`](plans/arch-0001-packaging-versioning.md)
   and pushed to `capsize-games/spikeforge-targets` with import root `spikeforge_targets`.
2. Deleted the four legacy submodules (`targets`, `energy`, `event_runtime`,
   plus `hub`) and `targets/backends` from the core package, and added the
   `spikeforge-targets` pinned dev dependency. Because the project is pre-1.0 and
   unpublished, **no re-export shim was added** (see "No back-compat aliases").
3. Moved console-script ownership of `spikeforge-energy` and `spikeforge-targets` to the new
   distribution.
4. Moved the tests for the moved code into `spikeforge-targets`; the server's
   imports now use `spikeforge_targets.*` (`spikeforge_targets.energy`,
   `spikeforge_targets.event_runtime`).

**Test relocation rule:** tests move with their subject. Tests that reference
`targets`, `energy`, or `event_runtime` relocated to `capsize-games/spikeforge-targets`. Tests
that exercise the *seam* (the server driving a target through the protocol) stay
in the cross-package integration suite described below.

**Rollback:** restore the four prefixes from the pre-phase tag and drop the pin.

## Phase 4 — extracted `spikeforge-hub` (go) and `spikeforge-server` (no-go)

- **`spikeforge-hub` (go, trigger T3; extracted).** Single-directory
  `git subtree split` of `spikeforge_hub` to `capsize-games/spikeforge-hub` with
  import root `spikeforge_hub`; moved the `spikeforge-hub` console script and hub
  tests. No re-export shim was retained.
- **`spikeforge-server` (conditional, trigger T4; no-go).** T4 did not fire, so
  no `capsize-games/spikeforge-server` repository was created. The split would
  run only when the server must release on its own cadence —
  `git subtree split --prefix=server` to `capsize-games/spikeforge-server`. The
  server keeps import root `server`
  ([`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md)),
  and its 17 server-touching test files and the `server/schemas/*` modules would
  move with it.

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

## No back-compat aliases

The extraction shipped **without** legacy import-path aliases. Because the
project is pre-1.0, unpublished, and has no external importers to protect, the
old `spikeforge.{targets,energy,event_runtime,hub}` module paths were deleted
outright rather than kept alive as a deprecated re-export shim:

| Removed core submodule | Replacement root |
|---|---|
| `targets` | `spikeforge_targets` |
| `targets.backends` | `spikeforge_targets.backends` |
| `energy` | `spikeforge_targets.energy` |
| `event_runtime` | `spikeforge_targets.event_runtime` |
| `hub` | `spikeforge_hub` |

There is no `DeprecationWarning` window, no "1 minor release" shim lifetime, and
no alias to retire later. Legacy WebSocket payload keys, `main.py`,
`main_encodings.py`, and the legacy checkpoint keys `_fc1/_lif1/_fc2/_lif2` are
**not** affected and remain stable throughout — this migration is about
packaging topology, not the public runtime surface.

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

# ARCH-0001 risk register

**Status: accepted (proposed for maintainer sign-off).**
**Date:** 2026-09-11 · **Issue:** ARCH-0001 *Phased repo split: core library, deploy targets, dashboard*
**Owner:** Capsize Games (maintainer) · **Depends on:** [`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md)

## Decision

Every cost identified in [`plans/repo_topology_plan.md`](plans/repo_topology_plan.md)
§5 is carried forward with a concrete mitigation and a single accountable owner
(**Capsize Games**, the maintainer). Additionally, the ARCH-0001 review surfaced six
implementation-level risks (PR-1…PR-6) that are registered here because they can
break the build or the release pipeline, not just the design.

Allowed status is `accepted` (mitigated and owned) or `open` (no mitigation
agreed yet). No risk in this register is currently `open`.

## Topology costs from the source plan

| ID | Risk | Source | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| SR-1 | **Version drift.** Satellites, core, and the dashboard pin incompatible versions; a protocol change breaks client/server across repos. | `plans/repo_topology_plan.md` §5 | High | `compatibility.json` records the satellite/core/`protocol_version`/dashboard triples; CI asserts every satellite pin matches the matrix; MAJOR protocol bumps require an ordered server-then-dashboard release ([`plans/arch-0001-packaging-versioning.md`](plans/arch-0001-packaging-versioning.md)). | Capsize Games |
| SR-2 | **CI and release overhead.** Today's six jobs become per-repo pipelines; cross-repo changes need coordinated PRs and tags. | `plans/repo_topology_plan.md` §5 | Medium | One reusable workflow definition parameterized by distribution; a single `release.yml` selects `packages/<dist>` from the tag prefix; no per-repo pipeline is created until its trigger fires ([`plans/arch-0001-decision-metrics.md`](plans/arch-0001-decision-metrics.md)). | Capsize Games |
| SR-3 | **Test fixtures.** Tests lean on `spikeforge` heavily and a relocation loses coverage or shared fixtures. | `plans/repo_topology_plan.md` §5 | Medium | Tests move with their owning package; a thin cross-package integration suite stays with the protocol authority; the parity tests are explicitly retained ([`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md)). | Capsize Games |
| SR-4 | **Docs fragmentation.** `plans/` is one authoritative site; splitting repos means a docs repo or per-repo docs plus a hub. | `plans/repo_topology_plan.md` §5 | Medium | `plans/` stays authoritative in the core repo; satellites carry only a README linking back; the docs pipeline keeps globbing `plans/*.md`. | Capsize Games |
| SR-5 | **Duplicated governance.** LICENSE, NOTICE, SECURITY, CODE_OF_CONDUCT, CI scaffolding, and Docker files multiply. | `plans/repo_topology_plan.md` §5 | Low | Create repositories from the shared templates at extraction time, not ahead; a sync check compares LICENSE/NOTICE hashes across repos at each release. | Capsize Games |
| SR-6 | **Contributor friction.** For a pre-1.0 project, a monorepo with clean distributions is friendlier than four repos. | `plans/repo_topology_plan.md` §5 | Medium | Stay on Option 0 until a numeric trigger fires; keep `CONTRIBUTING.md` pointing at the `packages/` workspace as the normal path; extraction is never scheduled on a date. | Capsize Games |

## Implementation-level risks discovered in ARCH-0001 review

| ID | Risk | Impact | Mitigation | Owner |
|---|---|---|---|---|
| PR-1 | **PEP 420 namespace collision.** Two distributions shipping into one top-level root would silently merge. | High | Every distribution owns a distinct top-level import root; PEP 420 is explicitly not used ([`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md)); a CI check asserts the core wheel's top-level entries are disjoint from each satellite's. | Capsize Games |
| PR-2 | **Blocked-deps blind spot.** `sitecustomize.py` blocks the SDKs but not `fastapi`/`pydantic`/`uvicorn`, so a core import of the server stack would pass CI. | High | Extend `BLOCKED` with the server stack and add `scripts/check_core_boundary.py` ([`plans/arch-0001-core-boundary.md`](plans/arch-0001-core-boundary.md)). | Capsize Games |
| PR-3 | **Packaging-authority migration.** `tests/test_packaging_profiles.py` reads [`setup.py`](setup.py) text; introducing `pyproject.toml` silently drops that coverage. | Medium | Port the test to read `packages/*/pyproject.toml` in the same PR that adds the workspace; add a CI assertion that the set of extras and console scripts is unchanged by the migration. | Capsize Games |
| PR-4 | **History rewrite during extraction.** `git filter-repo` rewrites history and can corrupt a working clone. | Medium | Run it on a throwaway clone only; extraction is additive-then-subtractive; the pre-phase tag is the rollback point ([`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md)). | Capsize Games |
| PR-5 | **Console-script collision.** A script name owned by two distributions fails install or shadows unpredictably. | Medium | The ownership table assigns each script to exactly one distribution; a CI check asserts no script name appears in two distributions' entry points ([`plans/arch-0001-packaging-versioning.md`](plans/arch-0001-packaging-versioning.md)). | Capsize Games |
| PR-6 | **Protocol drift after the split.** The dashboard hand-edits generated types or the pydantic models diverge from the schemas. | High | CI regenerates TS types and fails on any diff; `test_protocol_schema_parity.py` asserts model-enum == schema-enum; the schema is the only editable authority ([`plans/arch-0001-protocol-contract.md`](plans/arch-0001-protocol-contract.md)). | Capsize Games |

## Review cadence

This register is revisited at each of the following points, whichever comes
first: a phase trigger firing, a `protocol_version` MAJOR bump, or a quarterly
maintainer review of [`plans/arch-0001-decision-metrics.md`](plans/arch-0001-decision-metrics.md).
A risk may only move to `open` with a named owner and a dated decision recorded
in this file.

## Related decisions

- [`plans/repo_topology_plan.md`](plans/repo_topology_plan.md) — the cost list this register extends.
- [`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md) — the reversible procedure the mitigations rely on.
- [`plans/arch-0001-core-boundary.md`](plans/arch-0001-core-boundary.md) — mitigates PR-2.
- [`plans/arch-0001-protocol-contract.md`](plans/arch-0001-protocol-contract.md) — mitigates SR-1 and PR-6.
- [`plans/arch-0001-decision-metrics.md`](plans/arch-0001-decision-metrics.md) — the triggers reviewed alongside this register.

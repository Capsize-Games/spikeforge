# ARCH-0001 decision metrics

**Status: accepted — implemented.**
**Date:** 2026-09-11 · **Issue:** ARCH-0001 *Phased repo split: core library, deploy targets, dashboard*
**Owner:** Capsize Games (maintainer) · **Depends on:** [`plans/arch-0001-adr-repo-topology.md`](plans/arch-0001-adr-repo-topology.md)

## Decision

Extraction is gated by **observable, numeric triggers**. A phase fires when its
trigger's condition is met on the measurement described below; no extraction is
scheduled by date, and meeting a threshold is necessary but the maintainer still
records the decision in the ADR. The triggers map one-to-one to the phases in
[`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md).

## Measurement method

A `scripts/topology_metrics.py` helper (added in Phase 1) computes the inputs
from `git log`, GitHub releases, and CI timings, and prints one table per
release cycle. Definitions:

- **Release** = a pushed `*-v*` tag from the release workflow.
- **Unplanned core release** = a `spikeforge` tag whose changelog names only
  a backend-SDK or other satellite-owned cause.
- **Isolated change set** = a merged PR whose changed paths fall entirely inside
  one component's prefix (`client/`, or `spikeforge_targets/` (including
  `energy/` and `event_runtime/`), or `spikeforge_hub/`, or `server/`).
- **Rolling window** = the trailing 90 days.
- **CI wall-clock share** = component build/step time divided by total pipeline
  time, from the workflow run timing API.

Cadence: computed at each release and reviewed quarterly. Thresholds are
re-baselined after every phase that fires.

## Trigger definitions

| Trigger | Phase | Condition — fires if **all** clauses in a block hold | Rationale |
|---|---|---|---|
| **T1 — dashboard** | Phase 2 | (a) ≥ 4 isolated `client/`-only changes in the rolling window could not ship without a core release, **or** (b) client build time ≥ 10 minutes **or** ≥ 40% of pipeline wall-clock for 2 consecutive weeks, **or** (c) ≥ 3 PRs in the rolling window were blocked waiting on Python-side review. Any single clause fires. | The dashboard is already an isolated npm package; independence pays off only when client-only work is throttled by the library cadence. |
| **T2 — `spikeforge-targets`** | Phase 3 | (a) ≥ 2 unplanned **core** releases per quarter are caused solely by `norse` or `lava-nc` version churn, **or** (b) a single backend SDK forces ≥ 2 core releases within 30 days, **or** (c) ≥ 6 isolated `targets`/`energy`/`event_runtime` change sets per quarter require a core release. Any single clause fires. | The volatile external backend SDKs are the strongest technical reason to decouple; their churn should not gate the library. |
| **T3 — `spikeforge-hub`** | Phase 4 (go) | (a) hub-only release demand ≥ 2 in the rolling window, **and** (b) the catalog `schema_version` is unchanged across ≥ 2 consecutive releases, **and** (c) isolated `hub/` change sets are ≥ 15% of core-only commits over a quarter. All clauses must hold. | The hub owns network I/O and curated-catalog versioning; the catalog must be stable before it can own its own cadence. |
| **T4 — `spikeforge-server`** | Phase 4 (conditional) | (a) server-only release demand ≥ 3 in the rolling window, **and** (b) ≥ 2 core pin conflicts or downgrades within the window, **and** (c) isolated `server/` change sets are ≥ 20% of core commits over a quarter. All clauses must hold. | The server already has its own distribution after Phase 1; a repository is warranted only when it must release faster than core. |

## Ordering and conflict rules

- **Fixed extraction order:** dashboard, then `spikeforge-targets`, then `spikeforge-hub`, then
  `spikeforge-server`. If two triggers fire in the same review, only the earliest in
  this order proceeds; the other is re-evaluated next cycle.
- **One extraction per release cycle.** A cycle is a core minor release. This
  bounds coordination cost (SR-2 in
  [`plans/arch-0001-risk-register.md`](plans/arch-0001-risk-register.md)).
- **A trigger that stops holding de-escalates.** If a component fires a trigger
  but the maintainer defers, the metrics are re-measured next cycle; a component
  may never be extracted on a single period's signal alone unless a clause
  explicitly allows it (T1b, T2b).

## Health metrics — always zero

These are not extraction triggers; they are invariants that must hold in every
CI run. A non-zero value blocks the release.

| Metric | Target | Enforced by |
|---|---|---|
| Core boundary violations | 0 | `scripts/check_core_boundary.py` and the extended `blocked-deps` gate |
| Forbidden distributions present in a headless core install | 0 | the `headless` CI step |
| `server/` entries in the core wheel | 0 | wheel archive listing assertion |
| Protocol parity failures | 0 | `tests/test_protocol_schema_parity.py` |
| Generated TS types out of date | 0 | `git diff --exit-code client/src/protocol/generated.ts` |
| Scripts owned by more than one distribution | 0 | the packaging-profile port of `tests/test_packaging_profiles.py` |
| Top-level import roots shared by two distributions | 0 | the PEP 420 disjointness check (PR-1) |

## Baseline note

`scripts/topology_metrics.py` is implemented and run each review cycle. T1
(dashboard), T2 (`spikeforge-targets`), and T3 (`spikeforge-hub`) fired and
their extractions shipped; T4 (`spikeforge-server`) did not fire, so the server
extraction remains on the shelf and the trigger is re-measured every release
cycle.

## Related decisions

- [`plans/arch-0001-adr-repo-topology.md`](plans/arch-0001-adr-repo-topology.md) — the triggers named in the ADR.
- [`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md) — the phases these triggers gate.
- [`plans/arch-0001-core-boundary.md`](plans/arch-0001-core-boundary.md) — the health metrics it enforces.
- [`plans/arch-0001-risk-register.md`](plans/arch-0001-risk-register.md) — the risks tracked at each review.

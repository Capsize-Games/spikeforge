# ARCH-0001 ADR: repository topology

**Status: accepted (proposed for maintainer sign-off).**
**Date:** 2026-09-11 · **Issue:** ARCH-0001 *Phased repo split: core library, deploy targets, dashboard*
**Owner:** Capsize Games (maintainer) · **Extends:** [`plans/repo_topology_plan.md`](plans/repo_topology_plan.md)

## Context

The repository is one Python distribution (`spikeforge` version `0.2.0`,
declared only in [`setup.py`](setup.py:22)), one FastAPI + WebSocket adapter
(`server/`), and one React dashboard (`client/`, npm package
`spikeforge-dashboard`). The source analysis in
[`plans/repo_topology_plan.md`](plans/repo_topology_plan.md) establishes that the
component seams are real but that the right first move is *multiple
distributions in one repository*, not multiple repositories. This ADR records
the accepted topology and the conditions under which that decision is revisited.

Two facts about the current tree sharpen the decision:

- `find_packages(exclude=("tests", "tests.*"))` today also packages `server/`,
  so the library and the server are **not** separate distributions despite the
  plan's component table calling the server a "separate dist today"
  (see Corrections below).
- Every optional SDK is already confined to a lazy, degrading shim
  (`nir_bridge/api.py`, `onnx_bridge/api.py`, `targets/backends/api.py`,
  `events/tonic_api.py`, `hub/probe.py`, `targets/probe.py`, `tracking/*_sink.py`),
  so a headless core install is a packaging change, not a rewrite.

## Decision

**ADOPT Option 0: one repository, multiple distributions, each owning exactly
one top-level import root.** Repositories are extracted only when a named
trigger condition in this ADR fires. The concrete distribution names, import
roots, and GitHub repository names this decision resolves to live in
[`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md).

Option 0 is not a staging step that is expected to be discarded; it is the
default steady state. Extraction is an escalation that must be justified by the
triggers below, not a schedule.

## Why not Options 1–3

`plans/repo_topology_plan.md` §3 defines Options 1 (three repos: core |
targets | dashboard), 2 (four repos: + hub) and 3 (five-plus repos: + server,
client). Option 0 is adopted over all three for the beta-period reasons below;
the options remain the destination topology *after* the corresponding trigger
fires.

| Criterion | Option 0 (adopted now) | Option 1 | Option 2 | Option 3 |
|---|---|---|---|---|
| "Library without server/client" | Yes | Yes | Yes | Yes |
| Independent release cadence | Partial | Good | Good | Best |
| Version-drift risk | Lowest | Medium | Medium-high | High |
| CI/release overhead | about 1x | about 3x | about 4x | about 5x |
| Atomic cross-cutting changes | Easy | Hard | Hard | Very hard |
| Protocol ownership | One repo | One repo | One repo | Split risk |
| Fits a 0.2.0 beta with a small team | Best | Later | Later | Much later |

The decisive asymmetry is that Option 0 already delivers the user-visible win
("install the interpreter layer without the server or the dashboard") at close
to zero coordination cost, while every multi-repo option multiplies the
governance surface (LICENSE, NOTICE, SECURITY, CODE_OF_CONDUCT, CI scaffolding,
Docker files) and turns each cross-cutting change into coordinated PRs and tags.

## Extraction trigger conditions

Each later phase in [`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md)
is gated by an observable trigger. The trigger definitions and their numeric
thresholds are owned by [`plans/arch-0001-decision-metrics.md`](plans/arch-0001-decision-metrics.md);
this ADR records only the named conditions.

- **T1 — extract the dashboard (Phase 2).** Fires when dashboard-only work
  cannot ship without a library release at the rate defined in the metrics
  document, or when the client build dominates CI cost past its threshold.
- **T2 — extract `spikeforge-targets` (Phase 3).** Fires when backend SDK churn
  (`norse`, `lava-nc`) forces unplanned *core* patch releases at the rate
  defined in the metrics document. This is the strongest technical case for a
  split because those SDKs are external and fast-moving.
- **T3 — extract `spikeforge-hub` (Phase 4, **go**).** Fires when hub-only release
  demand and catalog-format stability both meet their thresholds. The hub is
  marked **go** because it owns network I/O and a curated-catalog versioning
  cadence that genuinely diverges from the library.
- **T4 — extract `spikeforge-server` (Phase 4, **conditional**).** Fires only when the
  server must release on a cadence that the core cannot absorb, as measured in
  the metrics document. Until then the server ships as a separate *distribution*
  in this repo (Phase 1), which already satisfies "install the library without
  the server".

While a trigger is unmet, the corresponding component remains in this
repository as a separate distribution. No repository is created speculatively.

## Consequences

- **Positive.** The core boundary becomes enforceable in CI
  ([`plans/arch-0001-core-boundary.md`](plans/arch-0001-core-boundary.md)); the
  protocol gets a versioned source of truth
  ([`plans/arch-0001-protocol-contract.md`](plans/arch-0001-protocol-contract.md));
  and packaging, versioning and release are defined once up front
  ([`plans/arch-0001-packaging-versioning.md`](plans/arch-0001-packaging-versioning.md)).
- **Negative.** Release cadence stays coupled until a trigger fires, and the
  monorepo must carry deliberate compat shims for moved import paths
  ([`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md)).
- **Neutral.** The `spikeforge` import root is not renamed by this ADR; the
  project rename noted in `plans/repo_topology_plan.md` §4 stays out of scope
  for ARCH-0001.

## Corrections to the source plan

The verified tree disagrees with `plans/repo_topology_plan.md` in three places.
The verified facts govern:

1. **Server is not a separate distribution today.** `plans/repo_topology_plan.md`
   §1.2 labels `server/` "`web` extra (separate dist today)". Verified: there is
   no separate distribution; `find_packages(exclude=("tests", "tests.*"))` in
   [`setup.py`](setup.py:37) packages `server/` into `spikeforge`. Phase 1
   makes the separation real.
2. **Test-file counts.** `plans/repo_topology_plan.md` §2 states 105 test files
   with "208 reference `spikeforge`" and "the 17 server tests". Verified:
   105 files under `tests/`; about 102 reference `spikeforge`; 17 reference
   `server`. The "17" is the number of test *files* touching `server/`, not a
   pass count.
3. **CI job count.** `plans/repo_topology_plan.md` §5 says "Today's five jobs".
   Verified: [`.github/workflows/ci.yml`](.github/workflows/ci.yml:13) defines
   **six** jobs — `lint`, `test` (py3.10–3.13), `extras`, `blocked-deps`,
   `docs`, `client`.

## Related decisions

- [`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md) — the accepted topology.
- [`plans/arch-0001-core-boundary.md`](plans/arch-0001-core-boundary.md) — the enforced core boundary.
- [`plans/arch-0001-protocol-contract.md`](plans/arch-0001-protocol-contract.md) — the versioned protocol.
- [`plans/arch-0001-packaging-versioning.md`](plans/arch-0001-packaging-versioning.md) — distributions and releases.
- [`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md) — reversible rollout.
- [`plans/arch-0001-risk-register.md`](plans/arch-0001-risk-register.md) — mitigations and owners.
- [`plans/arch-0001-decision-metrics.md`](plans/arch-0001-decision-metrics.md) — thresholds for T1–T4.

# spikeforge documentation

This site is generated from the authoritative design documents in `plans/`
and the project [README](README.md). The markdown remains the single source
of truth; the site is a browsable rendering of it. For *user* documentation —
install, extras, CLI, and the honest limitations — start with the README.

> **Status — implemented.** The professionalization program (workstreams
> WS-A…WS-F) described here shipped. See the README's "Model hub",
> "Backend execution", "Sequence primitives", "Event runtime and energy",
> "Operational maturity", and "Interop fold-ins" sections for what is
> delivered and where the honest boundaries are.

## Roadmap

- [Professionalization roadmap](plans/professional_roadmap.md) — the master
  program: six workstreams, interfaces, and the acceptance bar.
- [Ecosystem roadmap](plans/ecosystem_roadmap.md) — the earlier ecosystem
  plan this program builds on.
- [Ecosystem listings](plans/ecosystem_listings.md) — the two directories this
  field searches (NIR's framework table, Open Neuromorphic's software guide),
  what each currently lists, and drafted submissions for both.

## Architecture

- [Repo topology and split](plans/repo_topology_plan.md) — the component
  inventory, dependency layers, and the staged recommendation for whether (and
  when) to split the client, server, and interpreter/deploy layers into
  separate repositories.

### ARCH-0001 — phased repo split

The architecture for one repository with multiple distributions, now
implemented. These documents are the design of record for the split; the
`spikeforge` rename, the `capsize-games` organization move, and the
`spikeforge-targets`, `spikeforge-hub`, and dashboard extractions have shipped.
The `spikeforge-server` extraction did not fire (T4 no-go).

- [ADR: repository topology](plans/arch-0001-adr-repo-topology.md) — the
  accepted Option 0 decision, the comparison against Options 1–3, and the named
  extraction triggers.
- [Target topology](plans/arch-0001-target-topology.md) — the distribution
  names, import roots, GitHub repository names, and dependency direction.
- [Core boundary](plans/arch-0001-core-boundary.md) — the forbidden-import list
  and the CI checks that enforce a headless core install.
- [Protocol contract](plans/arch-0001-protocol-contract.md) — the JSON Schema
  source of truth, the `protocol_version` field and compatibility policy, and TS
  type generation/validation.
- [Packaging and versioning](plans/arch-0001-packaging-versioning.md) — the
  extras-to-package mapping, console-script ownership, pinning, and release
  automation.
- [Migration plan](plans/arch-0001-migration-plan.md) — the reversible Phases
  1–4, test/fixture relocation, and the no-back-compat-alias policy.
- [Risk register](plans/arch-0001-risk-register.md) — every split cost with a
  mitigation and an owner.
- [Decision metrics](plans/arch-0001-decision-metrics.md) — the numeric
  thresholds that gate the next extraction.

## Focused workstream designs

- [Interpreter spine](plans/interpreter_spine_plan.md) — the `TopologySpec`
  single source of truth and the NIR translation.
- [Model hub](plans/model_hub_plan.md) — curated catalog, downloads, and
  import/inspect (WS-A).
- [Backend execution](plans/backend_execution_plan.md) — substitution
  execution and real backends (WS-B).
- [Sequence primitives](plans/sequence_primitives_plan.md) — sequence/
  attention stage kinds and per-stage neurons (WS-C).
- [Event runtime](plans/event_runtime_plan.md) — sparse execution and energy
  accounting (WS-D).
- [Operations](plans/operations_plan.md) — persisted metrics, external
  tracking sinks, determinism, and this docs site (WS-E).
- [Interop fold-ins](plans/interop_foldins_plan.md) — ONNX, `nirtorch`,
  quantization, and geometry (WS-F).

## Production use cases (PC-0)

The chip-less production-use-case program (issue #12). The shared toolkit every
use case consumes is the
[hardware-free production toolkit](plans/production_toolkit_plan.md); the umbrella
and ticket seeds are [production use cases](plans/production_use_cases.md). Each
use case has a full specification mirroring UC-1 (problem, reference architecture,
offline/online design, phases P0–P6 with MVP = P0–P4, dependencies, out of scope,
and a ready-to-file GitHub issue payload).

| # | Use case | Spec | Status |
|---|---|---|---|
| 1 | Streaming time-series classification / anomaly detection | [UC-1](plans/use_case_streaming_timeseries.md) | implemented (spikeforge 0.3.0) |
| 2 | Always-on audio / keyword spotting / wake-word | [UC-2](plans/use_case_audio_keyword_spotting.md) | scoped |
| 3 | Event-camera (DVS) vision | [UC-3](plans/use_case_event_camera_vision.md) | scoped |
| 4 | Ultra-low-latency sensor-stream inference | [UC-4](plans/use_case_low_latency_sensor_stream.md) | scoped |
| 5 | Anomaly / intrusion detection (IoT, network, grid) | [UC-5](plans/use_case_intrusion_anomaly_detection.md) | scoped |
| 6 | Bio-signal & medical monitoring | [UC-6](plans/use_case_biosignal_medical_monitoring.md) | scoped |
| 7 | RL for control & robotics | [UC-7](plans/use_case_rl_control_robotics.md) | scoped |
| 8 | Efficient sequence models / spiking transformers | [UC-8](plans/use_case_spiking_transformers.md) | scoped |
| 9 | Edge/mobile inference under power budgets | [UC-9](plans/use_case_edge_power_budgets.md) | scoped |
| 10 | Computational neuroscience / neuromorphic R&D | [UC-10](plans/use_case_computational_neuroscience.md) | scoped |

Recommended implementation order: UC-1 → UC-2/UC-5 → UC-3/UC-4 → UC-6/UC-7 →
UC-8/UC-9/UC-10.

## Memory-system research (issues #22-#26)

A research track, distinct from PC-0: one-shot, non-forgetting spiking
memory, not a scoped production feature.

- [Memory-system research](plans/memory_system_research.md) — the
  held-out-digit POC (#22, shipped) and the camera-fed room memory
  pipeline design (#23, design only).

## Guides

- [Cookbook](COOKBOOK.md) — practical, runnable recipes for every shipped
  capability, organised by goal.
- [Examples](../examples/README.md) — small, runnable, offline-safe scripts,
  one per core journey, with the real output each one prints.
- [Open-source checklist](OPEN_SOURCE_CHECKLIST.md) — the pre-release
  readiness checklist (licensing, community, CI, packaging, docs).

## Reference

- [Project rules](rules.md) — the style and architecture contract.
- [Integration plan](INTEGRATION_PLAN.md) — the original integration notes.

## Building these docs

```bash
scripts/build_docs.sh          # build into build/docs
scripts/build_docs.sh --check  # fail on broken documentation links
```

# snn-interpreter documentation

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

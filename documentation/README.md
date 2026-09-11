# spikeforge — long-form reference

This folder holds the detailed, implementation-oriented documentation for
**spikeforge**. It is written for contributors and LLM coding agents that need
the full picture. The top-level [README](../README.md) is the short,
human-facing entry point and links back here.

Start with the [architecture](architecture.md) and
[project layout](project-layout.md), then follow the capability you need.

## Orientation

| Document | What it covers |
|---|---|
| [Quickstart](quickstart.md) | Install paths and the first run |
| [Requirements](requirements.md) | Dependencies, optional extras, and licensing |
| [Usage](usage.md) | Docker, local dev, CLI, and device selection |
| [Architecture](architecture.md) | The `TopologySpec` spine and the data flow |
| [Project layout](project-layout.md) | Module-by-module map and code conventions |
| [Development](development.md) | The `scripts/dev.sh` task runner |

## Capabilities

- [Features](features.md) — the full feature inventory.
- [Interpreter spine (Phase 1)](interpreter-spine.md) — topology presets, neuron registry, NIR export/validation.
- [Dual-mode introspection (Phase 2)](introspection.md) — educational/production execution, trajectory metrics, surrogate curves.
- [Dashboard (Phase 3)](dashboard.md) — panels, walkthroughs, and WebSocket actions.
- [Event datasets (Phase 4)](event-datasets.md) — N-MNIST, DVS, Tonic, and the event bridge.
- [Targets and interoperability (Phase 5)](targets-and-interop.md) — deployment targets, capability matrix, NIR round-trips.
- [Production workflows (Phase 6)](production-workflows.md) — manifests, model registry, scale-ups, benchmarks.
- [Model hub (WS-A)](model-hub.md) — curated catalog, downloads, inspect → compat → promote.
- [Backend execution (WS-B)](backend-execution.md) — substitutions and the reference/Norse/Lava backends.
- [Sequence primitives (WS-C)](sequence-primitives.md) — per-stage neurons and attention presets.
- [Event runtime and energy (WS-D)](event-runtime-and-energy.md) — sparse runner and SOP/MAC/AC accounting.
- [Operational maturity (WS-E)](operational-maturity.md) — persisted metrics, tracking sinks, determinism, docs site.
- [Interop fold-ins (WS-F)](interop-foldins.md) — event training, ONNX bridge, `nirtorch` extraction, quantization.

## Honesty and limits

Read these before trusting a number:

- [Implications and boundaries](implications-and-boundaries.md) — the six cross-cutting limitations, *why* they exist, and *what* they imply.
- [Notes](notes.md) — phase-specific limitations and caveats.

## Related material

- [Project rules](../rules.md) — code style, hard limits, and design invariants.
- [Cookbook](../COOKBOOK.md) — copy-pasteable recipes.
- [Examples](../examples/README.md) — ten runnable end-to-end scripts.
- [Protocol contract](../protocol/) — the WebSocket JSON Schema source of truth.
- [Open-source checklist](../OPEN_SOURCE_CHECKLIST.md) — pre-release readiness.
- `plans/` — the ARCH-0001 repository-split design, roadmaps, and workstream plans; use that folder's `index.md` as its table of contents.

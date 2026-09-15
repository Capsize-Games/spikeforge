# spikeforge — long-form reference

This is the complete reference for **spikeforge**, a spiking-neural-network
toolkit built on [snnTorch](https://snntorch.readthedocs.io/) and PyTorch —
install paths, the CLI tools, the architecture, and every shipped capability
with an honest note on how far each one actually goes.

Where to start depends on why you are here:

- **Evaluating it** — [Quickstart](quickstart.md) for the first run, then
  [Features](features.md) for the inventory and
  [Implications and boundaries](implications-and-boundaries.md) for what the
  numbers do and do not mean.
- **Building on it** — [Architecture](architecture.md) and
  [Project layout](project-layout.md) are the map, then follow the capability
  you need from the tables below.
- **Contributing to it** — [Development](development.md), then
  [Plans](../plans/index.md) for the design records behind each workstream.

The documentation is implementation-oriented and deliberately detailed, which
is also what makes it the reference contributors and coding agents work from.
The top-level [README](../README.md) is the short entry point and links back
here.

> **Editing this documentation?** If you also see a `docs/` folder locally,
> that is a generated, git-ignored MkDocs build (`scripts/build_docs.sh`
> regenerates it from this folder plus `plans/`) and never appears on GitHub.
> This folder is the one source of truth to read or edit.

## Orientation

| Document | What it covers |
|---|---|
| [Quickstart](quickstart.md) | Install paths and the first run |
| [Requirements](requirements.md) | Dependencies, optional extras, and licensing |
| [Usage](usage.md) | Docker, local dev, CLI, and device selection |
| [Architecture](architecture.md) | The `TopologySpec` spine and the data flow |
| [Project layout](project-layout.md) | Module-by-module map and code conventions |
| [Development](development.md) | The `scripts/dev.sh` task runner |
| [Benchmarks](benchmarks.md) | What the reference configurations score, and how to reproduce it |

## Capabilities

- [Features](features.md) — the full feature inventory.
- [Benchmarks](benchmarks.md) — published accuracy for each shipped
  dataset/topology reference configuration, with reproduction commands.
- [Interpreter spine (Phase 1)](interpreter-spine.md) — topology presets, neuron registry, NIR export/validation.
- [Dual-mode introspection (Phase 2)](introspection.md) — educational/production execution, trajectory metrics, surrogate curves.
- [Dashboard (Phase 3)](dashboard.md) — panels, walkthroughs, and WebSocket actions.
- [Event datasets (Phase 4)](event-datasets.md) — N-MNIST, DVS, Tonic, and the event bridge.
- [Targets and interoperability (Phase 5)](targets-and-interop.md) — deployment targets, capability matrix, NIR round-trips.
- [Production workflows (Phase 6)](production-workflows.md) — manifests, model registry, scale-ups, benchmarks.
- [Model hub (WS-A)](model-hub.md) — curated catalog, downloads, inspect → compat → promote.
- [Backend execution (WS-B)](backend-execution.md) — substitutions and the reference/Norse/Lava backends.
- [Sequence primitives (WS-C)](sequence-primitives.md) — per-stage neurons and attention presets.
- [Streaming time series (UC-1)](streaming-timeseries.md) — windowing, a synthetic stream, train/eval, and bundle serving parity.
- [Event runtime and energy (WS-D)](event-runtime-and-energy.md) — sparse runner and SOP/MAC/AC accounting.
- [Operational maturity (WS-E)](operational-maturity.md) — persisted metrics, tracking sinks, determinism, docs site.
- [Interop fold-ins (WS-F)](interop-foldins.md) — event training, ONNX bridge, `nirtorch` extraction, quantization.
- [Model deployment: bundles and modules](model-deployment.md) — bundling a checkpoint, running it headless, and installing it as a standalone command.

## Honesty and limits

Read these before trusting a number:

- [Implications and boundaries](implications-and-boundaries.md) — the six cross-cutting limitations, *why* they exist, and *what* they imply.
- [Notes](notes.md) — phase-specific limitations and caveats.

## Related material

- [Project rules](../rules.md) — code style, hard limits, and design invariants.
- [Cookbook](../COOKBOOK.md) — copy-pasteable recipes.
- [Examples](../examples/README.md) — fifteen runnable end-to-end scripts.
- [Protocol contract](../protocol/) — the WebSocket JSON Schema source of truth.
- [Open-source checklist](../OPEN_SOURCE_CHECKLIST.md) — pre-release readiness.
- `plans/` — the ARCH-0001 repository-split design, roadmaps, and workstream plans; use that folder's `index.md` as its table of contents.

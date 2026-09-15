# Spikeforge — WS-F: Interop and Fold-In Gaps

> Focused design for workstream **F** of the
> [`professional_roadmap.md`](plans/professional_roadmap.md). Read the roadmap
> first for the vision, cross-workstream interfaces, and packaging rules.

Design/spec only. Every claim about the current code is grounded in a file/line
reference so a Code-mode agent can execute this file-by-file.

**Objective.** Close the remaining gaps folded in from the earlier list:
train on **event datasets**, bridge to/from **ONNX**, ingest **third-party
PyTorch modules** via `nirtorch`, **apply** target quantization constraints, and
fix the small gaps — **non-square sensor geometry** and **per-step hidden-layer
animation** in the client. Each is a small, independently verifiable phase.

---

## 1. Current state and gap analysis

| Gap | Current reality | Anchor |
|---|---|---|
| Event-dataset training | `build_dataset` refuses event specs (`spec.cls is None`) | [`datasets.py`](spikeforge/data/datasets.py:111) |
| Event loading | `EventSampleSource`, bridge, frames exist | [`event_source.py`](spikeforge/events/event_source.py), [`event_bridge.py`](spikeforge/events/event_bridge.py) |
| ONNX | Absent | `setup.py` |
| `nirtorch` extraction | Probe exists, extraction deferred | [`api.py`](spikeforge/nir_bridge/api.py:63) |
| Quantization | Declared in constraints only | [`target_spec.py`](spikeforge_targets/target_spec.py:28) |
| Geometry | 28x28 / square conv math hardcoded | [`presets.py`](spikeforge/topology/presets.py:128), [`datasets.py`](spikeforge/data/datasets.py:73) |
| Hidden-layer animation | Raster snapshots only | [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md:21) |

### 1.1 Invariants that must not break

- The image training path is unchanged; event training is additive behind the
  `events` extra ([`datasets.py`](spikeforge/data/datasets.py:51)).
- ONNX and extraction are opt-in extras; their absence is reported, never raised
  at import ([`probe.py`](spikeforge_targets/probe.py:1) pattern).
- Quantization is only **applied** where the target declares support; otherwise
  it is reported as unapplied.
- Existing presets remain exactly 28x28 by default; geometry changes are
  opt-in via an explicit shape parameter.
- Existing WebSocket payload keys stay additive.

---

## 2. F1 — Event-dataset training

**Gap.** `build_dataset` refuses an event spec because it has no torchvision
class ([`datasets.py`](spikeforge/data/datasets.py:111)), so training never
reaches the event modality even though inference does.

**Design.** A thin event training engine that composes the existing pieces:

```
spikeforge/training/event_engine.py   EventTrainingEngine: steps over EventSampleSource
spikeforge/training/event_batches.py  batch an event stream into [T,B,...] via the bridge
```

- `event_batches` uses [`EventSpikeBridge`](spikeforge/events/event_bridge.py)
  to produce the same `[T, B, F]`/`[T, B, C, H, W]` contract the simulator
  already consumes, so the loss/optimizer loop reuses
  [`training_engine.py`](spikeforge/training/training_engine.py) unchanged.
- `TrainingEngine` learns a modality-aware dataset step: image modality uses
  `build_loader`, event modality uses `event_batches`; the rest of the loop,
  metrics, and checkpointing are shared.
- Honesty: when `tonic` is absent the engine raises the typed
  `EventsExtraMissingError` ([`event_errors.py`](spikeforge/data/event_errors.py)),
  and a synthetic event stream is never presented as a recording
  (see the `origin` convention in [`event_source.py`](spikeforge/events/event_source.py)).

**Acceptance:** a tiny event fixture trains for one epoch end to end and records
`modality: event` in checkpoint `meta`; the image path is unchanged.

---

## 3. F2 — ONNX export/import bridge

**Design.** An isolated `onnx_bridge/` package, the only module that imports
`onnx`/`onnxruntime`, behind the `onnx` extra:

```
spikeforge/onnx_bridge/
  __init__.py
  api.py           isolated onnx/onnxruntime probe + import helpers
  export.py        export the built module's forward graph to ONNX
  import_onnx.py   import an ONNX graph and map it to a topology spec
  errors.py        typed unsupported-op errors
```

- **Export** takes the built [`StageModule`](spikeforge/topology/stage_module.py)
  and traces a single step to ONNX, emitting metadata that names the topology
  and the temporal contract (the ONNX graph is one step; the loop stays in the
  simulator). This mirrors the "declarative-first" philosophy: the module is one
  rendering, ONNX is another.
- **Import** maps ONNX ops (`Gemm`, `Conv`, `MatMul`, `Relu`, ...) to stage
  kinds where a faithful mapping exists and raises a typed error naming any op
  it cannot map — never a silent partial.
- Honesty: an op with no SNN-equivalent is reported; an ONNX model with dynamic
  temporal behavior is explicitly out of scope and noted.

**Acceptance:** `conv_net` exports and re-imports to a graph whose summary
matches; an unsupported op raises a typed error naming it.

---

## 4. F3 — Third-party PyTorch extraction via `nirtorch`

**Gap.** [`api.py`](spikeforge/nir_bridge/api.py:63) probes `nirtorch` but
no extraction path is wired.

**Design.** `nir_bridge/extract.py` exposes
`extract(module) -> NIRGraph | UnsupportedNodeError`:

- Uses `nirtorch.extract_nir_graph` (isolated in
  [`api.py`](spikeforge/nir_bridge/api.py:1)) to lift an arbitrary
  `torch.nn.Module` into NIR.
- Any node `nirtorch` cannot map raises the existing typed
  `UnsupportedNodeError` in [`errors.py`](spikeforge/nir_bridge/errors.py),
  consistent with the ingest path ([`ingest.py`](spikeforge/nir_bridge/ingest.py:1)).
- The extracted graph is then runnable by the reference interpreter and
  classifyable by the capability matrix, so a third-party PyTorch model can be
  inspected and deployed like any other.

**Acceptance:** a small hand-built `nn.Sequential` extracts to NIR and
interprets; an unsupported node raises the typed error naming it.

---

## 5. F4 — Target quantization application

**Gap.** Targets declare `quantization` in constraints
([`catalog.py`](spikeforge_targets/catalog.py:51)) but nothing applies it.

**Design.** `targets/quantize.py` applies a target's declared quantization to a
built module's weights:

- Supported schemes (seed): `none` (no-op), `weight_int8`, `weight_uint8`
  (per-tensor symmetric/asymmetric, matching the declared constraint strings in
  [`catalog.py`](spikeforge_targets/catalog.py:86)).
- Applied **only** where the target declares support; a target that declares
  `none` gets a no-op report; an unknown scheme is reported unapplied, never
  guessed.
- A quantization report lists per-layer before/after ranges and the drift this
  induces, reusing the drift machinery
  ([`drift.py`](spikeforge/nir_bridge/drift.py)).
- The rewrite/backend pipeline (WS-B) can call `quantize` after substitution and
  before compile, so a deployment is quantized exactly when its target requires
  it.

**Acceptance:** quantizing `conv_net` for `lava_loihi2` clamps weights to int8
and reports the induced drift; a `none`-quantization target reports a no-op; an
unknown scheme is reported unapplied.

---

## 6. F5 — Geometry and client animation

### 6.1 Non-square sensor geometry

The transform hardcodes 28x28 ([`datasets.py`](spikeforge/data/datasets.py:73))
and `conv_net` derives its feature size from a square side
([`presets.py`](spikeforge/topology/presets.py:128)).

**Design:** allow an explicit `(height, width)` shape:

- `datasets.transform(size=(28, 28))` takes a shape; the default is unchanged.
- Presets accept `input_size` as `int` (square, unchanged) or `(h, w)`; the
  feature-size math becomes `channels * (h/4) * (w/4)`.
- `input_shape` ([`input_shape.py`](spikeforge/simulator/input_shape.py))
  and the `SampleSource` size accessor propagate the tuple.
- Defaults keep every shipped preset byte-identical.

**Acceptance:** a `(32, 28)` input builds and runs through `conv_net`; the
default path is unchanged.

### 6.2 Per-step hidden-layer animation

**Gap.** `INTEGRATION_PLAN.md` deliberately deferred per-step hidden-layer
*animation* ([`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md:21)); only rasters are
streamed.

**Design:** extend the existing `spike_frame` stream
([`messages.py`](server/messages.py:124)) to optionally emit a hidden/output
frame per step during `run`, gated by an additive `EncodeConfig.animate_hidden`
flag. The client
([`NetworkActivity.tsx`](client/src/components/NetworkActivity.tsx)) already
renders activity frames; it gains a per-step playback mode driven by the stream.
Default stays off, so existing payloads and the `run` cost are unchanged.

**Acceptance:** with `animate_hidden` set, hidden frames stream per step and the
client animates them; with it unset, the payload stream is identical to today.

---

## 7. Phases, deliverables, acceptance

| Phase | Deliverables | Acceptance |
|---|---|---|
| F1 Event training | `training/event_engine.py`, `training/event_batches.py`, modality routing in `training_engine.py` | one-epoch event fixture trains; `meta.modality == "event"`; image path unchanged |
| F2 ONNX bridge | `onnx_bridge/{__init__,api,export,import_onnx,errors}.py`; `onnx` extra | `conv_net` exports and re-imports; unsupported op raises typed error |
| F3 `nirtorch` extraction | `nir_bridge/extract.py` | `nn.Sequential` extracts + interprets; unsupported node raises typed error |
| F4 Quantization | `targets/quantize.py` | int8 applied with drift report; `none` is a no-op; unknown scheme reported |
| F5 Geometry + animation | `datasets.py`, `presets.py`, `input_shape.py`, `messages.py`, `EncodeConfig`, `NetworkActivity.tsx` | non-square runs; default unchanged; animation opt-in streams frames |

---

## 8. Risks and deferred items

- **ONNX temporal semantics:** ONNX has no native SNN time loop; the bridge is
  explicitly one-step + external loop, and mismatches are reported.
- **`nirtorch` coverage:** arbitrary modules may contain unmappable nodes; the
  typed error names them rather than silently truncating the graph.
- **Quantization fidelity:** applied quantization changes numerics; the report
  quantifies the drift so it is never mistaken for free.
- **Deferred:** event-driven training (WS-D is inference-only), ONNX *training*
  graphs, multi-precision quantization (int4), and non-uniform sensor
  downsampling.

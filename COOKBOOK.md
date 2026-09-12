# spikeforge cookbook

Practical, copy-pasteable recipes for every shipped capability, organised by
goal. Each recipe says **what it is for**, gives the **exact command(s)** (or a
short Python snippet), shows the **expected output/shape**, and lists the
**caveats**.

Every command below was executed against the repository at version `0.1.0`
with the project virtualenv (`venv/`). Outputs are trimmed to the fields that
matter; `…` means "more keys follow". Where an optional SDK is not installed
here, the recipe shows the *honest* output the tool prints and the install line
that would change it.

The reasoning behind the deliberate boundaries (estimated energy, single-step
ONNX, simulation-only attention, and so on) is in the README's
[Implications and boundaries](documentation/implications-and-boundaries.md) section.

---

## 0. Conventions

- **Interpreter.** All Python commands use the project venv: `venv/bin/python`.
  If you activated the venv, plain `python` works too.
- **Console scripts.** Packaging installs one script per surface
  (`venv/bin/spikeforge-verify`, `spikeforge-records`, `spikeforge-targets`, `spikeforge-hub`,
  `spikeforge-energy`, `spikeforge-benchmark`). Several of these are *only* reachable as
  console scripts or as subcommands of `spikeforge-verify` — see the note in
  [§12.1](#121-which-entry-point-to-use).
- **JSON everywhere.** Every CLI command prints indented JSON.
- **Exit codes are CI gates.** `validate`, `deploy`, `roundtrip`, `run`,
  `onnx-roundtrip`, `hub import`, and `benchmark --fail-on-regression` exit
  non-zero when their result is negative, so they can gate a pipeline.
- **Offline-safe.** The recipes use cached datasets (`build/`) and synthetic
  fixtures; none requires network access except the optional hub download and
  the (optional) live Hugging Face search.

---

## 1. Environment setup

### 1.1 Install the package and the extras you need

**What it's for.** The core install is deliberately lean; every capability
beyond it is an opt-in extra with an isolated probe, so an absent package is
*reported*, never raised at import.

**Command.**

```bash
# Core (torch, torchvision, snntorch, matplotlib, Pillow, numpy)
pip install -e ./packages/spikeforge

# The dashboard / WebSocket server (pulls core + the FastAPI stack)
pip install -e ./packages/spikeforge-server

# A representative "everything" install
pip install -e "./packages/spikeforge[dev,nir,events,onnx,norse,tracking,docs]"
pip install -e ./packages/spikeforge-hub
pip install -e ./packages/spikeforge-server

# Development (pytest, pytest-cov, ruff, jsonschema)
pip install -e "./packages/spikeforge[dev]"
```

**Extras matrix** (declared in
[`packages/spikeforge/pyproject.toml`](packages/spikeforge/pyproject.toml)):

The dashboard/WebSocket server is not a core extra; it is the
`packages/spikeforge-server` distribution. The model hub is its own
distribution too — `packages/spikeforge-hub` (import root `spikeforge_hub`) — and its
`huggingface_hub` dependency is a base dependency of that distribution rather
than a core `hub` extra.

| Extra | Packages | Enables | If absent |
|---|---|---|---|
| `nir` | `nir`, `nirtorch` | NIR export, interpretation, `nirtorch` extraction | typed unavailable error |
| `events` | `tonic` | Tonic event datasets (N-MNIST, DVS128 Gesture, CIFAR10-DVS, SSC) + event training | datasets reported unavailable; `EventsExtraMissingError` |
| `onnx` | `onnx`, `onnxruntime` | ONNX export/import bridge | typed unavailable error |
| `norse` | `norse` | real Norse simulator backend | `norse` target `available: false` |
| `lava` | `lava-nc` | Lava/Loihi 2 backend path | `lava_loihi2` target `available: false` |
| `tracking` | `tensorboard` | TensorBoard sink | local manifest remains the default |
| `tracking-wandb` | `wandb` | Weights & Biases sink | local manifest remains the default |
| `docs` | `mkdocs-material` | the generated docs site | `build_docs.sh` reports the gap |

> **Not executed here.** The installs above need network access and are shown
> for reference; `nir`, `events`, `onnx`, and `docs` were already present in
> this environment, while `norse`, `lava`, and `tracking` were intentionally
> absent so the recipes below can show the honest degradation. Live hub search
> reports unavailable whenever `huggingface_hub` cannot be imported.

### 1.2 Check what is actually available

**What it's for.** Confirm which optional capabilities resolved, without
importing a single optional package yourself.

**Command.**

```bash
venv/bin/python -c "import sys; print(sys.version)"
venv/bin/python -m spikeforge.cli.verify targets        # backend SDKs
venv/bin/python -m spikeforge_hub.cli search fc         # huggingface_hub
venv/bin/python -c "from spikeforge.tracking import sinks; print(sinks.describe('tensorboard'))"
```

**Expected output (this environment).**

```
3.12.x …
# targets: reference available:true; norse/lava_loihi2/spinnaker2/speck/xylo available:false
# hub search: "available": false, "reason": "requires the `hub` extra (huggingface_hub)"
{'requested': 'tensorboard', 'active': False, 'reason': "sink 'tensorboard' backend is not installed"}
```

**Caveats.** A missing extra is always a *reason*, never an exception — that is
the point of the probe pattern.

---

## 2. Train an image model

### 2.1 Headless training (Python API)

**What it's for.** Train the fully-connected LIF network on an image dataset
without the dashboard.

**Command.**

```python
from spikeforge.training.training_engine import TrainingEngine

eng = TrainingEngine(
    dataset="mnist", hidden=32, epochs=1, num_steps=5,
    subset=128, batch_size=64, device="cpu",
)
last = None
for metrics in eng.train():
    last = metrics
print(sorted(last.keys()))
print("loss=%.4f train_acc=%.3f test_acc=%s" % (
    last["loss"], last["train_accuracy"], last["test_accuracy"]))
```

**Expected output.**

```
['epoch', 'loss', 'step', 'test_accuracy', 'total', 'train_accuracy']
loss=1.8707 train_acc=-1.000 test_acc=52.775
```

**Caveats.**
- The accuracy key is `train_accuracy` (not `accuracy`); `test_accuracy` is a
  **percentage** and is `null` until an evaluation step runs.
- `train_acc=-1.000` is the sentinel when the metric is not yet computed for
  that step, not a real score.
- Downloading MNIST happens on first use; it is cached under `SPIKEFORGE_DATA_DIR`
  (default `build/`).
- Dataset normalisation to 28×28 is shared across all image datasets, so one
  architecture fits MNIST, Fashion-MNIST, KMNIST, QMNIST, USPS, EMNIST, and
  (grayscaled) CIFAR-10.

### 2.2 Training from the dashboard

**What it's for.** Interactive training with live loss/accuracy charts, model
management, and predictions.

**Command.**

```bash
# terminal 1 — FastAPI + WebSocket server on :8877
venv/bin/python -m server

# terminal 2 — Vite dev server (proxies /ws to :8877)
cd client && npm install && npm run dev      # opens http://localhost:5173
```

Or run both together:

```bash
scripts/dev.sh dev
```

Health check (from another terminal):

```bash
curl -fsS http://127.0.0.1:8877/health
```

**Expected output.**

```
{"status":"ok"}
```

**Caveats.** The server can also serve a prebuilt client on a single port
(:8877) — see the README's Docker section. The Vite proxy target must match the
server port (`client/vite.config.ts`).

---

## 3. Train an event model

**What it's for.** Train on an event (neuromorphic) dataset. N-MNIST and the
other event datasets load through Tonic; when Tonic is absent (or you ask for
it explicitly) the loader serves a **labelled synthetic** stream.

**Command (synthetic fallback — offline, no downloads).**

```python
from spikeforge.events.event_source import EventSampleSource
from spikeforge.events.event_bridge import EventSpikeBridge
from spikeforge.topology.registry import build_topology
from spikeforge.training.event_engine import EventTrainingEngine

src = EventSampleSource("n_mnist", synthetic_only=True)
print(src.origin, "|", src.description)
sample, label = src.load(0)
print("sensor:", sample.shape, "label:", label)
spec, module = build_topology("fc_legacy", {"num_classes": 10})
spikes, meta = EventSpikeBridge().encode(sample, spec)
print("bridged:", tuple(spikes.shape))

eng = EventTrainingEngine(dataset="n_mnist", synthetic_only=True, hidden=32,
                          epochs=1, num_steps=10, subset=32, batch_size=16,
                          device="cpu")
for metrics in eng.train():
    pass
print("loss=%.4f test_acc=%s" % (metrics["loss"], metrics["test_accuracy"]))
```

**Expected output.**

```
synthetic | synthetic n_mnist events on a 28x28 sensor (offline; no real recording)
sensor: (28, 28) label: 0
bridged: (10, 1, 784)
loss=2.3026 test_acc=12.5
```

**Caveats.**
- On the synthetic stream the accuracy is near chance (≈10–12% for ten
  classes); it is a wiring demo, not a benchmark.
- Without `synthetic_only=True`, a missing Tonic raises the typed
  `EventsExtraMissingError` instead of passing a synthetic stream off as a
  recording. Install the extra with
  `pip install -e "./packages/spikeforge[events]"`.
- `conv_net` needs a square single/dual-channel sensor matching its declared
  side; a feature-input topology (`fc_legacy`, `fc_small`, `recurrent_net`)
  accepts the flattened sensor. A mismatch raises `EventGeometryError` naming
  both sides.

---

## 4. Encode and inspect spikes

### 4.1 Encode one image into rate/latency/delta (+ reconstruction)

**What it's for.** See each spike coding, its firing rate/sparsity, and the
(explicitly approximate) reconstruction.

**Command.**

```python
from spikeforge.data.sample_source import SampleSource
from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.introspection.encoding import encoding_report

img = SampleSource(dataset="mnist").image(0)
for coding in ("rate", "latency", "delta", "random"):
    rep = encoding_report(img, SpikeEncoder(coding=coding, num_steps=10))
    print(coding, "firing_rate=%.4f" % rep["firing_rate"],
          "sparsity=%.4f" % rep["sparsity"],
          "reconstruction_supported:", rep["reconstruction_supported"])
```

**Expected output.**

```
rate    firing_rate=0.1366 sparsity=0.8634 reconstruction_supported: True
latency firing_rate=0.1000 sparsity=0.9000 reconstruction_supported: True
delta   firing_rate=0.1378 sparsity=0.8622 reconstruction_supported: True
random  firing_rate=0.2532 sparsity=0.7468 reconstruction_supported: False
```

**Caveats.** The `rate` encoder is a stochastic Bernoulli sampler, so its
numbers vary run to run. Reconstructions are approximate and say so in the
report's `approximation` field: latency is quantised to integer steps and
saturates for sub-threshold pixels; delta is a lower bound exact only when each
step rises by exactly the threshold; `random` carries no image signal, so
`reconstruction_supported` is `False` and `reconstruction` is `null`.

### 4.2 Run the simulator and read trajectory metrics

**What it's for.** Execute a topology over a spike train in **educational**
mode (which records `U[t]`/`I[t]`/`S[t]`) and gather per-stage metrics.

**Command.**

```python
from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.data.sample_source import SampleSource
from spikeforge.introspection.metrics import trajectory_metrics
from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.simulator import input_shape
from spikeforge.simulator.runner import run
from spikeforge.topology.registry import build_topology

img = SampleSource(dataset="mnist").image(0)
spikes = SpikeEncoder(coding="rate", num_steps=10).encode_image(img)
spec, module = build_topology("fc_legacy", {"num_classes": 10})
traj = run(module, input_shape.to_input_shape(spikes, spec),
           mode=ExecutionMode.EDUCATIONAL)
print("logits:", tuple(traj.logits.shape), "spike stages:", list(traj.spikes))
print("metric stages:", list(trajectory_metrics(traj).keys()))
```

**Expected output.**

```
logits: (1, 10) spike stages: ['_lif1', '_lif2']
metric stages: ['steps', 'stages']
```

**Caveats.** Production mode records nothing (the difference is recording
overhead, not behaviour); in production the trace dicts are empty. Metrics
include firing rate, sparsity, ISI stats, and a per-neuron histogram — all
plain JSON types (use `trajectory_metrics(...)["stages"][stage]` to reach them).

---

## 5. NIR: export, validate, round-trip, ingest, extract

### 5.1 Export a topology and validate drift

**What it's for.** Render a `TopologySpec` into a `nir.NIRGraph` and prove the
independent interpreter agrees with snnTorch. Both are CI gates.

**Command.**

```bash
venv/bin/python -m spikeforge.cli.verify export   --topology conv_net
venv/bin/python -m spikeforge.cli.verify validate --topology conv_net
venv/bin/python -m spikeforge.cli.verify validate --topology sequence_mlp
```

**Expected output (validate, trimmed).**

```json
{ "…": "…",
  "within_tolerance": true,
  "readout": { "max_abs": 0.0, "agreement": 1.0 },
  "worst": { "layer": "lif1", "quantity": "membrane",
             "metric": "mean_abs", "value": 1.6e-07 } }
```

`validate` exits `0` when `within_tolerance` is true. The membrane residual you
see is the documented Euler-vs-zero-order-hold difference, reported rather than
hidden.

**Caveats.** `export --out FILE` writes the **graph summary** (node/edge
inventory), *not* a re-loadable graph — see §5.3 for the ingestable form.

### 5.2 Sequence experiments (exportable vs simulation-only)

**What it's for.** `sequence_mlp` is built only from NIR-mappable kinds and
validates end to end; `sequence_attn` uses embedding/attention/normalisation
kinds the installed `nir` cannot represent.

**Command.**

```bash
venv/bin/python -m spikeforge.cli.verify validate --topology sequence_mlp
venv/bin/python -m spikeforge.cli.verify export   --topology sequence_attn
```

**Expected output.**

```
# sequence_mlp: exit 0, "within_tolerance": true
# sequence_attn: exit 1
{
  "error": "stage kind 'embedding' cannot be mapped to NIR: the installed nir has no Embedding primitive, so a token lookup table cannot be represented; the stage is simulation-only",
  "kind": "embedding"
}
```

**Caveats.** `sequence_attn` stays available for simulation and introspection
only; the typed `UnsupportedStageError` names the first unexportable stage.

### 5.3 Persist a graph, round-trip it, and ingest an external graph

**What it's for.** `roundtrip` persists a graph in a version-stamped JSON
envelope, reloads it, and compares — `identical: true` only on zero-error
structural equality. `ingest` runs an external envelope on the NIR interpreter.

**Command.**

```bash
# persist + reload + compare (also writes a loadable envelope with --out)
venv/bin/python -m spikeforge.cli.verify roundtrip --topology conv_net \
    --out build/graph_envelope.json

# ingest the very file roundtrip wrote
venv/bin/python -m spikeforge.cli.verify ingest --file build/graph_envelope.json
```

**Expected output.**

```json
// roundtrip
{ "identical": true, "steps": 8, "readout": { "max_abs": 0.0, "agreement": 1.0 } }
// ingest
{ "path": "build/graph_envelope.json", "topology": "conv_net",
  "steps": 8, "readout": [ 0.0, 0.0, "…" ],
  "spike_nodes": [ "lif1", "lif2", "out" ] }
```

**Caveats.** `ingest` expects the version-stamped envelope (`{"format":
"spikeforge-nir-graph", …}`), **not** the `export --out` summary; feeding
it the summary fails with `malformed graph file`. To build an envelope in
Python:

```python
from spikeforge.nir_bridge import to_nir, save_graph
from spikeforge.topology.registry import build_topology

spec, module = build_topology("conv_net")
save_graph(to_nir(spec, module), "build/api_graph.json")
```

### 5.4 Extract a third-party `nn.Module` into NIR

**What it's for.** Lift an arbitrary PyTorch module into NIR through `nirtorch`
and run it on the independent interpreter.

**Command.**

```python
import torch
from torch import nn
torch.save(nn.Sequential(nn.Flatten(), nn.Linear(784, 10)), "build/lin.pt")
```

```bash
venv/bin/python -m spikeforge.cli.verify extract --module build/lin.pt
```

**Expected output (trimmed).**

```json
{ "nodes": [ {"name": "input_1", "kind": "Input"},
             {"name": "_0", "kind": "Flatten"},
             {"name": "_1", "kind": "Affine"},
             {"name": "output", "kind": "Output"} ],
  "features": 784, "steps": 8, "runnable": true, "readout": [ "…" ] }
```

**Caveats.** Only `nn.Linear` and `nn.Flatten` are mapped; any other module
raises the typed `UnsupportedNodeError` naming the class — no silent
truncation. Requires the `nir` extra.

---

## 6. Model hub: browse, inspect, import

### 6.1 Browse and search the curated catalog

**What it's for.** The bundled catalog renders fully offline and ships **only
verified entries** (a real source plus a concrete license); live Hugging Face
search is provided by the `spikeforge-hub` distribution (`spikeforge_hub`). Adding an entry
requires a real repository or reference and a verified license — see
`spikeforge_hub/CURATION.md`. The
on-demand downloader stays fully available for any vetted repository you choose
to add.

**Command.**

```bash
venv/bin/spikeforge-hub list --framework nir
venv/bin/spikeforge-hub list --available
venv/bin/spikeforge-hub search fc --limit 5
```

**Expected output (search, trimmed).**

```json
{ "query": "fc", "limit": 5,
  "available": false,
  "reason": "requires the `hub` extra (huggingface_hub)",
  "results": [ { "id": "nir/fc_legacy", "framework": "nir",
                 "kind": "nir_graph", "license": "BSD-3-Clause",
                 "available": true, "cached": true }, "…" ] }
```

**Caveats.** `available: false` here is about *live HF search*, not the curated
catalog (which is always browsable). A catalog entry whose license is the
`"unverified-candidate"` marker is also reported `available: false` with a named
reason, so an unverified candidate is never presented as ready.

### 6.2 Download a bundled/remote artifact

**What it's for.** Materialize an artifact into the offline cache
(`SPIKEFORGE_HUB_DIR`, default `build/hub`).

**Command.**

```bash
venv/bin/spikeforge-hub download nir/conv_net
```

**Expected output.**

```json
{ "id": "nir/conv_net", "source": "bundled",
  "status": "unverified", "verified": false,
  "reason": "bundled artifact; materialization is deferred to import",
  "sha256": null, "size_bytes": 0 }
```

**Caveats.** A bundled graph publishes no checksum, so it verifies as
`unverified` rather than trusted — the honest label, not a failure. `download`
exits non-zero only when `status == "failed"` (unless `--no-verify`).

### 6.3 Inspect → compat → promote

**What it's for.** Run the three-gate import funnel: structural inspect,
compatibility verdict, then weight load + drift check + promote into
`MODEL_DIR`.

**Command.**

```bash
venv/bin/spikeforge-hub inspect nir/fc_legacy
venv/bin/spikeforge-hub import  nir/fc_legacy --topology fc_legacy
```

**Expected output (import, trimmed).**

```json
{ "id": "nir/fc_legacy", "kind": "nir_graph",
  "verdict": { "verdict": "exact", "topology": "fc_legacy", "mismatches": [] },
  "promoted": true,
  "destination": "build/models/hub_nir_fc_legacy.pt",
  "weights": { "loaded": true, "missing": [], "unexpected": [] },
  "validation": { "within_tolerance": true } }
```

**Caveats.** `import` exits non-zero on an `incompatible` verdict (printing the
named mismatches). A NIR-only artifact with no preset match is still runnable
through the reference interpreter, so import is useful without a weight mapping.

### 6.4 Hub from the dashboard

Use the **HubPanel** in the browser (`venv/bin/python -m server`). It drives
six WebSocket actions — `hub_list`, `hub_search`, `hub_download`, `hub_cancel`,
`hub_inspect`, `hub_import` — and renders entry cards, a compat badge, a
progress/cancel row, and an inline verdict that names every mismatch.

---

## 7. Deployment reports, substitutions, and execution

### 7.1 Capability report (`deploy`)

**What it's for.** Classify every node of a topology against a target into
`supported` / `substituted` / `unsupported`, with constraints and an optional
drift section. `deployable` is true only when the target is available *and* has
zero unsupported nodes.

**Command.**

```bash
venv/bin/spikeforge-verify deploy --topology conv_net --target reference   # exit 0
venv/bin/spikeforge-verify deploy --topology conv_net --target xylo        # exit 1
```

**Expected output (xylo, trimmed).**

```json
{ "…": "…",
  "deployable": false,
  "notes": [ "target SDK is not installed; run the enabling extra",
             "validation compares snnTorch against the exported graph" ] }
```

### 7.2 Apply substitutions (`rewrite`)

**What it's for.** Apply a target's **declared** substitutions and report what
changed (`applied` / `skipped` / `unfixable`) plus a post-rewrite drift check.
An unfixable primitive is named, never dropped.

**Command.**

```bash
venv/bin/spikeforge-verify rewrite --topology conv_net --target norse
```

**Expected output (trimmed).**

```json
{ "target": "norse", "applied": [], "skipped": [],
  "unfixable": [ {"node": "lif1__reset_delay", "primitive": "Delay",
                  "reason": "no substitution declared"}, "…" ],
  "ready": false, "counts": { "applied": 0, "skipped": 0, "unfixable": 3 },
  "drift": { "within_tolerance": true } }
```

**Caveats.** Two rules ship — `IF`→`beta=0` `LIF` for `norse` and
`AvgPool2d`→`SumPool2d`+`Scale` for `lava_loihi2`. `conv_net`'s neurons are
already LIF, so the `IF` rule does not fire here; the reset `Delay` nodes have
no declared substitution and are reported `unfixable`, leaving `ready: false`.
`rewrite` itself always exits `0` (its report is informational).

### 7.3 Execute (`run`)

**What it's for.** Rewrite → (optionally) quantize → gate on availability →
compile → run → compare to the reference interpreter.

**Command.**

```bash
venv/bin/spikeforge-verify run --topology conv_net --target reference   # exit 0
venv/bin/spikeforge-verify run --topology conv_net --target norse       # exit 1
venv/bin/spikeforge-verify run --topology conv_net --target lava_loihi2 # exit 1
```

**Expected output (norse, trimmed).**

```json
{ "target": "norse", "status": "unavailable", "steps": 0,
  "notes": [ "install the 'norse' extra to run the 'norse' target" ],
  "compare": null }
```

**Caveats.** `run` exits non-zero unless `status == "ok"` *and* the comparison
is within tolerance. `norse` and `lava_loihi2` execute once their extras are
installed; `spinnaker2`, `speck`, and `xylo` remain declarative placeholders.
No physical device is attached, so no hardware timing is measured.

---

## 8. Target quantization

**What it's for.** Restrict a graph's weight tensors to a target's declared
scheme and read the per-layer before/after ranges plus the induced drift.

**Command.**

```bash
venv/bin/spikeforge-verify run --topology conv_net --target lava_loihi2 | \
  python -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['quantization'], indent=2))"
```

**Expected output (trimmed).**

```json
{ "target": "lava_loihi2", "scheme": "weight_int8", "applied": true,
  "layers": [ { "node": "conv1", "primitive": "Conv2d",
                "before": [ -0.3238, 0.3272 ], "after": [ -0.3247, "…" ] }, "…" ],
  "counts": { "layers": "…" }, "drift": { "…": "…" } }
```

With `--target reference` (scheme `none`) the block is an honest no-op:

```json
{ "target": "reference", "scheme": "none", "applied": false,
  "reason": "target declares no quantization", "layers": [] }
```

**Caveats.** Weight-level only: no activation/membrane quantization, no
calibration dataset, no integer accumulation/saturation, no per-channel
schemes, and no device kernel. The source graph is never mutated, and
executing a quantized graph still needs the target SDK. An unknown scheme is
reported unapplied.

---

## 9. Energy and latency accounting

### 9.1 Sparse vs dense (`spikeforge-energy`)

**What it's for.** Count SOP/MAC/AC and timesteps, then map them onto a
per-target declared cost table. Without `--sparse` the report is the dense
baseline (`sop == mac`); with it, the event-driven reduction.

**Command.**

```bash
venv/bin/spikeforge-energy account --topology conv_net --target reference --sparse
venv/bin/spikeforge-energy account --topology conv_net --target reference            # dense
venv/bin/spikeforge-energy report  --topology conv_net --target reference --sparse   # + parity
```

**Expected output (sparse, trimmed).**

```json
{ "target": "reference", "estimate": true, "basis": "declared cost table",
  "timesteps": 8,
  "ops": { "sop": 725112, "mac": 4641280, "ac": 351552 },
  "efficiency": { "sop_over_mac": 0.1562 },
  "energy": { "total_pj": 3977112.0, "dense_pj": 23557952.0 },
  "latency": { "step_ns": 2000.0, "total_ns": 16000.0 },
  "measured": null,
  "notes": [ "estimate only; no device measured",
             "declared cost source: Declared in-repo order-of-magnitude estimate …" ] }
```

Dense run: `"sop": 4641280` and `"sop_over_mac": 1.0`. `report` adds a
`comparison` block proving the sparse and dense readouts agree within tolerance.

**Caveats.** Every number is an **estimate** from a `"measured": false` table,
valid for comparing models/targets and sparse-vs-dense trade-offs, **not** for
power budgets. A target with no table reports `basis: "unavailable"` and `null`
numbers rather than a fabricated figure.

### 9.2 Energy inside a benchmark

```bash
venv/bin/python -m spikeforge.benchmark --topology fc_small --steps 4 \
    --repeats 1 --energy --energy-target reference
```

The report's per-mode `energy` block carries the same `report` payload, so the
timing and the estimate travel together.

### 9.3 Reading `estimate` vs `measured`

- `estimate: true`, `basis: "declared cost table"` — the normal case.
- `estimate: true`, `basis: "unavailable"` — no table for that target.
- `estimate: false`, `basis: "device measurement"` — only when a device probe
  or an explicit `measurement=` reports one. There is a single integration
  point (`account(source, target, measurement=...)`) that flips this label, so
  adding a real device changes nothing else.

---

## 10. ONNX export / import / round-trip

**What it's for.** Export a topology's **single forward step** with spec
metadata, re-import it exactly, and (for foreign files) map a limited op set or
fail by name.

**Command.**

```bash
venv/bin/spikeforge-verify onnx-export    --topology conv_net --out build/model.onnx
venv/bin/spikeforge-verify onnx-import    --file build/model.onnx
venv/bin/spikeforge-verify onnx-roundtrip --topology conv_net
```

**Expected output (export / roundtrip, trimmed).**

```json
{ "path": "build/model.onnx", "topology": "conv_net", "opset": 17,
  "ops": [ "Add", "AveragePool", "Cast", "Clip", "Constant", "Conv",
           "Flatten", "Gemm", "Greater", "Identity", "Mul", "Sub" ],
  "nodes": 51, "input_shape": [ 2, 1, 28, 28 ],
  "temporal": "single-step; the time loop stays in the simulator" }

// roundtrip: { "source": "metadata", "identical": true, … }
```

Importing a **foreign** file (no metadata) maps only
`Gemm`/`MatMul`, `Conv`, `Flatten`, `AveragePool`, `Dropout`, and `Identity`:

```bash
# strip the bridge's metadata to simulate a third-party graph
venv/bin/python -c "import onnx; m=onnx.load('build/model.onnx'); [m.metadata_props.remove(p) for p in list(m.metadata_props)]; onnx.save(m,'build/model_nometa.onnx')"
venv/bin/spikeforge-verify onnx-import --file build/model_nometa.onnx
# exit 1: ONNX op 'Constant' has no faithful SNN stage mapping: no stage kind represents it
```

**Caveats.** An exported ONNX file is **not** a complete temporal SNN — the
time loop stays in the simulator, so another runtime will not reproduce
multi-timestep dynamics. Neuron-internal ops (`Greater`/`Sub`/`Clip`) have no
SNN stage mapping and are rejected by name. Requires the `onnx` extra.

---

## 11. Reproducibility, records, tracking, determinism

### 11.1 Save a checkpoint and search the registry (`spikeforge-records`)

```python
from spikeforge.network import model_store
from spikeforge.topology.registry import build_topology
spec, module = build_topology("fc_legacy", {"num_classes": 10})
model_store.save("cookbook_demo", module,
    {"dataset": "mnist", "topology": "fc_legacy", "coding": "rate",
     "input_mode": "rate", "device": "cpu"})
```

```bash
venv/bin/spikeforge-records list --dataset mnist --topology fc_legacy
venv/bin/spikeforge-records manifest cookbook_demo
```

**Expected output (manifest, trimmed).**

```json
{ "name": "cookbook_demo", "available": false,
  "manifest": { "available": false,
                "reason": "no manifest stored (legacy checkpoint)",
                "meta": { "dataset": "mnist", "topology": "fc_legacy" } } }
```

**Caveats.** `manifest` reports `available: false` for a checkpoint saved
without a reproducibility manifest (a "legacy" card) rather than erroring. A
trained checkpoint written by the training engine does carry a manifest.

### 11.2 Config hash, seeding, determinism

```python
from spikeforge.tracking.config_hash import config_hash
from spikeforge.tracking.determinism import enable_deterministic

print(config_hash({"a": 1, "b": [2, 3]}) == config_hash({"b": [2, 3], "a": 1}))  # True
print(enable_deterministic(seed=0).to_dict())
```

**Expected output.**

```
True
{'enabled': True, 'seed': 0, 'applied': {'python': True, 'numpy': True,
 'torch': True, 'cublas_workspace': True, 'cudnn_deterministic': True, …},
 'notes': ['torch.use_deterministic_algorithms(warn_only=True) is set',
           'hardware thread scheduling is reported, not enforced']}
```

**Caveats.** The hash is order-independent by construction (canonical JSON).
Determinism *narrows* the bit-exactness gap; it does not close it.

### 11.3 Tracking sinks

```python
from spikeforge.tracking import sinks
print(sinks.describe("tensorboard"))
print(sinks.describe("wandb"))
```

**Expected output (with the extras absent).**

```
{'requested': 'tensorboard', 'active': False, 'reason': "sink 'tensorboard' backend is not installed"}
{'requested': 'wandb', 'active': False, 'reason': "sink 'wandb' backend is not installed"}
```

**Caveats.** The **local manifest is always written first**; an absent tracker
is a recorded `reason`, not an error. Install with `pip install -e
".[tracking]"` (TensorBoard) or `".[tracking-wandb]"` (Weights & Biases).
Select a sink via `TrainConfig.tracking` (`tensorboard` / `wandb`, default
`null`).

### 11.4 Persisted metrics (opt-in)

```bash
SPIKEFORGE_METRICS_PERSIST=1 SPIKEFORGE_METRICS_DIR=build/cookbook_metrics venv/bin/python - <<'PY'
from spikeforge.observability import metrics, persistence
metrics.counter("train.steps", 3)
print(metrics.snapshot())
print(persistence.flush("cookbook").run_id)
print(persistence.status())
PY
```

**Expected output (trimmed).**

```
{'counters': {'train.steps': 3.0}, 'gauges': {}, 'timers': {}}
cookbook
{'enabled': True, 'root': 'build/cookbook_metrics', 'run_id': 'default', 'last_flush': …}
```

**Caveats.** With `SPIKEFORGE_METRICS_PERSIST` unset, `flush()` is a no-op and
behaviour is unchanged. The registry is per-process and in-memory; persistence
does not aggregate across workers.

---

## 12. Benchmarks and the docs site

### 12.1 Which entry point to use

Two entry points exist and they are **not** always interchangeable:

| Need | Works | Does **not** work |
|---|---|---|
| verify family (export/validate/targets/deploy/rewrite/run/roundtrip/ingest/extract/onnx-*) | `venv/bin/spikeforge-verify …` or `venv/bin/python -m spikeforge.cli.verify …` | — |
| records | `venv/bin/spikeforge-records …` or `venv/bin/spikeforge-verify records …` | `python -m spikeforge.cli.records_cli` (no `__main__`; prints nothing) |
| targets/deploy/rewrite/run/extract | `venv/bin/spikeforge-targets …` or `venv/bin/spikeforge-verify …` | bare `python -m spikeforge_targets.cli.target_cli` (needs a subcommand) |
| hub | `venv/bin/spikeforge-hub …` or `venv/bin/python -m spikeforge_hub.cli …` | — |
| energy | `venv/bin/spikeforge-energy …` or `venv/bin/python -m spikeforge_targets.energy.cli …` | — |
| benchmark | `venv/bin/spikeforge-benchmark …` or `venv/bin/python -m spikeforge.benchmark …` | — |

### 12.2 Record, list, and compare benchmark runs

**What it's for.** Store one JSON record per run and catch regressions over
time. `--compare` takes a stored **run id** (from `--list`), not a label.

```bash
python -m spikeforge.benchmark --topology fc_small --steps 8 --repeats 2 \
    --save --label cookbook
python -m spikeforge.benchmark --list
python -m spikeforge.benchmark --compare <run-id> --against <run-id> --threshold 0.5
```

**Expected output (compare, trimmed).**

```json
{ "threshold": 0.5, "compared": 2,
  "cases": [ { "topology": "fc_small", "mode": "production",
               "metrics": [ { "metric": "ms_per_step", "baseline": 0.43,
                              "candidate": 0.78, "change_fraction": 0.81,
                              "regressed": true }, "…" ] } ] }
```

**Caveats.** Wall-time noise on a shared runner moves metrics a few percent, so
use a threshold (CI default 10%). `--fail-on-regression` turns a regression
into a non-zero exit. Add `--energy` to attach the energy estimate (§9.2).

### 12.3 Build the docs site

```bash
scripts/build_docs.sh          # build into build/docs
scripts/build_docs.sh --check  # fail on broken documentation links
```

**Expected output (tail).**

```
INFO    -  Documentation built in 0.52 seconds
==> checking documentation links
```

**Caveats.** Requires the `docs` extra. `docs/` is generated from `plans/` plus
the root README (and this cookbook and the open-source checklist) and is never
edited by hand.

---

## 13. Other tools

### 13.1 Encoding demos and media exports

```bash
venv/bin/python main.py             # rate pipeline -> MP4/GIF/PNG/rasters in build/
venv/bin/python main_encodings.py   # latency / delta / random demos
```

Both write into `build/` (gitignored). `ffmpeg` is only needed for MP4 output.

### 13.2 Developer script

```bash
scripts/dev.sh help          # list every command
scripts/dev.sh check         # ruff + pytest + client type-check + client build
scripts/dev.sh bench         # benchmark suite (spikeforge-benchmark)
scripts/dev.sh dev           # run the API and Vite dev server together
scripts/dev.sh data          # show the dataset cache and sizes
```

### 13.3 Docker

```bash
docker compose up --build                 # CUDA image, dashboard on :8877
docker compose --profile cpu up --build   # CPU-only torch (smaller image)
```

Only one profile can own port 8877 at a time.

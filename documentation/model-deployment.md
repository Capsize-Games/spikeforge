# Model deployment: bundles and modules

A trained checkpoint (`.pt`, under `SPIKEFORGE_MODEL_DIR`) is a training-time
artifact — weights plus a topology spec, meant for `spikeforge` itself to load
and keep training. A **deployment bundle** (`.spkf`) is the portable version:
everything a runtime needs to load and run the model, and nothing else. A
**module** is one of those bundles installed under a name, so it runs as its
own standalone command.

## Bundling a trained model

From the dashboard: **Model & Data → Model → Bundle**, pick a saved
checkpoint, and click **Download .spkf**. The browser downloads a real file —
`GET /api/bundle/<name>` builds it server-side from the checkpoint via
[`spikeforge.serving.bundle.build`](../spikeforge/serving/bundle.py) and
streams it back.

From Python:

```python
from spikeforge.serving.bundle import build

build("my_model", out="my_model.spkf")
```

A `.spkf` is a zip with a fixed entry set — `manifest.json` (topology,
encode config, label map, provenance), `weights.pt`, and a `SHA256SUMS`
integrity file — documented in full in
[`spikeforge/serving/bundle_manifest.py`](../spikeforge/serving/bundle_manifest.py).
It carries no training code, no optimizer state, and no dataset.

## Running a bundle

`spikeforge-serve` (a separate distribution: `pip install spikeforge-serve`)
loads a `.spkf` and runs it two ways:

```bash
# As an HTTP/WebSocket service (predict / stream / metrics):
spikeforge-serve serve --bundle my_model.spkf --port 8899

# One-shot, no server: read a JSON request, print a JSON response, exit.
echo '{"frames": [[...]], "encoded": true}' | spikeforge-serve run my_model.spkf
```

The one-shot `run` path and the HTTP `/v1/predict` route go through the same
[`ServingService`](../spikeforge_serve/service.py), so a bundle behaves
identically served or run standalone.

## Serving through the C runtime

Both paths step through an inference backend: `torch` (the default) runs
the bundle's torch module, and `nir-c` runs the bundle's NIR graph in the
[`nir-c-runtime`](https://github.com/Capsize-Games/nir-c-runtime) C
library instead, one frame per step with neuron state carrying across
steps inside C:

```bash
# Build the shared library once from the nir-c-runtime sources:
cc -shared -fPIC -O2 -std=c11 \
  -I<nir-c-runtime>/include <nir-c-runtime>/src/*.c \
  -o libnir_c_runtime.so -lm
export NIR_C_RUNTIME_LIB=$PWD/libnir_c_runtime.so

# One-shot and HTTP take the same switch (--c-lib overrides the env var):
spikeforge-serve run my_model.spkf --backend nir-c --file frame.json
spikeforge-serve serve --bundle my_model.spkf --backend nir-c
```

The HTTP service takes the same switch as an argument:
`ServingService(bundle, backend="nir-c")` or
`create_app(bundle, backend="nir-c")`. From Python,
`InferenceSession.load(bundle, backend="nir-c", c_lib="...")` opens a
stateful C session with the same `step`/`reset` streaming contract.

What matches and what does not:

- Per-step logits, cumulative totals, and labels match the torch path
  bit-exactly on the committed conv-net fixture (acceptance bar: 1e-3
  readout max-abs, the same bar `nir-c-runtime` uses in its own
  end-to-end test).
- A bundle whose topology the C runtime rejects fails before serving
  with the typed `CExportError` naming the unsupported node kinds —
  never a crash or a silent mismatch.
- The C runtime steps one sample at a time on CPU: frames keep the
  leading batch dimension but the batch must be 1, and `device` only
  places the returned readout.
- Per-stage spikes are not observable through the C API, so C-backed
  predictions report no spikes; carried-state capture is refused for
  the same reason.
- Loading the `.spkf` and exporting its graph still needs Python with
  torch and the `nir` extra — only the per-step inference runs in C.

## Installing a bundle as a module

Think of a module the way a small Unix tool or a Linux kernel module works:
one name, one job, a plain input and a plain output. `spikeforge-serve
install` registers a bundle under `~/.local/share/spikeforge/modules/<name>/`
and writes an executable `<name>` wrapper to `~/.local/bin/`:

```bash
spikeforge-serve install my_model.spkf --name digit-classifier
# installed 'digit-classifier' -> ~/.local/share/spikeforge/modules/digit-classifier

echo '{"frames": [[...]], "encoded": true}' | digit-classifier
```

`spikeforge-serve list` shows every installed module; `spikeforge-serve
uninstall <name>` removes one. Running by name (`spikeforge-serve run
digit-classifier`) resolves the installed copy first and falls back to
treating the argument as a literal `.spkf` path, so a bundle works whether or
not it has been installed.

Because each module reads one JSON object from stdin and writes one JSON
object to stdout, several installed models chain into a pipeline the same way
any Unix tools do — reshape between them with `jq`, no shared code required:

```bash
cat frame.json \
  | digit-classifier \
  | jq '{frames: [.predictions[0].mean_logits.values]}' \
  | risk-scorer
```

Neither module needs to know the other exists; the contract is just the JSON
shape on the pipe.

## Chaining checkpoints visually: the Pipeline tab

The shell-pipe chaining above works from a terminal; the dashboard's
**Pipeline** tab is the same idea as a visual, in-app DAG editor. Each node
is a saved checkpoint (the same list the Load/Bundle tabs show); each edge
picks one of three fixed ways to shape the source node's output into the
target's next input — `mean_logits` (the full readout vector, the common
case), `predicted_class` (a single scalar), or `one_hot` (sized to the
target's own `num_classes`). There's no custom mapping expression and no
fan-in (a node with two incoming edges is refused) — see
[`spikeforge_serve/pipeline.py`](../spikeforge_serve/pipeline.py) for the
exact rules.

A pipeline is a DAG, not a state machine: nodes run once each, in
topological order, with no conditional branching and no loops. Running one
spawns a background worker (mirroring how training runs — see
[`server/pipeline_service.py`](../server/pipeline_service.py)) that streams
a `pipeline_node_result` message as each node finishes, so the canvas shows
live idle → running → done/error status per node. Under the hood it's the
same [`ServingService`](../spikeforge_serve/service.py) every other
serving path uses, built on demand from each node's checkpoint via
`spikeforge.serving.bundle.build` — no install step, no bundle file ever
touches disk.

Saved pipelines live server-side (`SPIKEFORGE_DATA_DIR/pipelines/`,
overridable via `SPIKEFORGE_PIPELINES_DIR`), alongside saved checkpoints.

## What a module does and does not give you

- **No physical install** in the sense of a compiled binary — a module is a
  `.spkf` file plus a small shell wrapper that invokes `python -m
  spikeforge_serve run`. `spikeforge-serve` (and by extension `spikeforge`
  core, i.e. PyTorch) must be installed wherever the module runs.
- **No sandboxing** — an installed module runs with the invoking user's
  permissions, same as any other local script.
- **No network step** — `run`/`install`/`list`/`uninstall` are all local; the
  HTTP path (`serve`) is the only one that talks over a socket.

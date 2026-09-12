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

## What a module does and does not give you

- **No physical install** in the sense of a compiled binary — a module is a
  `.spkf` file plus a small shell wrapper that invokes `python -m
  spikeforge_serve run`. `spikeforge-serve` (and by extension `spikeforge`
  core, i.e. PyTorch) must be installed wherever the module runs.
- **No sandboxing** — an installed module runs with the invoking user's
  permissions, same as any other local script.
- **No network step** — `run`/`install`/`list`/`uninstall` are all local; the
  HTTP path (`serve`) is the only one that talks over a socket.

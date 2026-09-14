# Quickstart

Two supported install paths: from PyPI (no clone needed), or editable from
a clone for development.

### From PyPI

```bash
pip install spikeforge
```

```python
from spikeforge.training.training_engine import TrainingEngine

engine = TrainingEngine(dataset="mnist", hidden=32, epochs=1, num_steps=5)
for metrics in engine.train():
    last = metrics
print(last)
```

That's the core training API with no Docker and no dashboard. For the
deployment/NIR/energy CLIs and the dashboard:

```bash
pip install "spikeforge[all]"  # library bundle: core + targets + hub
pip install spikeforge-server  # the server, pulling core + targets + hub
```

```bash
spikeforge-verify --help       # NIR export/validate, deploy, records, ONNX
spikeforge-server              # the dashboard/WebSocket server on :8877
```

Open <http://localhost:8877> for the single-port build, or run the Vite dev
server for hot reload (`cd client && npm install && npm run dev`; its proxy
target is `:8877`).

### From a clone (one command)

```bash
./install.sh              # editable: core + targets + hub + server
./install.sh --no-server  # the library only (no dashboard/server)
./install.sh --dev        # add the [dev] extra to every distribution
```

`./install.sh` installs all four distributions in editable mode. Then:

```bash
spikeforge --help         # tutorial demo: train + export every visual
spikeforge-verify --help  # NIR export/validate, deploy, records, ONNX
spikeforge-server         # the dashboard/WebSocket server on :8877
```

`python -m server` remains an equivalent way to launch the server.

From a clone you can also run any of the fifteen example journeys directly
(training, encoding, NIR/ONNX, the hub, backends, energy, sequence
experiments, and reproducibility):

```bash
python examples/04_nir_export_validate.py
```

listed in [`examples/README.md`](../examples/README.md); the copy-pasteable
recipes are in [`COOKBOOK.md`](../COOKBOOK.md). `examples/` is not part of
the PyPI wheel, so running these scripts specifically needs the clone (the
inline snippet above under "From PyPI" does not).

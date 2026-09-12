# Quickstart

Two supported install paths: editable from a clone, or from PyPI once the
distributions are published.

### From a clone (one command)

```bash
./install.sh              # editable: core + targets + hub + server
./install.sh --no-server  # the library only (no dashboard/server)
./install.sh --dev        # add the [dev] extra to every distribution
```

`./install.sh` installs all four distributions in editable mode. Then:

```bash
spikeforge --help         # the library CLI
spikeforge-server         # the dashboard/WebSocket server on :8877
```

`python -m server` remains an equivalent way to launch the server.

### From PyPI (once published)

```bash
pip install "spikeforge[all]"  # library bundle: core + targets + hub
pip install spikeforge-server  # the server, pulling core + targets + hub
```

Run one headless example (no browser needed), then launch the dashboard:

```bash
python examples/04_nir_export_validate.py
spikeforge-server
```

Open <http://localhost:8877> for the single-port build, or run the Vite dev
server for hot reload (`cd client && npm install && npm run dev`; its proxy
target is `:8877`). Ten runnable journeys — training, encoding, NIR/ONNX,
the hub, backends, energy, sequence experiments, and reproducibility — are
listed in [`examples/README.md`](../examples/README.md), and the
copy-pasteable recipes are in [`COOKBOOK.md`](../COOKBOOK.md).

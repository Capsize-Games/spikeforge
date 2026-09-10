# snn-interpreter

A rate-coding and spike-encoding playground for spiking neural networks
(SNNs), built on [snnTorch](https://snntorch.readthedocs.io/) and PyTorch.
It loads MNIST subsets, converts samples into **rate**, **latency**, and
**delta** spike codes (plus random spike generation), and renders them
through matplotlib exports and a live **browser dashboard** served by
FastAPI + React over WebSockets.

The encoding pipeline mirrors [snnTorch Tutorial 1](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html).

## Features

- MNIST loading + `snntorch.utils.data_subset` reduction
- Rate coding (`spikegen.rate`) at gain 1 and a lower gain
- Latency coding (`spikegen.latency`) with `tau`, `threshold`, `linear`,
  `normalize`, and `clip` variants (tutorial 2.3)
- Delta modulation (`spikegen.delta`) with on/off spikes (tutorial 2.4)
- Random spike generation from scratch via `spikegen.rate_conv` (tutorial 3)
- Matplotlib exports (MP4s, GIFs, rasters, reconstructions) into `build/`
- **Training**: a fully-connected LIF spiking network with a
  surrogate-gradient cross-entropy loss, streaming live loss and both
  batch + held-out accuracy
- **Datasets**: train on MNIST, Fashion-MNIST, KMNIST, QMNIST, USPS,
  EMNIST digits/letters, or (grayscaled) CIFAR-10, all normalised to
  28x28 so one architecture fits all
- **Model management**: save checkpoints, list/load/delete them, and
  continue training an already-trained model
- **Browser interface**: a dark-themed grid dashboard that streams encoded
  spike data over WebSockets and renders plots live in the client, with a
  training panel (dataset picker, network config, live charts, model
  management, predictions)
- **Compute selection**: a CPU/GPU device dropdown (GPU by default, with
  automatic CPU fallback) and a live CPU-RAM / VRAM resource monitor

## Requirements

- Python 3.8+
- `torch`, `torchvision`, `snntorch`
- `matplotlib`, `Pillow`, `numpy`
- Node.js 18+ and npm (for the `client/` dashboard)
- `ffmpeg` (only when exporting MP4s)

Install the package (with web extras for the dashboard):

```bash
pip install -e ".[web]"
```

## Usage

### 🐳 Run everything with Docker (recommended)

The whole stack — React dashboard, FastAPI server, WebSocket streaming —
builds into a single container and is served from **one port**.

```bash
docker compose up --build
```

Then open **http://localhost:8877**. The dashboard auto-connects to the
WebSocket on the same host/port (no separate backend or proxy to run).
Datasets download on first use into a Docker volume, so they persist across
restarts.

> Port 8877 was chosen to avoid clashing with other apps (e.g., 8000 is
> commonly used by other dev servers). To change it, edit the
> `ports:` mapping in [`docker-compose.yml`](docker-compose.yml).

### Compute device & resources

The training panel has a **Device** dropdown (CPU / GPU) that defaults to
GPU and falls back to CPU automatically when CUDA is unavailable. A live
**System resources** panel shows host CPU RAM and GPU VRAM
(used / total / free). The Docker image installs CUDA PyTorch (`cu132`) and
requests the host GPU, so `docker compose up --build` trains on the GPU out
of the box; build a smaller CPU-only image with:

```bash
docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
```

### Local (non-Docker) development

#### CLI exports (rate pipeline)

```bash
python main.py
```

### Additional encoding demos (latency / delta / random)

```bash
python main_encodings.py
```

Both write PNGs/GIFs/MP4s into `build/`.

### Local dev (server + Vite)

```bash
# terminal 1 - FastAPI on the canonical dev port :8877 (same port as Docker)
venv/bin/python -m server   # defaults to 8877 with reload

# terminal 2 - Vite (proxy target in client/vite.config.ts must match :8877)
cd client && npm install && npm run dev
```

Open the printed `http://localhost:5173` URL. Pick a coding type (rate,
latency, delta, or random), tune parameters, then hit **Apply & Run** to
stream spike frames into the raster and image panels in real time. The Vite
dev server proxies `/ws` to the FastAPI server on port 8877 (see
[`client/vite.config.ts`](client/vite.config.ts) and
[`server/__main__.py`](server/__main__.py)).

## Developer script

[`scripts/dev.sh`](scripts/dev.sh) bundles the common tasks (setup, lint,
tests, dev servers, dataset cache, Docker):

```bash
scripts/dev.sh help          # list every command
scripts/dev.sh check         # ruff + client type-check + client build
scripts/dev.sh dev           # run the API and Vite dev server together
scripts/dev.sh data          # show the dataset cache and sizes
scripts/dev.sh data-clear    # clear dataset caches (keeps models)
scripts/dev.sh docker-reset  # rebuild the Docker volume from scratch
```

## Project layout

```
main.py                      Thin entry point: rate pipeline -> exporters
main_encodings.py            Extra tutorial-1 encodings (latency/delta/random)
setup.py                     Packaging metadata + web extra
snn_interpreter/
  config.py                  Paths/settings resolved from the environment
  data/                      Dataset registry, loaders, sample access
    datasets.py              Dataset registry (MNIST/Fashion/KMNIST/...)
    data_loader.py           Loader construction with subset reduction
    sample_source.py         Single transformed images/labels for the viewer
  encoding/                  Spike-encoding transforms
    spike_encoder.py         SpikeEncoder: rate/latency/delta/random
    latency_trainer.py       LatencyTrainer (tutorial 2.3)
    delta_trainer.py         DeltaTrainer (tutorial 2.4)
    random_spikegen.py       RandomSpikeGenerator (tutorial 3)
  network/                   Model, inference, and persistence
    spiking_net.py           SpikingNet: fully-connected LIF model
    inference.py             Per-sample prediction + layer activity
    model_store.py           Save/load/list/delete model checkpoints
  training/                  Training loop and its collaborators
    trainer.py               SSNTrainer: MNIST loading + rate coding
    logger.py                SNNTrainerLogger: diagnostic prints
    training_engine.py       TrainingEngine: train loop yielding metrics
    checkpoint_mixin.py      Checkpoint save/restore behaviour
    encoding_mixin.py        Raw-pixel / spike input encoding
  exporters/                 matplotlib/GIF/MP4 output -> build/
    exporter.py              Exporter base + build/ output resolution
    plot_utils.py            shared fig/GIF helpers
    *_exporter.py            per-visual exporters
  runtime/                   Compute environment
    device.py                CPU/GPU selection + auto benchmark
    system_stats.py          CPU RAM / GPU VRAM snapshots
server/
  app.py                     FastAPI app + WebSocket endpoint
  handlers.py                Inbound message routing (encode + train)
  messages.py                Outbound WS message helpers
  encoder.py                 Config -> encoder engine (JSON payloads)
  training.py                Threaded training bridge -> asyncio queue
  session.py                 Per-connection state (engine/train/stream)
  schemas/                   Pydantic WS message/config schemas
    encode_config.py         EncodeConfig + CodingType
    train_config.py          TrainConfig
    client_message.py        ClientMessage
    server_message.py        ServerMessage
client/                      Vite + React + TypeScript dashboard
  src/                       app shell, theme, types, WS/training hooks
  src/hooks/                 viewer state, encode config, model actions
  src/components/            controls, charts, canvas panels (incl. training)
  src/styles/                split stylesheet (base/sections/controls/...)
```

Code is kept tidy by construction: modules are grouped into focused
subpackages, each Python file is under 200 lines, every Python function
stays under 20 lines, and each class lives in its own file.

## Notes

- Dataset downloads run in an isolated worker process, so the dashboard stays
  responsive: it shows a progress overlay with a live byte counter and a
  **Cancel** button instead of appearing frozen.
- The Bernoulli encoder in `spikegen.rate` is stochastic, so the reported
  spiking percentage and spike patterns vary run to run — expected.
- Larger `num_steps` values (e.g., 100) produce longer, richer animations;
  `subset`/`batch_size` trade dataset coverage for encode speed.

## License

Released under the BSD 3-Clause License — see [`LICENSE`](LICENSE) and
[`AUTHORS`](AUTHORS).

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
# terminal 1 - FastAPI (use a port not taken by other apps)
venv/bin/python -m uvicorn server.app:app --port 8765

# terminal 2 - Vite (proxy target in client/vite.config.ts must match)
cd client && npm install && npm run dev
```

Open the printed `http://localhost:5173` URL. Pick a coding type (rate,
latency, delta, or random), tune parameters, then hit **Apply & Run** to
stream spike frames into the raster and image panels in real time. The Vite
dev server proxies `/ws` to the FastAPI port in
[`client/vite.config.ts`](client/vite.config.ts).

## Project layout

```
main.py                      Thin entry point: rate pipeline -> exporters
main_encodings.py            Extra tutorial-1 encodings (latency/delta/random)
setup.py                     Packaging metadata + web extra
snn_interpreter/
  trainer.py                 SSNTrainer: MNIST loading + rate coding
  latency_trainer.py         LatencyTrainer (tutorial 2.3)
  delta_trainer.py           DeltaTrainer (tutorial 2.4)
  random_spikegen.py         RandomSpikeGenerator (tutorial 3)
  datasets.py                Dataset registry (MNIST/Fashion/KMNIST/...)
  spiking_net.py             SpikingNet: fully-connected LIF model
  training_engine.py         TrainingEngine: train loop yielding metrics
  model_store.py             Save/load/list/delete model checkpoints
  *_exporter.py              matplotlib exporters -> build/
  plot_utils.py              shared fig/GIF helpers
  exporter.py                Exporter base + build/ output resolution
server/
  app.py                     FastAPI app + WebSocket endpoint
  handlers.py                Inbound message routing (encode + train)
  messages.py                Outbound WS message helpers
  encoder.py                 Config -> encoder engine (JSON payloads)
  training.py                Threaded training bridge -> asyncio queue
  session.py                 Per-connection state (engine/train/stream)
  schemas.py                 Pydantic WS message/config schemas
client/                      Vite + React + TypeScript dashboard
  src/                       dark grid UI, canvas panels, WS hook
  src/components/            controls, charts, panels (incl. training)
```

Code is kept tidy by construction: each Python file is under 200 lines,
every Python function stays under 20 lines, and each class lives in its own
file.

## Notes

- The Bernoulli encoder in `spikegen.rate` is stochastic, so the reported
  spiking percentage and spike patterns vary run to run — expected.
- Larger `num_steps` values (e.g., 100) produce longer, richer animations;
  `subset`/`batch_size` trade dataset coverage for encode speed.

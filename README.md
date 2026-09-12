# spikeforge

[![CI](https://github.com/capsize-games/spikeforge/actions/workflows/ci.yml/badge.svg)](https://github.com/capsize-games/spikeforge/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/capsize-games/spikeforge/blob/main/CONTRIBUTING.md)
[![Docs](https://img.shields.io/badge/docs-long--form%20reference-blue.svg)](documentation/README.md)

A spiking-neural-network (SNN) toolkit built on
[snnTorch](https://snntorch.readthedocs.io/) and PyTorch. It loads MNIST-style
and neuromorphic event datasets, encodes them into **rate**, **latency**,
**delta**, and **random** spikes, and trains, validates, exports, and deploys
LIF networks — with a live **browser dashboard** served over WebSockets.

The encoding pipeline mirrors [snnTorch Tutorial 1](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html).

> **Pre-1.0 and unpublished.** `install.sh` is the supported path today;
> the PyPI commands below apply once the distributions are published.
> Before trusting any number the toolkit produces, read
> [Implications and boundaries](documentation/implications-and-boundaries.md).

## Features

- **Encoding** — rate, latency, delta, and random spike coders (snnTorch tutorials 1–3).
- **Training** — fully-connected and convolutional LIF networks with surrogate-gradient cross-entropy, live loss/accuracy, checkpointing, and opt-in AMP / gradient checkpointing / truncated BPTT / multi-GPU.
- **Topologies** — `fc_legacy`, `fc_small`, `conv_net`, `recurrent_net`, plus the sequence presets `sequence_mlp` and `sequence_attn`.
- **Datasets** — MNIST, Fashion-MNIST, KMNIST, QMNIST, USPS, EMNIST, CIFAR-10, and (via the `events` extra) N-MNIST, DVS128 Gesture, CIFAR10-DVS, and Spiking Speech Commands.
- **Interpreter spine** — NIR export, an independent NIR interpreter, and numerical drift validation.
- **Introspection** — educational-mode `U[t]`/`I[t]`/`S[t]` traces, trajectory metrics, encoding/decoding reports, and surrogate-derivative curves.
- **Deployment** — a capability matrix, substitution/rewrite reports, weight quantization, energy accounting, and executable `reference`, `norse`, and `lava_loihi2` backends.
- **Model hub** — a curated, offline-first catalog plus optional live Hugging Face search.
- **Dashboard** — a React + TypeScript UI with training, introspection, analysis, targets, energy, and hub panels, and seven guided walkthroughs.

## Requirements

- **Python 3.10–3.13** (matches `requires-python` and CI)
- `torch`, `torchvision`, `snntorch`, `matplotlib`, `Pillow`, `numpy`
- **Node.js 18+ and npm** — only for the `client/` dashboard
- **ffmpeg** — only when exporting MP4s
- **Docker + Compose** — optional, for the single-container run

See [documentation/requirements.md](documentation/requirements.md) for the full
dependency and optional-extras matrix.

## Installation

### From a clone (recommended)

```bash
git clone https://github.com/capsize-games/spikeforge.git
cd spikeforge
./install.sh            # editable install of all four distributions
```

`./install.sh` installs `spikeforge`, `spikeforge-targets`, `spikeforge-hub`,
and `spikeforge-server` in editable mode. Useful flags:

```bash
./install.sh --no-server   # library only (no dashboard/WebSocket server)
./install.sh --dev         # add the [dev] extra (pytest, ruff, ...)
./install.sh --user        # install into the user site
```

### From PyPI (once published)

```bash
pip install "spikeforge[all]"   # core library + targets + hub
pip install spikeforge-server   # FastAPI dashboard/WebSocket server
```

## Quickstart

### Run everything with Docker (single port)

```bash
docker compose up --build
```

Open <http://localhost:8877> — the dashboard connects to the WebSocket on the
same host and port, so there is no separate backend or proxy to run.

```bash
scripts/docker_server.sh --cpu       # smaller CPU-only image
scripts/docker_server.sh --gpu       # explicit CUDA image
scripts/docker_server.sh --help      # every option
```

### Run a headless example

```bash
python examples/04_nir_export_validate.py   # export conv_net to NIR and validate drift
```

### Local development (server + Vite)

```bash
# terminal 1 — FastAPI + WebSocket on :8877
spikeforge-server            # equivalent to: python -m server

# terminal 2 — Vite dev server with hot reload
cd client && npm install && npm run dev
```

Open the printed <http://localhost:5173> URL. The Vite server proxies `/ws`
to the FastAPI server on port 8877.

### Command-line tools

| Script | Purpose |
|---|---|
| `spikeforge` | Rate-encoding pipeline → matplotlib/GIF/MP4 exports |
| `spikeforge-encodings` | Latency / delta / random encoding demos |
| `spikeforge-verify` | NIR export, validation, targets, deploy, round-trip, ONNX |
| `spikeforge-records` | Search, diff, and inspect saved model checkpoints |
| `spikeforge-targets` | Deployment targets, rewrite, and extract |
| `spikeforge-hub` | Browse, inspect, and import hub models |
| `spikeforge-energy` | SOP/MAC/AC energy estimates per target |
| `spikeforge-benchmark` | Timing/memory harness with regression gating |

## Packages

This repository is a single workspace that publishes four distributions:

| Distribution | Import root | Purpose |
|---|---|---|
| `spikeforge` | `spikeforge` | Core package: encoders, topologies, training, simulator, NIR bridge, tracking |
| `spikeforge-targets` | `spikeforge_targets` | Deployment targets, quantization, energy accounting, sparse event runtime |
| `spikeforge-hub` | `spikeforge_hub` | Curated model hub and optional Hugging Face access |
| `spikeforge-server` | `server` | FastAPI + WebSocket server and dashboard hosting |

## Documentation

The top-level README stays short on purpose. The full reference lives in
[`documentation/`](documentation/README.md):

| Document | Description |
|---|---|
| [documentation/](documentation/README.md) | Index of the long-form reference |
| [Quickstart](documentation/quickstart.md) | Install paths and first run |
| [Usage](documentation/usage.md) | Docker, local dev, CLI, and device selection |
| [Architecture](documentation/architecture.md) | The `TopologySpec` spine and data flow |
| [Project layout](documentation/project-layout.md) | Module-by-module map |
| [Implications and boundaries](documentation/implications-and-boundaries.md) | What the results do and do not tell you |
| [COOKBOOK.md](COOKBOOK.md) | Copy-pasteable recipes |
| [examples/](examples/) | Ten runnable end-to-end journeys |
| [plans/](plans/) | Design documents, ARCH-0001 split, and roadmap |
| [protocol/](protocol/) | Versioned WebSocket JSON Schema contract |

## Development

[`scripts/dev.sh`](scripts/dev.sh) bundles the common tasks:

```bash
scripts/dev.sh help          # list every command
scripts/dev.sh check         # ruff + client type-check + client build
scripts/dev.sh dev           # run the API and Vite dev server together
scripts/dev.sh bench         # run the benchmark suite
scripts/dev.sh data          # show the dataset cache and sizes
scripts/dev.sh test          # run the test suite
```

Tests run with `pytest`; lint with `ruff check .`. See
[CONTRIBUTING.md](https://github.com/capsize-games/spikeforge/blob/main/CONTRIBUTING.md)
and [rules.md](rules.md) before opening a pull request.

## Citing

If spikeforge is useful in your research, please cite it — see
[CITATION.cff](CITATION.cff) (GitHub renders a "Cite this repository"
button from it automatically).

## License

Released under the **BSD 3-Clause License** — see [LICENSE](LICENSE) and
[AUTHORS](AUTHORS).

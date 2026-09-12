# spikeforge

[![CI](https://github.com/capsize-games/spikeforge/actions/workflows/ci.yml/badge.svg)](https://github.com/capsize-games/spikeforge/actions/workflows/ci.yml)
[![Status: pre-1.0](https://img.shields.io/badge/status-pre--1.0-orange.svg)](OPEN_SOURCE_CHECKLIST.md)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/capsize-games/spikeforge/blob/main/CONTRIBUTING.md)
[![Docs](https://img.shields.io/badge/docs-long--form%20reference-blue.svg)](documentation/README.md)

A spiking-neural-network (SNN) toolkit built on
[snnTorch](https://snntorch.readthedocs.io/) and PyTorch. Loads MNIST-style
and neuromorphic event datasets, encodes them into rate, latency, delta, and
random spikes, and trains, validates, exports, and deploys LIF networks —
with a live browser dashboard served over WebSockets.

> **Pre-1.0 and unpublished.** `install.sh` is the supported path today;
> PyPI ships once the distributions are published. Before trusting any
> number this produces, read
> [Implications and boundaries](documentation/implications-and-boundaries.md).

## Quickstart

```bash
git clone https://github.com/capsize-games/spikeforge.git
cd spikeforge
docker compose up --build
```

Open <http://localhost:8877> — the dashboard connects to the WebSocket on
the same host and port. No separate backend or proxy to run.

Prefer a local install, a headless example, or the CLI tools instead? See
[Usage](documentation/usage.md) and [Quickstart](documentation/quickstart.md)
for every path (`./install.sh`, local dev with Vite, `examples/`, and the
eight `spikeforge-*` console scripts).

## Features

- **Encoding** — rate, latency, delta, and random spike coders.
- **Training** — fully-connected and convolutional LIF networks with
  surrogate-gradient cross-entropy, checkpointing, and opt-in AMP / gradient
  checkpointing / truncated BPTT / multi-GPU.
- **Topologies** — `fc_legacy`, `fc_small`, `conv_net`, `recurrent_net`, plus
  the sequence presets `sequence_mlp` and `sequence_attn`.
- **Datasets** — MNIST, Fashion-MNIST, KMNIST, QMNIST, USPS, EMNIST,
  CIFAR-10, and (via the `events` extra) N-MNIST, DVS128 Gesture,
  CIFAR10-DVS, and Spiking Speech Commands.
- **Interpreter spine** — NIR export, an independent NIR interpreter, and
  numerical drift validation.
- **Introspection** — educational-mode `U[t]`/`I[t]`/`S[t]` traces,
  trajectory metrics, and surrogate-derivative curves.
- **Deployment** — a capability matrix, weight quantization, energy
  accounting, and executable `reference`, `norse`, and `lava_loihi2`
  backends.
- **Model hub** — a curated, offline-first catalog plus optional live
  Hugging Face search.
- **Dashboard** — a React + TypeScript UI with training, introspection,
  analysis, targets, energy, and hub panels, and seven guided walkthroughs.

## Packages

This repository is a single workspace that publishes four distributions:

| Distribution | Import root | Purpose |
|---|---|---|
| `spikeforge` | `spikeforge` | Core package: encoders, topologies, training, simulator, NIR bridge, tracking |
| `spikeforge-targets` | `spikeforge_targets` | Deployment targets, quantization, energy accounting, sparse event runtime |
| `spikeforge-hub` | `spikeforge_hub` | Curated model hub and optional Hugging Face access |
| `spikeforge-server` | `server` | FastAPI + WebSocket server and dashboard hosting |

## Documentation

This README stays short on purpose.
[`documentation/`](documentation/README.md) is the full reference — install
paths, the CLI tools, architecture, module layout, and the dev workflow —
written for contributors and coding agents alike. Also see
[COOKBOOK.md](COOKBOOK.md) for copy-pasteable recipes,
[examples/](examples/) for runnable end-to-end scripts, and
[plans/](plans/) for design documents and the roadmap.

See [CONTRIBUTING.md](https://github.com/capsize-games/spikeforge/blob/main/CONTRIBUTING.md)
and [rules.md](rules.md) before opening a pull request.

## Citing

If spikeforge is useful in your research, please cite it — see
[CITATION.cff](CITATION.cff) (GitHub renders a "Cite this repository"
button from it automatically).

## License

Released under the **BSD 3-Clause License** — see [LICENSE](LICENSE) and
[AUTHORS](AUTHORS).

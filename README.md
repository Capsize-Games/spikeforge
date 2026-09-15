# spikeforge

[![CI](https://github.com/Capsize-Games/spikeforge/actions/workflows/ci.yml/badge.svg)](https://github.com/Capsize-Games/spikeforge/actions/workflows/ci.yml)
[![Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white)](https://capsizegames.com/discord)
[![Status: pre-1.0](https://img.shields.io/badge/status-pre--1.0-orange.svg)](https://github.com/Capsize-Games/spikeforge/blob/main/OPEN_SOURCE_CHECKLIST.md)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/Capsize-Games/spikeforge/blob/main/LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Capsize-Games/spikeforge/blob/main/CONTRIBUTING.md)
[![Docs](https://img.shields.io/badge/docs-long--form%20reference-blue.svg)](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/README.md)

![spikeforge dashboard](https://raw.githubusercontent.com/Capsize-Games/spikeforge/main/images/dashboard.png)

A spiking-neural-network (SNN) toolkit built on
[snnTorch](https://snntorch.readthedocs.io/) and PyTorch. Loads MNIST-style
and neuromorphic event datasets, encodes them into rate, latency, delta, and
random spikes, and trains, validates, exports, and deploys LIF networks —
with a live browser dashboard served over WebSockets.

> **Pre-1.0.** Before trusting any number this produces, read
> [Implications and boundaries](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/implications-and-boundaries.md).

## Quickstart

No clone, no Docker — five lines of Python:

```bash
pip install spikeforge
```

```python
from spikeforge import TrainingEngine

engine = TrainingEngine(dataset="mnist", hidden=32, epochs=1, num_steps=5)
for metrics in engine.train():
    last = metrics
print(last)  # {'loss': ..., 'train_accuracy': ..., 'test_accuracy': ...}
```

That finishes in seconds — about 3 s on a 12-thread desktop CPU, nearer 14 s
on a laptop — and lands in the mid-80s for accuracy. The snippet sets no seed,
so the exact figure moves between runs, and the `test_accuracy` it prints is
the engine's fast progress probe rather than the whole test split. For numbers
measured on the *complete* held-out split, with the command that reproduces
each one, see [Benchmarks](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/benchmarks.md).
The first run downloads MNIST; later runs are offline.

`TrainingEngine` is the entry point for new code — it owns the training loop,
topology selection, encoding, checkpointing, and evaluation. (`SNNTrainer`,
also exported, is the older MNIST rate-coding helper behind the tutorial demo;
reach for `TrainingEngine`.) `spikeforge-verify`, `spikeforge-benchmark`, and
the NIR/deployment/energy pieces live in
[Usage](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/usage.md).

**On a machine with no GPU**, install the CPU torch wheels *first* so pip does
not pull the entire CUDA stack in behind them:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install spikeforge
```

That is a ~1.1 GB environment instead of ~5.5 GB, and it is the same flow the
Dockerfile and CI use. The order matters: with `--extra-index-url` pip still
prefers the CUDA build from PyPI.

Want the live browser dashboard instead? That's the richer, second path:

```bash
git clone https://github.com/Capsize-Games/spikeforge.git
cd spikeforge
docker compose up --build
```

Open <http://localhost:8877> — the dashboard connects to the WebSocket on
the same host and port. No separate backend or proxy to run.

See [Usage](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/usage.md) and [Quickstart](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/quickstart.md)
for every path (`./install.sh`, local dev with Vite, `examples/`, and the
`spikeforge-*` console scripts).

## Why not just snnTorch, Norse, Lava, or SpikingJelly?

spikeforge doesn't replace those — it trains through
[snnTorch](https://snntorch.readthedocs.io/) and can deploy to Norse and
Lava. The question is what it adds on top: a
[NIR](https://github.com/neuromorphs/NIR)-described interpreter spine with
an *independent* interpreter that re-executes every exported graph and
reports numerical drift against the original model (not just export —
snnTorch and Norse both export NIR too; Lava-DL currently only reads it,
not writes it), a deployment-capability
matrix that reports per-target op support and availability instead of just
declaring support, an event-driven energy estimator, and a live training/
introspection dashboard. None of the other libraries ship the latter two as
part of the library itself, to our knowledge — file an issue if that's
stale.

| | snnTorch | Norse | Lava | SpikingJelly | spikeforge |
|---|---|---|---|---|---|
| Surrogate-gradient training in PyTorch | ✅ | ✅ | ✅ (Lava-DL) | ✅ | ✅ (via snnTorch) |
| NIR export (write) | ✅ | ✅ | import-only (Lava-DL reads NIR, doesn't write it) | not part of NIR's official framework list | ✅ |
| Independent re-execution + drift check of the exported graph | — | — | — | — | ✅ |
| Per-target deployment capability matrix (op support, honest availability) | — | — | Loihi-focused | — | ✅ (reference/Norse/Lava/SpiNNaker2/Speck/Xylo) |
| Built-in event-driven energy estimator (SOP/MAC/AC) | — | — | — | — | ✅ (labelled `estimate`, not hardware-measured) |
| Pretrained model zoo | — | — | — | ✅ (some vision tasks) | small: six of this project's own reference checkpoints with published accuracy, plus untrained preset shapes ([details](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/benchmarks.md)) |
| Live browser training/introspection dashboard | — | — | — | — | ✅ |

If you already have a training loop in snnTorch/Norse/Lava/SpikingJelly and
don't need interop, deployment reporting, energy estimates, or the
dashboard, you may not need spikeforge on top of it — that's a fair
"obviously no," and better than an uninformed "obviously yes."

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
- **Deployment** — a capability matrix, weight quantization with a drift
  check that can simulate activation/membrane rounding, energy accounting,
  and executable `reference`, `norse`, and `lava_loihi2` backends.
- **Model hub** — a curated, offline-first catalog carrying this project's
  own trained reference checkpoints (with the accuracy each scores), plus
  optional live Hugging Face search.
- **Dashboard** — a React + TypeScript UI with training, introspection,
  analysis, targets, energy, and hub panels, and seven guided walkthroughs.

## What's implemented vs. experimental vs. spec-only

The features list above spans very different levels of maturity. This is
the ten-second version; each row links to the honest detail.

| Capability | Status | Detail |
|---|---|---|
| Training, encoding, topologies, checkpointing | Shipped | [Features](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/features.md) |
| NIR export + independent-interpreter drift validation | Shipped | [Interpreter spine](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/interpreter-spine.md) |
| Deployment — `reference` target | Shipped, always available | [Backend execution](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/backend-execution.md) |
| Deployment — `norse`, `lava_loihi2` targets | Shipped, gated on an SDK extra (`pip install spikeforge-targets[norse]` or `[lava]`) | [Targets and interop](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/targets-and-interop.md) |
| Deployment — `spinnaker2`, `speck`, `xylo` targets | Spec-only — registered in the capability matrix, no installable SDK integration yet | [Targets and interop](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/targets-and-interop.md) |
| Energy accounting (SOP/MAC/AC) | Shipped as an explicit `estimate`, not hardware-measured | [Event runtime and energy](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/event-runtime-and-energy.md) |
| Model hub catalog, CLI, dashboard panel, opt-in live HF search | Shipped | [Model hub](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/model-hub.md) |
| Model hub catalog *content* | Six trained reference checkpoints with published accuracy, alongside untrained preset shapes — not a large zoo | [Benchmarks](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/benchmarks.md) |
| Sequence/attention topologies (`sequence_mlp`, `sequence_attn`) | Experimental — research scope, not production sequence training | [Sequence primitives](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/sequence-primitives.md) |
| ONNX interop | Shipped, single-step export/import only | [Interop fold-ins](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/interop-foldins.md) |
| Production use-case toolkit: streaming time-series (UC-1) | Shipped | [UC-1](https://github.com/Capsize-Games/spikeforge/blob/main/plans/use_case_streaming_timeseries.md) |
| Production use-case toolkit: UC-2 through UC-10 | Spec-only (design docs behind issues #13–#21) | [plans/index.md](https://github.com/Capsize-Games/spikeforge/blob/main/plans/index.md#production-use-cases-pc-0) |
| Live browser dashboard | Shipped | [Dashboard](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/dashboard.md) |

## What it scores

Reference configurations on the datasets and topologies that ship, each
measured on the **complete** held-out test split, with the command that
reproduces it:

| Dataset | Topology | Test accuracy | Train time (CPU) |
|---|---|---|---|
| MNIST | `conv_net` | **97.13%** | 249 s |
| MNIST | `fc_legacy` | **94.29%** | 50 s |
| MNIST | `recurrent_net` | **92.46%** | 55 s |
| MNIST | `fc_small` | **92.16%** | 40 s |
| Fashion-MNIST | `fc_legacy` | **75.55%** | 62 s |
| KMNIST | `fc_legacy` | **70.79%** | 49 s |

Every figure is the **complete** held-out test split, on a 12-thread x86-64
CPU with no GPU. The full table — epochs, time steps, hardware, seed, and the
one command that reproduces each row — is in
[Benchmarks](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/benchmarks.md).

These are reference configurations with stock hyperparameters and a single
seed, **not** tuned attempts at state of the art; read them as a floor the
shipped defaults reach, not as a ceiling. The checkpoints they produce are the
trained entries in the model hub (`spikeforge-hub list --trained`).

## Packages

This repository is a single workspace that publishes seven distributions,
each versioned independently:

| Distribution | Import root | Purpose |
|---|---|---|
| `spikeforge` | `spikeforge` | Core package: encoders, topologies, training, simulator, NIR bridge, tracking |
| `spikeforge-targets` | `spikeforge_targets` | Deployment targets, quantization, energy accounting, sparse event runtime |
| `spikeforge-hub` | `spikeforge_hub` | Curated model hub and optional Hugging Face access |
| `spikeforge-server` | `server` | FastAPI + WebSocket server and dashboard hosting |
| `spikeforge-serve` | `spikeforge_serve` | Headless REST/WebSocket inference service for a deployment bundle |
| `spikeforge-clients` | `spikeforge_clients` | Python/TypeScript/CLI clients for `spikeforge-serve` (no torch dependency) |
| `spikeforge-io` | `spikeforge_io` | Recorded-stream I/O adapters and windowing |

The satellites sit at lower version numbers than core by design, not neglect —
each package is versioned independently and moves only when it changes, and
[`compatibility.json`](https://github.com/Capsize-Games/spikeforge/blob/main/compatibility.json) is the source of truth for which
satellite versions go with which core release. If you're pinning versions by
hand, read that file rather than assuming semver alignment across packages.

## Documentation

This README covers first contact and positioning; it deliberately doesn't
go deeper. [`documentation/`](https://github.com/Capsize-Games/spikeforge/blob/main/documentation/README.md) is the full reference — install
paths, the CLI tools, architecture, module layout, and the dev workflow —
written for contributors and coding agents alike. Also see
[COOKBOOK.md](https://github.com/Capsize-Games/spikeforge/blob/main/COOKBOOK.md) for copy-pasteable recipes,
[examples/](https://github.com/Capsize-Games/spikeforge/tree/main/examples/) for runnable end-to-end scripts, and
[plans/](https://github.com/Capsize-Games/spikeforge/tree/main/plans/) for design documents and the roadmap.

See [CONTRIBUTING.md](https://github.com/Capsize-Games/spikeforge/blob/main/CONTRIBUTING.md)
and [rules.md](https://github.com/Capsize-Games/spikeforge/blob/main/rules.md) before opening a pull request.

## Citing

If spikeforge is useful in your research, please cite it — see
[CITATION.cff](https://github.com/Capsize-Games/spikeforge/blob/main/CITATION.cff) (GitHub renders a "Cite this repository"
button from it automatically).

## License

Released under the **BSD 3-Clause License** — see [LICENSE](https://github.com/Capsize-Games/spikeforge/blob/main/LICENSE) and
[AUTHORS](https://github.com/Capsize-Games/spikeforge/blob/main/AUTHORS).

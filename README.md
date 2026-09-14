# spikeforge

[![CI](https://github.com/capsize-games/spikeforge/actions/workflows/ci.yml/badge.svg)](https://github.com/capsize-games/spikeforge/actions/workflows/ci.yml)
[![Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white)](https://capsizegames.com/discord)
[![Status: pre-1.0](https://img.shields.io/badge/status-pre--1.0-orange.svg)](OPEN_SOURCE_CHECKLIST.md)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/capsize-games/spikeforge/blob/main/CONTRIBUTING.md)
[![Docs](https://img.shields.io/badge/docs-long--form%20reference-blue.svg)](documentation/README.md)

![spikeforge dashboard](images/dashboard.png)

A spiking-neural-network (SNN) toolkit built on
[snnTorch](https://snntorch.readthedocs.io/) and PyTorch. Loads MNIST-style
and neuromorphic event datasets, encodes them into rate, latency, delta, and
random spikes, and trains, validates, exports, and deploys LIF networks —
with a live browser dashboard served over WebSockets.

> **Pre-1.0.** Before trusting any number this produces, read
> [Implications and boundaries](documentation/implications-and-boundaries.md).

## Quickstart

No clone, no Docker — five lines of Python:

```bash
pip install spikeforge
```

```python
from spikeforge.training.training_engine import TrainingEngine

engine = TrainingEngine(dataset="mnist", hidden=32, epochs=1, num_steps=5)
for metrics in engine.train():
    last = metrics
print(last)  # {'loss': ..., 'train_accuracy': ..., 'test_accuracy': ...}
```

The first run downloads MNIST; later runs are offline. That's the training
API — `spikeforge-verify`, `spikeforge-benchmark`, and the NIR/deployment/
energy pieces live in [Usage](documentation/usage.md).

Want the live browser dashboard instead? That's the richer, second path:

```bash
git clone https://github.com/capsize-games/spikeforge.git
cd spikeforge
docker compose up --build
```

Open <http://localhost:8877> — the dashboard connects to the WebSocket on
the same host and port. No separate backend or proxy to run.

See [Usage](documentation/usage.md) and [Quickstart](documentation/quickstart.md)
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
| Pretrained model zoo | — | — | — | ✅ (some vision tasks) | catalog infra ships; mostly untrained preset shapes today ([details](spikeforge_hub/models.json)) |
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
- **Deployment** — a capability matrix, weight quantization, energy
  accounting, and executable `reference`, `norse`, and `lava_loihi2`
  backends.
- **Model hub** — a curated, offline-first catalog plus optional live
  Hugging Face search.
- **Dashboard** — a React + TypeScript UI with training, introspection,
  analysis, targets, energy, and hub panels, and seven guided walkthroughs.

## What's implemented vs. experimental vs. spec-only

The features list above spans very different levels of maturity. This is
the ten-second version; each row links to the honest detail.

| Capability | Status | Detail |
|---|---|---|
| Training, encoding, topologies, checkpointing | Shipped | [Features](documentation/features.md) |
| NIR export + independent-interpreter drift validation | Shipped | [Interpreter spine](documentation/interpreter-spine.md) |
| Deployment — `reference` target | Shipped, always available | [Backend execution](documentation/backend-execution.md) |
| Deployment — `norse`, `lava_loihi2` targets | Shipped, gated on an SDK extra (`pip install spikeforge-targets[norse]` or `[lava]`) | [Targets and interop](documentation/targets-and-interop.md) |
| Deployment — `spinnaker2`, `speck`, `xylo` targets | Spec-only — registered in the capability matrix, no installable SDK integration yet | [Targets and interop](documentation/targets-and-interop.md) |
| Energy accounting (SOP/MAC/AC) | Shipped as an explicit `estimate`, not hardware-measured | [Event runtime and energy](documentation/event-runtime-and-energy.md) |
| Model hub catalog, CLI, dashboard panel, opt-in live HF search | Shipped | [Model hub](documentation/model-hub.md) |
| Model hub catalog *content* | Thin — bundled NIR shapes today, not trained weights | [`spikeforge_hub/models.json`](spikeforge_hub/models.json) |
| Sequence/attention topologies (`sequence_mlp`, `sequence_attn`) | Experimental — research scope, not production sequence training | [Sequence primitives](documentation/sequence-primitives.md) |
| ONNX interop | Shipped, single-step export/import only | [Interop fold-ins](documentation/interop-foldins.md) |
| Production use-case toolkit: streaming time-series (UC-1) | Shipped | [UC-1](plans/use_case_streaming_timeseries.md) |
| Production use-case toolkit: UC-2 through UC-10 | Spec-only (design docs behind issues #13–#21) | [plans/index.md](plans/index.md#production-use-cases-pc-0) |
| Live browser dashboard | Shipped | [Dashboard](documentation/dashboard.md) |

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

`spikeforge-hub` and `spikeforge-targets` sit at `0.1.x` while core is at
`0.3.x` by design, not neglect — each package is versioned independently, and
[`compatibility.json`](compatibility.json) is the source of truth for which
satellite versions go with which core release. If you're pinning versions by
hand, read that file rather than assuming semver alignment across packages.

## Documentation

This README covers first contact and positioning; it deliberately doesn't
go deeper. [`documentation/`](documentation/README.md) is the full reference — install
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

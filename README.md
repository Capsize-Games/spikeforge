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
- **Interpreter spine (Phase 1)**: topology presets, a neuron registry, NIR
  export, an independent NIR interpreter, and numerical drift validation
  (see below)
- **Dual-mode introspection (Phase 2)**: an educational mode that records
  per-step `U[t]`/`I[t]`/`S[t]`, trajectory metrics, encoding/decoding
  reports, surrogate-gradient curves, a neuron comparison lab, and a
  production-mode benchmark harness (see below)
- **Unified dashboard (Phase 3)**: an Educational/Production mode toggle,
  topology/neuron/surrogate pickers, neuron-state trajectory and NIR graph
  viewers, a drift-validation panel, trajectory-metrics/encoding/surrogate/
  benchmark analysis panels, and seven guided walkthroughs (see below)

## Interpreter spine (Phase 1)

Phase 1 lifts the models onto the Neuromorphic Intermediate Representation
([NIR](https://github.com/neuromorphs/NIR)) and proves the translation. A
model is declared once as a `TopologySpec` and rendered twice: into the
snnTorch module that trains, and into the `nir.NIRGraph` that exports.

### Topology presets

Pick a topology by name — the training server takes a `topology` field and
the CLI takes `--topology`:

| Preset | Shape |
|---|---|
| `fc_legacy` | The original two-layer FC LIF `SpikingNet` (default) |
| `fc_small` | Small FC LIF with an explicit flatten entry stage |
| `conv_net` | Conv/pool feature extractor with a linear LIF readout |
| `recurrent_net` | FC LIF with a one-step delayed feedback edge |

`fc_legacy` stays the default and keeps `SpikingNet`'s
`_fc1`/`_lif1`/`_fc2`/`_lif2` state-dict keys, so existing checkpoints load
and infer unchanged.

### Neuron registry

`snn_interpreter/neurons/` maps a neuron name to its snnTorch factory and
its canonical NIR parameter contract. The registry ships `leaky`,
`lapicque`, `synaptic`, and `recurrent` (RLeaky) neurons.

### NIR export, interpretation, and validation

- `to_nir(spec, module)` exports a JSON-able `nir.NIRGraph`; `graph_summary`
  describes its nodes and edges.
- `NirInterpreter` executes the exported graph independently of snnTorch, so
  `validate(spec, module, spikes)` reports a genuine `ValidationReport`
  (per-layer max/mean/relative error plus spike agreement, and an overall
  `within_tolerance` flag). Every shipped preset validates with bit-exact
  spikes.
- All `nir`/`nirtorch` imports are confined to `nir_bridge/api.py`.

### Verify CLI

Headless `export` and `validate` commands. `validate` exits non-zero when a
report falls outside tolerance, so it doubles as a CI gate:

```bash
python -m snn_interpreter.cli.verify export --topology conv_net
python -m snn_interpreter.cli.verify export --topology fc_legacy --out g.json
python -m snn_interpreter.cli.verify validate --topology conv_net
python -m snn_interpreter.cli.verify validate --topology recurrent_net
```

### WebSocket actions

The server answers two new client actions: `nir_export` returns the graph
summary for the active or configured topology, and `nir_validate` returns a
drift report for the active model.

## Dual-mode introspection (Phase 2)

Phase 2 adds the educational/professional execution split and full
neuron-state introspection behind one shared code path.

### Execution modes

`ExecutionMode` ([`runtime/execution_mode.py`](snn_interpreter/runtime/execution_mode.py))
is a flag on the single temporal loop, not a fork:

- `EDUCATIONAL` records every per-step trace; `PRODUCTION` records none and
  runs lean.
- [`simulator.run()`](snn_interpreter/simulator/runner.py:21) takes
  `mode=...` and also exposes `track` / `membrane` / `current` for
  finer-grained capture.
- [`simulator.run_production()`](snn_interpreter/simulator/production.py:14)
  returns a `ProductionResult(trajectory, compiled, status)`. `compiled` is
  opt-in (`torch.compile`) and falls back transparently to eager, with
  `status` in `{"eager", "unavailable", "compiled", "fallback"}`.

Both paths share one loop, so the difference is recording overhead, not
behaviour.

### Trajectory capture

`Trajectory` ([`simulator/trajectory.py`](snn_interpreter/simulator/trajectory.py:9))
carries the averaged readout `logits` plus per-neuron-stage traces of spikes
`S[t]`, membrane `U[t]`, and input current `I[t]` (`currents` is the merged
inbound activation each stage received before its update). Educational mode
fills all three; production mode leaves them empty.

### Trajectory metrics

[`introspection.metrics.trajectory_metrics()`](snn_interpreter/introspection/metrics.py:28)
gathers, per stage:

- **firing rate** — mean spikes per neuron per step.
- **sparsity** — fraction of silent entries.
- **ISI** — count/mean/median/std/cv of inter-spike intervals (`null` when
  fewer than two spikes).
- **histogram** — per-neuron firing-rate bin edges and counts.

The result holds only plain JSON types.

### Encoding and decoding introspection

[`introspection.encoding.encoding_report()`](snn_interpreter/introspection/encoding.py:142)
encodes one image, reconstructs it where the coding is invertible, and
reports firing rate, sparsity, and coding-specific stats. The reconstruction
is explicitly approximate, documented in the report's `approximation` field:

- **rate** — mean spike count; a Bernoulli estimate of the clamped intensity
  that converges as `num_steps` grows.
- **latency** — inverts the time-to-first-spike map; quantised to integer
  steps and saturating at the threshold ceiling for sub-threshold pixels.
- **delta** — integrates the on/off stream crediting one threshold per
  spike; a lower bound, exact only when each step rises by exactly the
  threshold.
- **random** — carries no image signal, so `reconstruction` is `null` and
  `reconstruction_supported` is `false`.

### Surrogate gradients

[`introspection.surrogate`](snn_interpreter/introspection/surrogate.py:1)
discovers the selectable surrogate factories from the installed
`snntorch.surrogate` (so the list always matches what snnTorch provides).
`list_surrogates()` names them and
[`surrogate_curve()`](snn_interpreter/introspection/surrogate.py:94) samples
the backward-pass derivative `dS/dU` into parallel `x`/`y` lists. Neurons
accept an optional `surrogate` build parameter; leaving it unset (the
default) keeps the build byte-identical to before.

### Neuron comparison lab

[`introspection.comparison.compare_neurons()`](snn_interpreter/introspection/comparison.py:59)
runs the same seeded input through every registered neuron kind (Leaky,
Lapicque, Synaptic, recurrent LIF, and Alpha) and returns
`{kind: Trajectory}` for side-by-side diffing.

### Benchmark harness

[`snn_interpreter/benchmark/`](snn_interpreter/benchmark/__init__.py:1)
measures wall time and memory of forward and backward passes for each mode
(and, with `--compiled`, the compiled production path):

```bash
python -m snn_interpreter.benchmark                     # tiny default fixture
python -m snn_interpreter.benchmark --topology conv_net --steps 16 --compiled
python -m snn_interpreter.benchmark --out bench.json
```

The same report is available from Python via
`benchmark.run_benchmark(BenchmarkConfig(...))`; every measurement is seeded
and warmed up, and unavailable metrics are reported as `null`.

### WebSocket actions

Six data-only actions were added, with client types in
[`client/src/introspectionTypes.ts`](client/src/introspectionTypes.ts:1):

| Action | Server reply | Payload |
|---|---|---|
| `trajectory` | `trajectory` | Bounded `U[t]`/`I[t]`/`S[t]` rows (≤8 stages, ≤64 neurons) |
| `metrics` | `metrics` | Firing rate, sparsity, ISI, histogram per stage |
| `encoding_report` | `encoding_report` | Reconstruction + approximation note for the sample |
| `surrogates` | `surrogate_list` | Selectable surrogate names |
| `surrogate_curve` | `surrogate_curve` | Derivative `x`/`y` samples for one surrogate |
| `benchmark` | `benchmark` | Config, environment, and per-mode results |

All are read-only; precondition failures emit the existing `error` message.

## Dashboard (Phase 3)

Phase 3 turns the browser dashboard into the go-to surface for both
audiences. Every panel below renders from live server payloads over the
existing WebSocket protocol, so nothing needs a page reload.

### Execution-mode toggle

The top bar carries an **Educational / Production** toggle wired to
`TrainConfig.mode` — the same `ExecutionMode` the runtime uses. It changes
what a run *records*, not what it computes:

- **Production** (the default) skips trajectory capture and runs lean, so
  the introspection panels show their gated empty state.
- **Educational** records per-step `U[t]`/`I[t]`/`S[t]` and unlocks the
  trajectory viewer, the metrics panel, and the firing-rate histogram.

The mode is applied when the engine is built, so switch it and then start a
run (`train`) or load a checkpoint (`load_model`) to see the panels fill in.

### LEFT column — model controls

- **Topology** picker -> `TrainConfig.topology` (`fc_legacy`, `fc_small`,
  `conv_net`, `recurrent_net`).
- **Neuron** picker -> `TrainConfig.topology_params.neuron` (registry kinds).
- **Surrogate** picker -> `TrainConfig.topology_params.surrogate`; it also
  drives the surrogate-curve panel's initial selection.
- The existing **dataset** and **coding** controls (rate / latency / delta /
  random) and the model-zoo browser.

The hardware-target picker is deferred to Phase 5 (see Notes).

### CENTER column — introspection panels

- **Neuron-state trajectory viewer** — `U[t]` (membrane) and `I[t]` (input
  current) per stage, with a stage selector and the shared time cursor.
- **NIR topology graph viewer** — the graph summary drawn as nodes and edges,
  with non-linear (skip/conv) and delayed (recurrent) edges rendered
  distinctly.
- **NIR drift-validation panel** — the independent-interpreter
  `ValidationReport`, per layer and overall `within_tolerance`.
- The existing sample / spike-frame / reconstruction / raster panels.

### RIGHT column — analysis panels

- **Trajectory metrics** — firing rate, sparsity, and ISI per stage, plus a
  firing-rate histogram.
- **Encoding report** — reconstruction plus the approximation note for the
  current sample (see the Phase 2 decoding caveats).
- **Surrogate-derivative curve** — sampled `dS/dU` for the selected
  surrogate gradient.
- **Benchmark readout** — config, environment, and per-mode timing/memory.
- The training and prediction panels.

### Guided walkthroughs

Seven short in-app lessons (one per tutorial theme) launch from the **Tours**
menu in the top bar. Each step highlights its target control or panel and
explains what it does:

| Lesson | Theme |
|---|---|
| `encoding` | Spike encoding |
| `datasets` | Neuromorphic datasets |
| `snn` | Spiking neural networks (neuron model, mode, state viewer) |
| `training` | Training SNNs (surrogate, curve, loss/accuracy) |
| `cnn` | Spiking CNNs (topology, graph viewer) |
| `recurrent` | Recurrent SNNs (delayed edges) |
| `nir` | NIR export, validation, and benchmarking |

Targets are marked with `data-tour` attributes on the panels. A step whose
target is gated (for example the trajectory viewer in production mode) shows
its explanatory note instead of a highlight. The help tips are expanded to
cover topology, neuron model, mode, and NIR concepts.

### WebSocket actions

The dashboard drives the server with the actions below; precondition
failures emit the existing `error` message rather than raising.

| Action | Reply | Mode / precondition |
|---|---|---|
| `configure` | `config_ack`, sample panels | needs an encode config |
| `run` / `stop` | `run_state`, rasters | needs a configured sample |
| `select_sample` | sample panels | needs a configured sample |
| `infer` | `inference` | needs a trained/loaded model |
| `train` / `stop_train` | `train_metrics`, `train_state` | builds the engine; applies `mode` |
| `save_model` | `model_saved` | needs a name |
| `load_model` | `model_loaded` | needs a saved name; applies `mode` |
| `delete_model` / `new_model` | `model_list` / `model_cleared` | — |
| `list_models` | `model_list` | — |
| `stats` | `system_stats` | — |
| `cancel_download` | `download_state` | — |
| `nir_export` | `nir_graph` | active or configured topology |
| `nir_validate` | `nir_validation` | needs a configured sample |
| `trajectory` | `trajectory` | **educational** mode + active model |
| `metrics` | `metrics` | **educational** mode + active model |
| `encoding_report` | `encoding_report` | needs a configured sample |
| `surrogates` | `surrogate_list` | — |
| `surrogate_curve` | `surrogate_curve` | needs a surrogate name |
| `benchmark` | `benchmark` | runs the tiny default fixture |

The schema also still declares a legacy `predict` type, but the client no
longer sends it and the server does not dispatch it.

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
  topology/                  Topology specs, presets, module builder
    spec.py                  TopologySpec: stages + edges (+ chain helpers)
    presets.py               fc_legacy / fc_small / conv_net / recurrent_net
    registry.py              name -> builder; build_topology/resolved_params
    builder.py               build_module(spec) -> snnTorch StageModule
  neurons/                   Neuron registry + canonical NIR param contract
    registry.py              NEURONS: name -> factory (incl. alpha); build()
    alpha.py                 snn.Alpha handler (simulation/introspection only)
    spike_grad.py            Optional `surrogate` param -> spike_grad callable
  simulator/                 The single temporal loop and trajectory capture
    execution.py             execute(...): one loop, both execution modes
    runner.py                run(module, spikes, mode=...) -> Trajectory
    compiled_step.py         Opt-in torch.compile wrapper + eager fallback
    production.py            run_production(...) -> ProductionResult
    trajectory.py            Per-stage S[t] / U[t] / I[t] traces
  introspection/             Educational mode: metrics, codings, surrogates
    metrics.py               trajectory_metrics(...) -> JSON-able metrics
    firing_rate.py           Per-stage firing rate
    sparsity.py              Per-stage sparsity
    isi.py                   Inter-spike-interval statistics
    histogram.py             Per-neuron firing-rate histograms
    encoding.py              encoding_report(...) + reconstruction
    decoding.py              Approximate per-coding image inverses
    surrogate.py             Surrogate registry + derivative curve
    comparison.py            compare_neurons(...) across every kind
  benchmark/                 Production-mode timing and memory harness
    config.py                BenchmarkConfig fixture
    harness.py               run_benchmark(...) -> JSON-able report
    timing.py                Warmup/repeat call timing
    memory.py                CUDA/RSS/tracemalloc snapshots
    __main__.py              python -m snn_interpreter.benchmark
  nir_bridge/                NIR export, independent interpreter, validation
    api.py                   The only module importing nir/nirtorch
    exporter.py              to_nir(spec, module); graph_summary(...)
    interpreter.py           NirInterpreter: runs a graph without snnTorch
    validator.py             validate(...) -> ValidationReport
    drift.py                 Error metrics between two trajectories
  cli/                       Headless commands
    verify.py                export / validate subcommands
  exporters/                 matplotlib/GIF/MP4 output -> build/
    exporter.py              Exporter base + build/ output resolution
    plot_utils.py            shared fig/GIF helpers
    *_exporter.py            per-visual exporters
  runtime/                   Compute environment
    device.py                CPU/GPU selection + auto benchmark
    execution_mode.py        Educational/Production execution flag
    system_stats.py          CPU RAM / GPU VRAM snapshots
server/
  app.py                     FastAPI app + WebSocket endpoint
  handlers.py                Inbound message routing (encode + train)
  protocol_handlers.py       Router for the NIR + introspection actions
  nir_handlers.py            nir_export / nir_validate handlers
  introspection_handlers.py  trajectory / metrics / encoding / surrogate
  introspection_payloads.py  JSON payload builders for introspection
  payloads.py                Model-list / model-load / NIR payload builders
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
  src/hooks/                 viewer state, encode config, model/tour actions
  src/components/            controls, charts, canvas panels (incl. training)
  src/tour/                  guided-walkthrough lessons + target highlighting
  src/styles/                split stylesheet (base/sections/controls/mode/
                             panels/analysis/tour/...)
```

Code is kept tidy by construction: modules are grouped into focused
subpackages, each Python file is under 250 lines, every Python function
stays under 20 lines, and each class lives in its own file.

## Notes

- Dataset downloads run in an isolated worker process, so the dashboard stays
  responsive: it shows a progress overlay with a live byte counter and a
  **Cancel** button instead of appearing frozen.
- The Bernoulli encoder in `spikegen.rate` is stochastic, so the reported
  spiking percentage and spike patterns vary run to run — expected.
- Larger `num_steps` values (e.g., 100) produce longer, richer animations;
  `subset`/`batch_size` trade dataset coverage for encode speed.
- **Interpreter limitations (Phase 1d).** snnTorch's `Lapicque` uses a
  first-order Euler update while the reference interpreter uses the exact
  zero-order-hold form, and `Synaptic`'s subtract reset carries a small
  residual; both residuals are reported in the `ValidationReport` rather
  than hidden. Extracting NIR graphs from arbitrary external modules via
  `nirtorch` is deferred to Phase 5.
- **Introspection limitations (Phase 2).** Four behaviours are deliberate.
  (1) `snn.Alpha` is simulation/introspection-only: the installed `nir` has
  no alpha-function primitive that matches its three-state dynamics, so
  exporting an `alpha` stage raises the typed `UnsupportedStageError`
  instead of inventing a lossy mapping. (2) `torch.compile` is opt-in
  because dynamo retraces on the shape-changing neuron state, so eager stays
  the default and `ProductionResult.status` makes a genuine compiled run
  distinguishable from a transparent eager fallback. (3) The decoding
  inverses are approximate: latency is quantised to integer steps and
  saturates at the threshold ceiling for sub-threshold pixels, delta
  integrates to a lower bound that is exact only when each step rises by
  exactly the threshold, and rate is a noisy Bernoulli estimate; `random`
  carries no image signal at all. (4) Upstream's `LSO` surrogate is listed
  because snnTorch exposes it, but applying it raises `TypeError` from
  upstream's wrapper.
- **Dashboard limitations (Phase 3).** (1) The hardware-target picker moves
  to Phase 5, so the LEFT column exposes topology, neuron, and surrogate
  controls only. (2) Event datasets are Phase 4, so the dataset
  walkthrough's "where event data lands" step is explanatory for now.
  (3) The trajectory viewer, metrics, and histogram are gated to educational
  mode and show an explanatory empty state in production.

## License

Released under the BSD 3-Clause License — see [`LICENSE`](LICENSE) and
[`AUTHORS`](AUTHORS).

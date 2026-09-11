# Spikeforge — PC-0: Production Use Cases (no neuromorphic chip)

> Umbrella for the ten chip-less production use cases. Each section is a
> **ticket seed**: enough scope to open an issue and architect it later. The
> shared hardware-free enablers live in
> [`production_toolkit_plan.md`](plans/production_toolkit_plan.md); the first
> use case is fully scoped in
> [`use_case_streaming_timeseries.md`](plans/use_case_streaming_timeseries.md).

Design/spec only. Every claim about current code is grounded in a file/line
reference so a Code-mode agent can execute this file-by-file.

**Objective.** Capture the ten realistic, chip-less production uses for SNN
models, tie each to the spikeforge pieces that already help and the toolkit
workstreams that unblock it, and open one umbrella plus one ticket per use case
to be tackled later. **No implementation starts here.**

---

## 1. How to read this

- **Chip-less feasibility** answers "does this need silicon?" — all ten are
  feasible on CPU/GPU/edge today; the neuromorphic chip adds power efficiency,
  not feasibility.
- **Enablers** reference the toolkit workstreams `W1…W7` from
  [`production_toolkit_plan.md`](plans/production_toolkit_plan.md:1).
- **Recommended order** is easiest-to-hardest so the first ticket (streaming
  time series) exercises the most shared machinery.

## 2. Use-case index

| # | Use case | Nearest fit in repo | Key enablers | Difficulty |
|---|---|---|---|---|
| 1 | Streaming time-series classification / anomaly detection | `sequence_mlp`, `fc_small`, `sequence_source` | W1, W2, W3 | **Lowest** |
| 2 | Always-on audio / keyword spotting / wake-word | event bridge, `EventSample` | W3, W5, W6 | Low–Med |
| 3 | Event-camera (DVS) vision | `dvs128_gesture`, `EventSpikeBridge` | W3, W6 | Medium |
| 4 | Ultra-low-latency sensor-stream inference | simulator single loop | W3, W6 | Medium |
| 5 | Anomaly / intrusion detection (IoT, network, grid) | as #1 + `event_runtime` | W1, W5 | Medium |
| 6 | Bio-signal & medical monitoring | `sequence_mlp`, `sequence_source` | W3, W6, W7 | Medium–High |
| 7 | RL for control & robotics | topology builder, NIR export | W1, W3 | Medium–High |
| 8 | Efficient sequence models / spiking transformers | `sequence_attn` (simulation-only) | W1, W6 | High |
| 9 | Edge/mobile inference under power budgets | `spikeforge-targets` quantize/energy | W5 (targets) | High |
| 10 | Computational neuroscience / neuromorphic R&D | `TopologySpec`, `bootology` registry, NIR | W1 | High |

## 3. Ticket seeds

### UC-1 — Streaming time-series classification / anomaly detection
- **What:** classify or flag anomalies on a continuous numeric stream (vibration,
  telemetry, network flows), one window at a time, with low latency.
- **Why SNN:** temporal memory + sparse, event-driven updates; no need to buffer
  large frame windows.
- **Chip-less:** fully CPU/GPU; the canonical "run it anywhere" case.
- **Repo fit:** `sequence_mlp` over `[T, B, L, D]` ([`sequence_presets.py`](spikeforge/topology/sequence_presets.py:1)),
  toy task ([`sequence_source.py`](spikeforge/data/sequence_source.py:1)),
  encoder ([`spike_encoder.py`](spikeforge/encoding/spike_encoder.py:1)).
- **Enablers:** W1 (stateful stream), W2 (encode-at-inference), W3 (serve).
- **Scope:** *fully specified* in [`use_case_streaming_timeseries.md`](plans/use_case_streaming_timeseries.md:1).

### UC-2 — Always-on audio / keyword spotting / wake-word
- **What:** continuous audio classification on a mic stream under a tight power
  budget (KWS, audio-event detection).
- **Why SNN:** streaming, low-power, event-sparse front end; a real deployed SNN niche.
- **Chip-less:** CPU/edge GPU; MFCC or event front end feeding the encoder.
- **Repo fit:** windowing via W7 I/O; event bridge
  ([`event_bridge.py`](spikeforge/events/event_bridge.py:1)); `fc_small`.
- **Enablers:** W3, W5, W6, W7.
- **Scope seed:** define an audio windowing + encoding spec, a public KWS dataset
  adapter, and a latency/power SLA.

### UC-3 — Event-camera (DVS) vision
- **What:** gesture/eye-tracking/automotive detection on asynchronous event
  streams.
- **Why SNN:** the input is already spikes; SNNs consume sparse events natively.
- **Chip-less:** GPU/CPU; strongest near-term traction.
- **Repo fit:** `dvs128_gesture`, `cifar10_dvs` ([`event_datasets.md`](documentation/event-datasets.md:1)),
  polarity rasters, `EventSpikeBridge`.
- **Enablers:** W3, W6; optionally a DVS-camera adapter (W7).
- **Scope seed:** topologies for `[T, 2, H, W]` events, event-native metrics, a
  recorded-stream replay harness.

### UC-4 — Ultra-low-latency sensor-stream inference
- **What:** per-sample decisions on radar/LiDAR/RF/vibration with hard latency
  bounds.
- **Why SNN:** temporal processing without buffering a full window; deterministic
  per-step latency.
- **Chip-less:** CPU/GPU; the value is scheduling/latency, not power.
- **Repo fit:** single temporal loop ([`execution.py`](spikeforge/simulator/execution.py:1)).
- **Enablers:** W3, W6 (p50/p99 benchmarks are the whole point).
- **Scope seed:** latency-budget harness, per-step scheduling, tail-latency CI gate.

### UC-5 — Anomaly / intrusion detection (IoT, network, grid)
- **What:** flag unusual patterns in telemetry without labels.
- **Why SNN:** temporal dynamics + low-power always-on monitoring.
- **Chip-less:** CPU/edge.
- **Repo fit:** as UC-1 plus the sparse runtime
  ([`sparse_runner.py`](spikeforge_targets/event_runtime/sparse_runner.py:1)) and
  energy estimates ([`accounting.py`](spikeforge_targets/energy/accounting.py:1)).
- **Enablers:** W1, W5.
- **Scope seed:** unsupervised/one-class objective, drift + false-positive budget.

### UC-6 — Bio-signal & medical monitoring
- **What:** EEG/ECG/EMG monitoring, seizure/arrhythmia detection, wearables.
- **Why SNN:** low latency at the sensor plus privacy-preserving on-device inference.
- **Chip-less:** CPU/wearable; **regulated** (e.g. MDR/FDA), so validation and
  traceability (W7) matter as much as accuracy.
- **Repo fit:** `sequence_mlp`; determinism ([`determinism.py`](spikeforge/tracking/determinism.py:82)).
- **Enablers:** W3, W6, W7.
- **Scope seed:** regulatory-grade evaluation, calibration, audit trail, PHI-safe I/O.

### UC-7 — Reinforcement learning for control & robotics
- **What:** SNN policies with temporal memory for control, trained in sim.
- **Why SNN:** recurrent temporal state in a compact, event-driven controller.
- **Chip-less:** simulation-trained; deployed on CPU/embedded.
- **Repo fit:** topology builder ([`builder.py`](spikeforge/topology/builder.py:1));
  `recurrent_net`; NIR export for cross-sim.
- **Enablers:** W1, W3.
- **Scope seed:** an env adapter + rollout harness; stability/sample-efficiency metrics.

### UC-8 — Efficient sequence models / spiking transformers
- **What:** long-sequence modeling with spiking attention.
- **Why SNN:** potential efficiency for very long sequences.
- **Chip-less:** GPU training/serving.
- **Repo fit:** `sequence_attn` — **simulation-only**, export refused with a
  typed `UnsupportedStageError` ([`stages_unmappable.py`](spikeforge/nir_bridge/stages_unmappable.py:1)).
- **Enablers:** W1, W6; NIR primitive gap is upstream (not ours).
- **Scope seed:** long-sequence benchmarks; define an exportable attention contract.

### UC-9 — Edge/mobile inference under power budgets
- **What:** deploy compressed SNNs on MCU/FPGA/edge CPU.
- **Why SNN:** sparse event-driven compute can undercut dense MACs at the edge.
- **Chip-less:** edge CPU/FPGA today.
- **Repo fit:** quantize ([`quantize_schemes.py`](spikeforge_targets/quantize_schemes.py:1)),
  energy estimates, NIR export.
- **Enablers:** W5 (compression + activation quant), targets runtime.
- **Scope seed:** a size/latency/power budget matrix and an edge export path.

### UC-10 — Computational neuroscience / neuromorphic R&D
- **What:** brain-circuit modeling and neuromorphic algorithm research.
- **Why SNN:** the model *is* spiking dynamics.
- **Chip-less:** HPC/GPU clusters.
- **Repo fit:** `TopologySpec`, neuron registry
  ([`neurons/registry.py`](spikeforge/neurons/registry.py:1)), NIR interpreter.
- **Enablers:** W1.
- **Scope seed:** biophysical neuron parameters, large-scale run harness, reproducibility.

## 4. Dependency map (use cases -> toolkit workstreams)

| Workstream | Unblocks |
|---|---|
| W1 stateful runtime + bundle | UC-1, UC-5, UC-7, UC-8, UC-10 |
| W2 encode-at-inference | UC-1 |
| W3 `spikeforge-serve` + clients | UC-2, UC-3, UC-4, UC-6, UC-7 |
| W5 compression + quantization | UC-2, UC-5, UC-9 |
| W6 observability + serving benchmarks | UC-3, UC-4, UC-6 |
| W7 registry + I/O adapters | UC-2, UC-3, UC-6 |

## 5. Recommended order

1. **UC-1** (streaming time series) — first, because it exercises W1+W2+W3, the
   smallest set of new primitives, and produces a reusable end-to-end reference.
2. UC-2 audio, UC-5 anomaly — reuse the UC-1 pipeline with a new I/O + objective.
3. UC-3 DVS, UC-4 latency — build on the replay harness and latency gate.
4. UC-6 medical, UC-7 RL — add regulation/traceability and an env adapter.
5. UC-8 sequence, UC-9 edge, UC-10 neuro — research-heavy; revisit after the toolkit lands.

## 6. Umbrella acceptance

One umbrella issue tracks the ten seeds; one fully scoped ticket exists for
UC-1; the remaining nine are described well enough to become tickets later. No
use case is implemented under this umbrella.

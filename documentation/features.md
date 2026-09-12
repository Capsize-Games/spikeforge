# Features

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
- **Event datasets (Phase 4)**: N-MNIST, DVS128 Gesture, CIFAR10-DVS, and
  Spiking Speech Commands through Tonic, with a modality-aware dataset
  picker, an event-to-spike bridge, and polarity-aware rasters (see below)
- **Targets and interoperability (Phase 5)**: a deployment-target registry
  with an honest capability matrix, per-target deployment reports, external
  NIR import/export, and a round-trip fidelity guarantee (see below)
- **Production workflows (Phase 6)**: a reproducibility manifest and config
  hash, a searchable checkpoint registry with metadata diffing, opt-in
  training scale-ups (AMP, gradient checkpointing, truncated BPTT,
  multi-GPU), a stored benchmark suite with regression gating, opt-in JSON
  logging and a metrics snapshot, packaged console scripts, and Docker
  CPU/GPU profiles (see below)
- **Model hub (WS-A)**: a bundled curated catalog (10 verified entries across
  five frameworks) plus optional live Hugging Face access, an isolated
  downloader with progress/cancel and checksum verification, and an
  inspect → compat → promote import funnel, surfaced through `spikeforge-hub`, six
  WebSocket actions, and the `HubPanel` browser (see below)
- **Backend execution (WS-B)**: a substitution executor that applies a
  target's declared rewrites with a report and drift check, and executable
  `reference`, `norse`, and `lava_loihi2` backends behind one `compile_run`
  entry point, surfaced through `deploy`/`rewrite`/`run` (see below)
- **Sequence primitives (WS-C)**: per-stage heterogeneous neurons, ten new
  stage kinds with explicit NIR contracts, and the `sequence_mlp`/`sequence_attn`
  demonstration presets (see below)
- **Event runtime and energy (WS-D)**: a sparse/event-driven runner with a
  dense-parity check, SOP/MAC/AC counting, and a `spikeforge-energy` report that maps
  op counts to a declared per-target cost table (see below)
- **Operational maturity (WS-E)**: opt-in persisted metrics, optional
  TensorBoard/W&B tracking sinks, determinism tooling, and a generated docs
  site (see below)
- **Interop fold-ins (WS-F)**: event-dataset training, an ONNX bridge,
  `nirtorch` extraction of third-party PyTorch modules, weight-level
  quantization, non-square sensor geometry, and per-step hidden-layer
  animation (see below)

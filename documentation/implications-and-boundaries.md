# Implications and boundaries

This section states, for the six cross-cutting limitations that shape real
use, **what** each one is, **why** it exists, and **what it implies for you**.
The per-phase "Limitations" notes elsewhere in this README describe
phase-specific behaviour; when a limit reaches beyond one phase it is reasoned
about once, here, and those notes link back to it instead of restating the
consequence.

### 1. Energy and latency are modelled, not measured

- **Why it exists.** The tool runs on x86/GPU through PyTorch, which exposes
  no per-operation energy counters. A real energy number needs a physical
  neuromorphic device with on-board instrumentation (for example Loihi 2
  energy probes) or an external power monitor, and none is attached.
- **What it does instead.** It counts operations deterministically (SOP/MAC/AC
  and timesteps) and multiplies them by per-target cost coefficients declared
  from published, cited literature.
- **Implication.** Every number is an order-of-magnitude *planning* estimate:
  it is sound for comparing models and targets and for sparse-vs-dense
  trade-offs, but it is **not** valid for procurement, thermal, or power-budget
  guarantees. Reports are labelled `estimate: true` with a `basis` and a
  `source`; a single integration point (`account(..., measurement=...)`, or a
  target probe) lets a real device report a measurement and flip the label to
  `estimate: false` without changing any other code.

### 2. `sequence_attn` is simulation-only

- **Why it exists.** The installed `nir` standard has no primitives for
  embedding, attention, layer-norm, or positional encoding, so a faithful
  export is impossible; inventing a lossy mapping would violate the project's
  honesty rule and could silently misbehave on hardware.
- **What it does instead.** It keeps those stage kinds available for
  simulation and raises a typed `UnsupportedStageError` that names the first
  unexportable stage when export is attempted.
- **Implication.** You can build, train, and introspect `sequence_attn` in the
  snnTorch simulator, but you cannot export it to NIR or deploy it:
  `spikeforge-verify export`/`validate --topology sequence_attn` exit non-zero with the
  named stage (for the shipped preset, `embedding`). `sequence_mlp` is the
  NIR-exportable sequence preset because it uses only mappable kinds.

### 3. Quantization is simulated, not device-exact

- **What it does.** Weights are restricted in the graph itself: the weight
  tensors of affine/conv NIR nodes go onto the target's declared scheme (int8
  symmetric per-tensor, uint8 asymmetric per-tensor) with a per-layer
  before/after range report, and the original graph is never mutated.
  Activations and membranes cannot be restricted in a NIR graph, so they are
  quantized *at execution time* instead: the post-quantize drift check runs
  the quantized graph through the reference interpreter under a hook that
  snaps every computed node output, every carried membrane and synaptic
  current, and the membrane each step records onto a symmetric fixed-point
  grid, and the same schemes serve a bundle through `InferenceSession`. The grid is fixed from a calibration
  (folded from the drift fixture itself unless a calibration dataset's ranges
  are supplied), and the report records per-tensor ranges, the error each
  grid introduced, and under `drift.includes` which roundings the drift
  figure covers.
- **Why the boundary remains.** The snap happens after each node's floating
  update, so the threshold comparison still sees a float membrane: the grid
  is applied to the values a step produces, never to the arithmetic that
  produced them. Integer accumulation, accumulator overflow, per-channel
  schemes, and the vendor compiler's numerics are not modelled, and executing
  a quantized graph still requires the target SDK. No shipped
  target declares an activation scheme: the fixed-point widths a vendor's
  neuron state actually uses are not verified in this repository, so
  `activation_quantization` is `none` on every target and a caller opts in
  explicitly (`quantize(..., activation=...)`,
  `spikeforge-verify deploy --activation-quantization <scheme>`).
- **Implication.** The drift check can include the rounding a fixed-point
  device applies to what flows through the network, not only to its weights,
  so it no longer understates that part of deployment error — but it is still
  an *estimate* of a target's precision impact, never a reproduction of the
  device. A target whose declared weight scheme is `none` is a reported
  no-op, an unsupported scheme (for example `int4`) is reported unapplied
  rather than silently ignored, and an activation scheme requested without a
  spike fixture is reported unapplied because there is nothing to simulate
  on.
- **What this page got wrong.** From the day it was written, 2026-09-11 —
  the same day `spikeforge-targets` began shipping the serving-side
  `ActivationQuantizer` and its calibration hook — until 2026-09-15 this
  section said there was "no activation or membrane quantization, no
  calibration dataset". Both existed on the serving path the whole time;
  what did not exist until 2026-09-15 was any of it in the target drift
  check, which stayed weight-only.

### 4. ONNX is a single-step structural bridge

- **What it does.** It exports the per-step module (the time loop stays in the
  simulator) stamped with topology/spec/temporal metadata, and imports by
  reading that metadata exactly or by mapping a limited op set (`Gemm`/`MatMul`,
  `Conv`, `Flatten`, `AveragePool`, `Dropout`, and the `Identity` passthrough),
  rejecting every other op by name. The round-trip reports `identical` only on
  structural equality.
- **Implication.** An exported ONNX file is **not** a complete temporal SNN:
  running it in another runtime will not reproduce multi-timestep dynamics.
  Importing an arbitrary ONNX model works only for the mapped op set, and
  neuron-internal ops (`Greater`/`Sub`/`Clip`) are rejected by name — so a
  model exported *with* metadata round-trips through the bridge, while the op
  mapper only accepts the mapped set. Use it for interop and visualization of
  the feed-forward math and for structural round-trips, not for cross-runtime
  time execution.

### 5. Only this project's own weights are redistributed

- **What it does.** The catalog holds three kinds of entry, and the `source`
  field says which. `bundled` renders a NIR graph from a shipped topology
  preset — structure with freshly-initialised weights. `reference` carries
  **trained weights this project produced itself**, shipped inside the
  `spikeforge-hub` wheel under `weights/` and verified against the checksum the
  catalog pins. `url`/`hf_repo` point at bytes elsewhere, which download into a
  cache on first use.
- **Implication.** No *third-party* weights are redistributed here, for
  licensing and size reasons, so for anything you add yourself **you** are
  responsible for honouring its upstream license. The reference checkpoints are
  the exception and are this project's own artifacts under its own
  BSD-3-Clause terms — but the datasets they were trained on are not: each
  entry records the training data's own `dataset_license` and
  `dataset_attribution` separately from the `license` covering the weights, and
  `NOTICE.md` reproduces them. An entry with no published checksum verifies as
  `unverified` rather than trusted, network access (plus the `hub` extra for
  Hugging Face) is required for remote artifacts not already cached, and
  integrity checking for those is best-effort.
- **What this page got wrong.** Until 2026-09-15 this section said the
  repository "does not redistribute weights" without qualification. That
  stopped being true when `spikeforge-hub` 0.2.0 began shipping six trained
  checkpoints; `NOTICE.md` and the curation policy were updated then and this
  page was not.

### 6. Hardware backends are declared until their SDKs are installed

- **Why it exists.** The hardware and simulator SDKs are heavyweight and
  platform-specific and are not installed.
- **Implication.** `reference` is always available; `norse`, `lava_loihi2`,
  `spinnaker2`, `speck`, and `xylo` report `available: false` with a named
  reason (see `spikeforge-verify targets`), and no deployment report claims a device
  result that was not produced — `compile_run` returns `status: "unavailable"`
  with a note naming the enabling extra. Note that `norse` is pip-installable
  and pure-PyTorch, so it is usable for CPU cross-checking even without any
  neuromorphic hardware.

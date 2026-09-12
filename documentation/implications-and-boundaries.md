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

### 3. Quantization is weight-level only

- **What it does.** It restricts the weight tensors of affine/conv NIR nodes
  to the target's declared scheme (int8 symmetric per-tensor, uint8 asymmetric
  per-tensor), with a per-layer before/after range report and an optional
  post-quantize drift check. The original graph is never mutated.
- **Implication.** There is no activation or membrane quantization, no
  calibration dataset, no integer-accumulation/saturation modelling, no
  per-channel schemes, and no device kernel — so it *estimates* a target's
  precision impact; it does not reproduce the vendor compiler's numerics, and
  executing a quantized graph still requires the target SDK. A target whose
  declared scheme is `none` is a reported no-op, and an unsupported scheme
  (for example `int4`) is reported unapplied rather than silently ignored.

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

### 5. Hub weights are fetched on demand

- **What it does.** The catalog stores metadata (source, expected size, and a
  checksum when one is published) for **verified entries only**; bundled NIR
  preset graphs materialize locally and remote artifacts a user chooses to add
  download into a cache on first use.
- **Implication.** The repository does not redistribute weights, for licensing
  and size reasons, so **you** are responsible for honouring each model's
  license. An entry with no published checksum verifies as `unverified` rather
  than trusted, network access (plus the `hub` extra for Hugging Face) is
  required unless the artifact is already cached, and integrity checking is
  best-effort.

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

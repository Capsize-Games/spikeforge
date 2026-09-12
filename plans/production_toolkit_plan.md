# Spikeforge — PT-0: Hardware-Free Production Toolkit

> Umbrella design for the production-toolkit program. Companion documents:
> the user-facing use-case umbrella
> [`production_use_cases.md`](plans/production_use_cases.md) and the first
> fully scoped application
> [`use_case_streaming_timeseries.md`](plans/use_case_streaming_timeseries.md).
> Read this file first: it defines the shared interfaces every use case consumes.

Design/spec only. Every claim about current code is grounded in a file/line
reference so a Code-mode agent can execute this file-by-file.

**Objective.** Close the gaps identified in the "Q1 / Q2" analysis so a user can
go from `train` to a running, observable service **and** a simulator-backed test
deploy **without owning a neuromorphic chip**. Today the repo trains well and
declares deployment honestly, but the path from a checkpoint to a served model
is missing the SNN-specific pieces (stateful inference, a portable artifact,
encoder-at-inference, a headless serving runtime) and the chip-less simulators
(Sinabs/Rockpool/SpiNNaker) are declared but not wired.

---

## 1. Scope

Two tracks, both hardware-free:

- **Track A — runtime & serving.** Stateful inference, the deployment bundle,
  the encode-at-inference contract, a headless inference service, client SDKs,
  compression/quantization extensions, observability exporters, a governed
  registry, I/O adapters, and serving-level benchmarks.
- **Track B — simulator-backed test deploy.** Keep `reference`, Norse, and the
  Lava Loihi 2 CPU emulator; wire the SynSense and SpiNNaker software
  simulators; unify them into one `test-deploy` matrix with an honest report.

**Explicitly out of scope** (and why): measured power/timing and on-device
capacity (needs silicon); a full temporal SNN in ONNX (ONNX has no temporal
spiking semantics by design); reproducing vendor compilers' numerics. These are
recorded in [Implications and boundaries](documentation/implications-and-boundaries.md:1)
and stay there.

---

## 2. Current state and gap analysis

| Capability | Current reality | Anchor | Gap to close |
|---|---|---|---|
| Closed-loop inference | `run(module, spikes, ...)` consumes a whole `[T, …]` tensor | [`runner.py`](spikeforge/simulator/runner.py:21) | no per-step API |
| Temporal loop | one loop, two modes; `Trajectory` holds `S/U/I` | [`execution.py`](spikeforge/simulator/execution.py:1), [`trajectory.py`](spikeforge/simulator/trajectory.py:9) | loop body not exposed as a reusable step |
| Inference entry point | `infer_spikes` returns **UI payloads** (rasters) | [`network/inference.py`](spikeforge/network/inference.py:15) | no plain scoring API |
| Server surface | `/ws`, `/health`, static mount | [`server/app.py`](server/app.py:130), [`server/web.py`](server/web.py:57) | no REST/streaming predict, auth, batching |
| Session | per-connection dashboard state | [`server/session.py`](server/session.py:1) | not a model-serving session |
| Checkpoint | weights + meta (`topology`, `topology_params`) | [`checkpoint_mixin.py`](spikeforge/training/checkpoint_mixin.py:75), [`model_store.py`](spikeforge/network/model_store.py:41) | no single deploy bundle; encode config not a serving contract |
| Manifest | config hash, seed, versions, metric history | [`manifest.py`](spikeforge/tracking/manifest.py:35) | reproducibility, not a runtime contract |
| NIR serialization | version-stamped graph envelope | [`serialization.py`](spikeforge/nir_bridge/serialization.py:52) | not bundled with weights/encoder |
| ONNX | single forward step + metadata | [`onnx_bridge/export.py`](spikeforge/onnx_bridge/export.py:1) | not a temporal runtime (by design) |
| Encoding | `SpikeEncoder` owns rate/latency/delta/random | [`spike_encoder.py`](spikeforge/encoding/spike_encoder.py:1), [`encode_config.schema.json`](protocol/payloads/encode_config.schema.json:1) | encoding not frozen into the artifact |
| Targets | 6 declared; `reference`/`norse`/`lava_loihi2` executable | [`catalog.py`](spikeforge_targets/catalog.py:145), [`backends/api.py`](spikeforge_targets/backends/api.py:20) | `speck`/`xylo`/`spinnaker2` not wired |
| Lava path | defaults to **Loihi 2 CPU emulator**, device opt-in | [`lava_backend.py`](spikeforge_targets/backends/lava_backend.py:33) | document it; broaden lowering |
| Energy | declared tables, `measured: false`; measurement hook exists | [`accounting.py`](spikeforge_targets/energy/accounting.py:1) | nothing (honest by design) |
| Quantization | weight-only schemes | [`quantize.py`](spikeforge_targets/quantize.py:103), [`quantize_schemes.py`](spikeforge_targets/quantize_schemes.py:1) | no activation/membrane quant |
| Compression | none | — | no pruning / sparsity tooling |
| Benchmarks | training-step timing/memory + store/compare | [`benchmark/`](spikeforge/benchmark/__init__.py:1) | no serving p50/p99/throughput |
| Observability | opt-in logs; in-process registry | [`observability/`](spikeforge/observability/__init__.py:1) | no Prometheus/OTel exporters |
| Registry | file store + search/diff | [`model_store.py`](spikeforge/network/model_store.py:1) | no stages/approvals/signing/lineage |

### 2.1 Invariants that must not break

- `reference` stays available unconditionally ([`catalog.py`](spikeforge_targets/catalog.py:61)); every report stays `estimate: true` where applicable.
- The closed-loop `run()` / `run_production()` surfaces, the WebSocket protocol ([`protocol/`](protocol/README.md:1)), and all existing payload keys stay **additive**.
- Core stays **headless and dependency-light**: no `fastapi`/`uvicorn`/`pydantic` in `spikeforge` (enforced by [`check_core_boundary.py`](scripts/check_core_boundary.py:1)); SDK imports stay confined to isolated probes ([`backends/api.py`](spikeforge_targets/backends/api.py:1)).
- House style: one class per file, ≤250-line files, ≤20-line functions, 79 columns ([`rules.md`](rules.md:1)).

---

## 3. Track A — runtime & serving

### 3.1 W1 — Stateful inference runtime (`spikeforge/serving/`)

The keystone. SNN inference is stateful across timesteps; today that state is
internal to the closed loop, so no streaming service can drive it.

- **New package `spikeforge/serving/`** (core; torch-only). Note: `spikeforge/runtime/`
  already exists (device, execution_mode, system_stats), so the serving API must
  not reuse that name.
- **API:**
  - `InferenceSession.load(bundle, device=..., mode=PRODUCTION)` — build the module from
    the bundle's spec and load weights.
  - `.reset()` — zero every carried neuron state.
  - `.step(frame) -> Prediction(logits, spikes, class_totals)` — advance one timestep.
  - `.run_stream(frames) -> Iterator[Prediction]` — drive `step` over an iterable.
  - `.state()` / `.load_state(state)` — JSON/tensor-serializable carried state.
- **Mechanism:** extract the per-step body now inside
  [`execution.py`](spikeforge/simulator/execution.py:1) into a shared
  `step_stages(module, inputs, state) -> (outputs, state)` that **both** the
  closed loop and `InferenceSession` call, so they cannot diverge. `Trajectory`
  remains unchanged for closed runs.
- **State object:** a `StateTree` (one name -> tensor per stage) with
  `reset()`, `to_dict()`/`from_dict()` (numpy-tagged like
  [`array_codec.py`](spikeforge/nir_bridge/array_codec.py:1)), and a device move.
- **Acceptance:** `session.run_stream` over a full sequence produces readouts
  equal (within tolerance) to `run(module, spikes)`; `.state()` round-trips; a
  reset between sequences reproduces a fresh run.

### 3.2 W1 — Deployment bundle (`DeploymentBundle`)

One portable artifact a runtime can load, replacing today's scattered
`checkpoint + manifest + NIR envelope + ONNX` set.

- **Format:** a zip `model.spkf` with:
  - `manifest.json` — `TopologySpec`, resolved stage params, library versions,
    expected metrics, label map, protocol/app version.
  - `weights.pt` — the `state_dict`.
  - `encode_config.json` — the frozen `EncodeConfig` ([`encode_config.py`](server/schemas/encode_config.py:1)).
  - `preprocessing.json` — normalization/windowing (W2).
  - `graph.nir.json` — optional NIR envelope ([`serialization.py`](spikeforge/nir_bridge/serialization.py:52)).
  - `SHA256SUMS` + optional detached signature.
- **Builder:** `spikeforge.serving.bundle.build(checkpoint, encode_config, ...)`
  and a CLI `spikeforge-serve bundle --from-model <name> --out model.spkf`.
- **Reuse:** `ReproducibilityManifest` ([`manifest.py`](spikeforge/tracking/manifest.py:35))
  for provenance; `TopologySpec` for rebuild; schema version pinned like
  [`compatibility.json`](compatibility.json:1).
- **Acceptance:** a bundle rebuilds the exact module + weights on a fresh
  process; loading rejects a tampered bundle with a typed error.

### 3.3 W2 — Encode-at-inference contract

Prevents train/serve skew, the most common production failure for coded inputs.

- Promote encoding to a frozen, executable **preprocessing spec** stored in the
  bundle (W1). The training path and serving path must call the **same** pure
  function `spikeforge.serving.preprocess.encode(sample, spec) -> spikes`
  backed by [`spike_encoder.py`](spikeforge/encoding/spike_encoder.py:1).
- Add `encode_config` + `preprocessing` to the checkpoint's `_meta()`
  ([`checkpoint_mixin.py`](spikeforge/training/checkpoint_mixin.py:75)) so the
  artifact is self-describing; existing keys stay.
- Also ship the **label map** and input geometry
  ([`image_size.py`](spikeforge/data/image_size.py:16)).
- **Acceptance:** encoding the same sample at train time and serve time yields
  byte-identical spike tensors; a mismatched spec version is refused.

### 3.4 W3 — `spikeforge-serve` (new distribution)

A headless inference service, separate from the dashboard server (which is
session/WebSocket oriented, [`server/session.py`](server/session.py:1)).

- **Distribution:** `spikeforge-serve`, import root `spikeforge_serve`; depends
  on core (+ optional `spikeforge-targets` for lowered execution). `fastapi`/
  `uvicorn` are **this** distribution's deps, never core's.
- **Endpoints:**
  - `POST /v1/predict` — batch of pre-encoded or raw frames -> logits/label.
  - `POST /v1/reset` — reset a session id's temporal state.
  - `GET|WS /v1/stream` — streaming `step` for continuous sensors.
  - `GET /health`, `GET /metrics`.
  - `GET /v1/bundle` — loaded bundle metadata (spec, versions, metrics).
- **Runtime semantics:** session store keyed by id; bounded batching; concurrency
  limits; request timeouts; backpressure on the stream; optional bearer auth.
- **CLI:** `spikeforge-serve --bundle model.spkf --host 0.0.0.0 --port 8899`.
- **Container:** a minimal inference image (distinct from the dashboard
  [`Dockerfile`](Dockerfile:1)) + a compose profile.
- **Acceptance:** `predict` matches the in-process reference; `stream` maintains
  state across frames and resets on demand; `/metrics` exposes W6 metrics; auth
  rejects an unauthenticated call.

### 3.5 W4 — Client SDKs

- `spikeforge-clients`: a small Python client + a TypeScript client + a CLI,
  generated/validated against an OpenAPI schema so the service and clients
  cannot drift. Mirrors the existing protocol-codegen discipline
  ([`protocol/codegen/generate_ts.mjs`](protocol/codegen/generate_ts.mjs:1)).
- **Acceptance:** a round-trip test drives `predict` and `stream` from the
  Python client and asserts parity with the server.

### 3.6 W5 — Compression & quantization extensions

- **Compression (`spikeforge/compression/`):** magnitude and structured pruning,
  sparsity schedule/report (weight density), and lossless export to the bundle;
  a before/after drift check reusing [`drift.py`](spikeforge/nir_bridge/drift.py:1).
- **Quantization (`spikeforge-targets`):** add activation/membrane schemes +
  a calibration-dataset hook alongside the existing weight-only path
  ([`quantize_schemes.py`](spikeforge_targets/quantize_schemes.py:1)); keep
  weight-only the default; report unapplied schemes rather than ignoring them.
- **Acceptance:** pruning a topology then exporting/validating reports the
  sparsity gained and the induced drift; a new quantization scheme reports
  per-layer ranges and is refused honestly when unsupported.

### 3.7 W6 — Observability & serving benchmarks

- **Exporters:** add Prometheus and OpenTelemetry exporters over the existing
  `MetricsRegistry` ([`observability/registry.py`](spikeforge/observability/registry.py:1));
  add request-level tracing to `spikeforge-serve`. Persistence stays opt-in
  (`SPIKEFORGE_METRICS_PERSIST`).
- **Serving benchmarks:** extend [`benchmark/`](spikeforge/benchmark/__init__.py:1)
  with a serving mode reporting p50/p99 latency, throughput at concurrency N,
  cold-start, and peak memory; feed the existing regression gate
  ([`compare.py`](spikeforge/benchmark/compare.py:123)).
- **Acceptance:** a CI job asserts p99 latency and throughput stay within a
  configured threshold.

### 3.8 W7 — Registry governance & I/O adapters

- **Registry:** extend the model store + hub import with explicit stages
  (`dev -> staging -> prod`), approvals, artifact signing/verification, and
  lineage (dataset -> model -> bundle -> deployment). Build on
  [`model_search.py`](spikeforge/network/model_search.py:87),
  [`model_diff.py`](spikeforge/network/model_diff.py:102), and
  [`spikeforge_hub/import_model.py`](spikeforge_hub/import_model.py:1).
- **I/O (`spikeforge-io`):** windowing/normalization plus adapters for event
  cameras, audio, MQTT, and Kafka, feeding `spikeforge-serve`'s stream.
- **Acceptance:** a bundle can be promoted with a recorded approver and verified
  signature; an I/O adapter replays a recorded stream into `/v1/stream`.

---

## 4. Track B — simulator-backed test deploy

The repo already executes against CPU simulators; this track completes the
matrix so "test deploy" is a first-class, documented experience.

| Target | Simulator (no chip) | Install | Status now | Work |
|---|---|---|---|---|
| `reference` | in-process NIR interpreter | — | done | — |
| `norse` | Norse pure-PyTorch | `spikeforge-targets[norse]` | done | — |
| `lava_loihi2` | Lava `Loihi2SimCfg` CPU emulator | `spikeforge-targets[lava]` | done (default) | document; broaden lowering past linear chains |
| `speck` | Sinabs / Speck simulator | vendor SDK | declared only | isolated probe + backend |
| `xylo` | Rockpool Xylo simulator | vendor SDK | declared only | isolated probe + backend |
| `spinnaker2` | sPyNNaker / py-spinnaker2 host sim | vendor SDK | declared only | isolated probe + backend |

- **Design:** each backend follows [`backends/api.py`](spikeforge_targets/backends/api.py:1)
  (isolated import, `available()`, `compile()`, `run()`) and returns a
  `BackendResult` with a `path` + notes that name the emulator
  (pattern: [`lava_backend.py`](spikeforge_targets/backends/lava_backend.py:33)).
- **Unified matrix:** a `spikeforge-targets test-deploy` command that runs every
  *available* simulator, compares each to `reference` with
  [`backends/compare.py`](spikeforge_targets/backends/compare.py:1), and emits a
  JSON report; a CI job runs it with whichever extras install.
- **Honesty:** an emulator run is never reported as a device measurement; energy
  stays `estimate: true` until a device reports its own timing
  ([`accounting.py`](spikeforge_targets/energy/accounting.py:1)).
- **Acceptance:** with `[lava,norse]` installed, one command test-deploys a
  linear topology on `reference`, `norse`, and `lava_loihi2` (CPU emulator) and
  reports parity; absent SDKs report `available: false` with a named reason.

---

## 5. Packaging and topology

| Distribution | Import root | Depends on | Change |
|---|---|---|---|
| `spikeforge` | `spikeforge` | torch stack only | add `serving/`, `compression/`, encode-at-inference, exporters |
| `spikeforge-targets` | `spikeforge_targets` | core | add simulator backends, activation quant, `test-deploy` |
| `spikeforge-serve` | `spikeforge_serve` | core (+ targets) | **new** |
| `spikeforge-io` | `spikeforge_io` | core | **new** |
| `spikeforge-clients` | `spikeforge_clients` | core (schema only) | **new** |
| `spikeforge-registry` | `spikeforge_registry` | core + hub | **new** (or extend hub) |

Rules carried over from [ARCH-0001](plans/arch-0001-target-topology.md:1): core
never depends on a satellite; satellites pin `spikeforge~=0.3.0` and record it in
[`compatibility.json`](compatibility.json:1); SDK imports stay in isolated probes;
everything degrades with a typed error.

---

## 6. Workstream breakdown (ticket seeds)

| ID | Workstream | Deliverable | Depends on |
|---|---|---|---|
| W1 | Stateful runtime + bundle | `spikeforge/serving/` step API + `.spkf` bundle | — |
| W2 | Encode-at-inference | frozen preprocessing spec + shared encode fn | W1 |
| W3 | `spikeforge-serve` | REST + streaming service, auth, container | W1, W2 |
| W4 | Client SDKs | Python + TS + CLI, OpenAPI-validated | W3 |
| W5 | Compression + quantization | pruning/sparsity + activation schemes | W1 |
| W6 | Observability + serving benchmarks | Prometheus/OTel exporters + p50/p99 | W3 |
| W7 | Registry governance + I/O | stages/approvals/signing + adapters | W1 |

---

## 7. Acceptance bar

A user with no neuromorphic hardware can: train a topology, export a
`DeploymentBundle`, serve it with `spikeforge-serve`, call it from a client SDK,
read Prometheus metrics, and **test-deploy** the same model on `reference`,
`norse`, and the Lava CPU emulator with a parity report — all with the existing
evidence rules intact (`estimate: true`, `available: false` when an SDK is
absent, additive protocol only).

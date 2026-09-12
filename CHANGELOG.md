# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project has **not** been published to any package index; the versions below
describe the local source tree.

## [Unreleased]

### Added

- **`weight_decay` on `TrainingEngine`.** A new constructor parameter
  (default `0.0`, matching prior behaviour) is forwarded straight into
  the Adam optimizer and recorded in checkpoint metadata and the
  reproducibility manifest's hyperparameter block, alongside `lr`.
  Every subclass (`EventTrainingEngine` included) already forwards
  unrecognised keyword arguments, so no subclass changes were needed.
- **`dropout` on the `conv_net` topology preset.** A new `dropout`
  parameter (default `0.0`, the identity) adds a `dropout`-kind stage
  between the flattened features and the readout, using the stage
  system's existing generic `dropout` kind (already NIR-export-safe
  as a passthrough at inference). Reaches training the same way
  every other `conv_net` param does, via `topology_params`.

### Fixed

- **Copyright holder corrected to `Capsize LLC`.** `LICENSE`, `AUTHORS`,
  `NOTICE.md`, `CITATION.cff`, and every distribution's `pyproject.toml`
  named `Capsize Games` as the copyright holder/author, which is not the
  actual legal entity. Corrected across all of them; the entry below
  (2f6e3ec) documented the wrong entity name at the time and is left as
  the historical record rather than rewritten.
- **`sequence_mlp` CUDA warmup crashed with a shape mismatch.**
  `TopologyMixin._input_features()` only ever checked `input_size`
  (the image-shaped presets' param name); `sequence_mlp` has no
  `input_size` at all and names the same concept `features`, so it
  silently fell through to the historical `28 * 28` MNIST-shaped
  default. Invisible on CPU (`device.warmup()` no-ops there), but a
  real CUDA run's warmup pass forwarded a wrongly-shaped dummy tensor
  and crashed. Now checks `features` before that fallback.

## [spikeforge-targets-v0.1.1] - 2026-09-11

A `spikeforge-targets`-only follow-up release. Core stays `0.3.0` and every
other distribution stays `0.1.0`; the wire protocol stays at `1.0`.

### Added

- **Vendor simulator backends (Speck, Xylo, SpiNNaker2).** The
  simulator-backed test-deploy matrix gains an executable backend for every
  registered target: Speck runs through Sinabs, Xylo through Rockpool, and
  SpiNNaker2 through its host simulator. Each follows the existing backend
  protocol — availability comes from the isolated SDK probe plus a minimal
  capability check, the graph is lowered with the shared linear lowering, and
  a missing SDK yields `available: false` with a named reason rather than a
  failure.
- A shared `VendorBackend` adapter
  (`spikeforge_targets.backends.vendor_backend`) and the `linear_program`
  dense/neuron lowering in `spikeforge_targets.backends.lowering`, reused by
  the Norse, Lava, and vendor backends so one code path recovers the execution
  order and sizes each layer.
- A `test-deploy` CI job that runs the matrix for `fc_legacy`, asserts exactly
  one cell per registered target, and checks the command exits zero when every
  SDK-backed cell honestly reports `available: false`.

### Changed

- `BackendResult` and the matrix `DeployCell` now carry `estimate: true`: every
  simulator or emulator run is labelled an estimate and only a real device
  result can set it `false`, so a simulated trajectory is never read as a
  measurement. A cell now reports availability from its wired backend
  (`backend.available()`) instead of the registry probe.

## [0.3.0] - 2026-09-11

The hardware-free production toolkit (workstreams PT-W1…PT-W8) and the
streaming time-series use case UC-1, on top of the ARCH-0001
repository-topology work recorded below. Every capability below is additive:
the wire protocol stays at `1.0`, the core import root stays `spikeforge`,
and the existing WebSocket payload keys are preserved.

### Added

- **Serving runtime and the `.spkf` deployment bundle (PT-W1).** A stateful
  inference runtime with a `StateTree` that carries per-layer neuron state
  across steps, and the `.spkf` `DeploymentBundle` that packages a trained
  model with its frozen encode configuration and target plan.
- **Frozen encode-at-inference contract (PT-W2).** The window → normalize →
  encode pipeline is serialized into the bundle and replayed byte-for-byte at
  inference, so a served model encodes inputs exactly as it did in training.
- **Headless `spikeforge-serve` (PT-W3).** A REST/WebSocket inference service
  that loads a `.spkf` bundle and exposes `/v1/encode`, `/v1/predict`, and
  `/v1/stream`, packaged as the `spikeforge-serve` distribution.
- **Client SDKs (PT-W4).** Python, TypeScript, and CLI clients for
  `spikeforge-serve` in the dependency-light `spikeforge-clients`
  distribution, validated against the `protocol/serve/` schemas so an install
  never pulls the torch-based core.
- **Compression and quantization (PT-W5).** Delta/spike codecs plus activation
  and membrane quantization of a deployed model, with reports that record the
  scheme and the achieved size/accuracy trade-off.
- **Observability and serving benchmarks (PT-W6).** Prometheus and JSON
  exporters for the serving metrics and a benchmark harness that measures
  encode, inference, and end-to-end latency for a bundle.
- **Registry governance and I/O adapters (PT-W7).** Stage/approver/signature/
  lineage governance for model promotion in `spikeforge-hub`, and the
  `spikeforge-io` recorded-stream adapters (CSV/JSON/NPY/in-memory and the
  dataset hook) that feed the frozen windowing contract and replay into
  `/v1/stream`.
- **Simulator-backed test-deploy matrix (PT-W8).** A matrix that lowers each
  target through the reference/norse/lava simulators and records the per-target
  deploy outcome before a bundle is promoted.
- **Streaming time-series use case (UC-1).** An end-to-end
  classification/anomaly-detection MVP that trains, bundles, serves, and
  replays a sliding-window time-series signal.
- **Project rename to `spikeforge`.** The project, its core distribution, its
  import root, and its console scripts all read `spikeforge`; the earlier
  bring-up name is gone from every package, repo, and document.
- **Move to the `capsize-games` organization (four repositories).** The project
  now lives under [`capsize-games`](https://github.com/capsize-games):
  [`capsize-games/spikeforge`](https://github.com/capsize-games/spikeforge)
  (core; ARCH-0001 issue #1),
  [`capsize-games/spikeforge-dashboard`](https://github.com/capsize-games/spikeforge-dashboard),
  [`capsize-games/spikeforge-targets`](https://github.com/capsize-games/spikeforge-targets),
  and [`capsize-games/spikeforge-hub`](https://github.com/capsize-games/spikeforge-hub).
  The `spikeforge-server` repository is a deliberate no-go (trigger T4 did not
  fire); the server ships as a distribution in the core repository.
- **One-command install.** [`install.sh`](install.sh) installs all four
  distributions editable from a clone; once the distributions are published,
  `pip install "spikeforge[all]"` installs the library bundle (core + targets +
  hub) and `pip install spikeforge-server` adds the server.
- Open-source readiness scaffolding: [`CONTRIBUTING.md`](CONTRIBUTING.md),
  [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), [`SECURITY.md`](SECURITY.md),
  [`NOTICE.md`](NOTICE.md), issue and pull-request templates, and a
  `py.typed` marker.
- A hub catalog curation policy,
  [`spikeforge_hub/CURATION.md`](spikeforge_hub/CURATION.md): remote
  entries must name a real repository/reference plus a verified license (and a
  checksum where available), and unverified candidates must be marked
  `unverified-candidate` and reported `available: false`.
- A versioned protocol contract under [`protocol/`](protocol/README.md): the
  JSON Schema envelope and the pydantic models carry `protocol_version`,
  currently `"1.0"`, with a TypeScript codegen/validation step
  (`client/src/protocol/generated.ts`) and
  [`tests/test_protocol_schema_parity.py`](tests/test_protocol_schema_parity.py)
  keeping both sides in agreement.
- A `packages/` workspace defining two PEP 621 distributions **without moving a
  source file**: `spikeforge` `0.3.0` (core, import root `spikeforge`)
  and `spikeforge-server` `0.1.0` (import root `server`, which depends on
  core). `packages/*/pyproject.toml` is now the packaging authority.
- [`compatibility.json`](compatibility.json) at the repository root, recording
  the `protocol_version` (`"1.0"`), the released distribution versions, and the
  pinned dashboard bundle version (`dashboard: "0.1.0"`, now built in
  `capsize-games/spikeforge-dashboard`).
- The dashboard extraction (ARCH-0001 Phase 2): the browser UI now lives in its
  own repository,
  [`capsize-games/spikeforge-dashboard`](https://github.com/capsize-games/spikeforge-dashboard), created
  from this repository's history with `git subtree split --prefix=client`.
  `client/` remains here for one release as a read-only mirror.
- `SPIKEFORGE_DASHBOARD_DIST` (see [`spikeforge/config.py`](spikeforge/config.py))
  lets the server serve a pinned prebuilt dashboard bundle instead of the
  in-repo `client/dist`; [`server/web.py`](server/web.py) prefers it when
  present and otherwise keeps today's resolution, covered by
  [`tests/test_dashboard_bundle.py`](tests/test_dashboard_bundle.py).

- The `spikeforge-targets` distribution (ARCH-0001 Phase 3): the deploy layer —
  `targets/` (plus `backends/`), `energy/`, `event_runtime/`, and the
  `target_cli` entry point — moved out of core into the top-level `spikeforge_targets`
  import root, packaged as `packages/spikeforge-targets` (`spikeforge-targets` `0.1.0`,
  depending on `spikeforge~=0.3.0`). The `norse` and `lava` extras moved
  with the code they gate, and the server pins `spikeforge-targets~=0.1.0`. The
  old `spikeforge.{targets,energy,event_runtime}` import paths were deleted
  outright — the project is pre-1.0 and unpublished, so no back-compat shim was
  added.
- The standalone [`capsize-games/spikeforge-targets`](https://github.com/capsize-games/spikeforge-targets)
  satellite repository (ARCH-0001 Phase 3): created (private) and populated from
  this repository's history with `git subtree split --prefix=spikeforge_targets`, it
  lays out the `spikeforge_targets/` package at its root alongside `pyproject.toml`
  (`spikeforge-targets` `0.1.0`, core pin `spikeforge~=0.3.0`), `README.md`,
  `LICENSE`, `.gitignore`, a Python 3.10–3.13 CI workflow, and the 19 test
  modules that exercise `spikeforge_targets`. Core has not been pushed and no
  distribution is on PyPI yet, so the satellite CI installs core from its git
  remote as a temporary stopgap until `spikeforge` `0.3.0` is published.
- The `spikeforge-hub` distribution (ARCH-0001 Phase 4): the model hub — the curated
  catalog, cache, isolated download worker, and the inspect → compat → promote
  import funnel — moved out of core into the top-level `spikeforge_hub` import root,
  packaged as `packages/spikeforge-hub` (`spikeforge-hub` `0.1.0`, depending on
  `spikeforge~=0.3.0`). `huggingface_hub` is now a base dependency of this
  distribution rather than a core `hub` extra, and the server pins
  `spikeforge-hub~=0.1.0`. The old core-relative hub import path was deleted
  outright rather than kept as a deprecated re-export shim.
- The standalone [`capsize-games/spikeforge-hub`](https://github.com/capsize-games/spikeforge-hub)
  satellite repository (ARCH-0001 Phase 4): created (private) and populated
  from this repository's history with `git subtree split --prefix=spikeforge_hub`, it
  lays out the `spikeforge_hub/` package at its root alongside `pyproject.toml`
  (`spikeforge-hub` `0.1.0`, core pin `spikeforge~=0.3.0`, a `dev` extra),
  `README.md`, `LICENSE`, `.gitignore`, a Python 3.10–3.13 CI workflow, and the
  hub test modules that exercise `spikeforge_hub`. Its CI uses the same PyPI-or-git
  core-install stopgap as `capsize-games/spikeforge-targets` until `spikeforge` `0.3.0`
  is published.
- Removed the stray, unreferenced `packages/spikeforge/` duplicate (263
  tracked files) that shadowed the real `spikeforge/` package; no
  `pyproject.toml`, CI job, test, or build script referenced it (the builds use
  the `packages/spikeforge/spikeforge` symlink to the real package).

### Changed

- **License holder is `Capsize Games`.** The BSD 3-Clause [`LICENSE`](LICENSE)
  and [`AUTHORS`](AUTHORS) name Capsize Games, and every distribution's
  `pyproject.toml` lists `Capsize Games <contact@capsizegames.com>` as author and
  maintainer.
- **Documentation finalized to the delivered state (R4).** The plan documents now
  describe the landed `spikeforge` rename and the `capsize-games` move, the four
  distributions and their import roots, and the shim-free extraction. Stale
  core-relative module paths were rewritten to the real `spikeforge_targets/`
  and `spikeforge_hub/` trees.
- **No back-compat aliases.** The legacy
  `spikeforge.{targets,energy,event_runtime,hub}` import paths were deleted, not
  kept as deprecated re-export shims: the project is pre-1.0 and was never
  published, so no compatibility window is needed.
- Console-script ownership of `spikeforge-energy` and `spikeforge-targets` moved to the
  `spikeforge-targets` distribution and `spikeforge-hub` moved to the `spikeforge-hub` distribution,
  resolving to `spikeforge_targets.energy.cli:main`, `spikeforge_targets.cli.target_cli:main`,
  and `spikeforge_hub.cli:main`; core keeps the five remaining scripts
  (`spikeforge`, `spikeforge-encodings`, `spikeforge-verify`, `spikeforge-records`,
  `spikeforge-benchmark`). The `spikeforge_targets` and `spikeforge_hub` import roots
  are the only paths; the old `spikeforge.{targets,energy,event_runtime,hub}`
  paths were deleted.
- The maintainer contact address is now `contact@capsizegames.com` in
  [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), [`SECURITY.md`](SECURITY.md), and
  the `packages/` packaging metadata; the `<maintainer@example.com>` placeholder
  is gone.
- The catalog schema rejects free-text licenses: an entry's `license` must be a
  concrete SPDX-style id or the explicit `unverified-candidate` marker.
- Core is now version `0.3.0` and no longer packages `server/`: a headless
  `pip install spikeforge` ships only `spikeforge/`, `main.py`, and
  `main_encodings.py`. The core distribution excludes the server root in its
  package discovery, and the server's FastAPI stack is no longer a core extra.
- The core `web` extra is removed; its `fastapi`, `uvicorn[standard]`,
  `websockets`, and `pydantic` dependencies become the base dependencies of
  `spikeforge-server`.
- [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml)
  install the `packages/spikeforge-server` distribution instead of the
  removed `web` extra, keeping the `TORCH_INDEX_URL` build arg and the `:8877`
  port mapping.
- [`requirements.txt`](requirements.txt) no longer lists the FastAPI server
  stack, so a core dev install stays headless.
- **Strict protocol-version enforcement (ARCH-0001 Phase 2).** An inbound
  message that omits `protocol_version` is now rejected with a `type: "error"`
  frame carrying `payload.code = "protocol_version_mismatch"`, closing the
  Phase 1 legacy-acceptance window ([`server/app.py`](server/app.py),
  [`protocol/README.md`](protocol/README.md)), covered by
  [`tests/test_protocol_negotiation.py`](tests/test_protocol_negotiation.py).
- The core CI `client` job is reduced to the protocol-codegen guard
  (`npm ci`; `npm run gen:protocol`; `git diff --exit-code --
  client/src/protocol/generated.ts`); the full dashboard build is now owned by
  `capsize-games/spikeforge-dashboard`.

### Removed

- `setup.py`: retired in favour of the per-distribution
  `packages/*/pyproject.toml` files (the design-doc Phase 1b packaging split).
- Four invented hub catalog entries under a fictional `snn-community/*`
  namespace (`hf/snn-fc-mnist`, `hf/snn-conv-mnist`,
  `hf/snn-recurrent-mnist`, `snntorch/fc_mnist_weights`). They declared
  `"license": "see upstream"` and had no real upstream, so they were removed
  rather than shipped. The Hugging Face ingestion path and the `hub` extra are
  unchanged.

### Phase 4 status

- **`spikeforge-hub` extraction shipped.** ARCH-0001 Phase 4 moved the model hub to the
  `spikeforge_hub` import root and the `packages/spikeforge-hub` distribution, and published
  the private satellite repository
  [`capsize-games/spikeforge-hub`](https://github.com/capsize-games/spikeforge-hub) (see Added/Changed
  above). The hub extraction was executed per the Phase 4 directive; the T3
  catalog-stability clause still reads *unavailable* because no release baseline
  exists yet (`scripts/topology_metrics.py`).
- **`spikeforge-server` extraction: no-go — `capsize-games/spikeforge-server` was not created.** Per
  [`plans/arch-0001-decision-metrics.md`](plans/arch-0001-decision-metrics.md)
  the server is extracted only when trigger **T4** fires, which requires *all*
  of: (a) server-only release demand ≥ 3 in the trailing 90 days, (b) ≥ 2 core
  pin conflicts in the window, and (c) isolated `server/` change sets ≥ 20% of
  core commits over a quarter. On 2026-09-11 only clause (b) held;
  `scripts/topology_metrics.py` reported verbatim:

  ```text
  T4 spikeforge-server (Phase 4 conditional): not fired
      [         no] server-only release demand >= 3
      [        yes] core pin conflicts >= 2 in window
      [         no] isolated server change sets >= 20% of core commits
  ```

  With T4 not fired, the no-go stands and the server stays in the monorepo as
  the `spikeforge-server` distribution. The trigger is re-measured each
  release cycle.

## [0.2.0] - 2026-09-10

The professionalization program (workstreams WS-A…WS-F) plus the first
open-source-readiness pass. Every capability below is additive; the
`TopologySpec` single source of truth, the legacy `_fc1/_lif1/_fc2/_lif2`
checkpoint keys, and the existing WebSocket payload keys are preserved.

### Added

- **Model hub (WS-A).** A bundled, offline-first curated catalog
  (`spikeforge_hub/models.json`, 10 verified entries across five
  frameworks),
  optional live Hugging Face access behind the `hub` extra, an isolated
  download worker with progress/cancel and checksum verification, and an
  inspect → compat → promote import funnel. Surfaces: the `spikeforge-hub` CLI, six
  additive WebSocket actions, and the `HubPanel` client browser.
- **Backend execution (WS-B).** A substitution executor that applies a
  target's declared rewrites with a rewrite report and a drift check, and
  executable `reference`, `norse`, and `lava_loihi2` backends behind one
  `compile_run` entry point. Surfaces: `deploy`/`rewrite`/`run` and the
  `BackendRunPanel`.
- **Sequence primitives (WS-C).** Per-stage heterogeneous neurons, ten new
  stage kinds with explicit NIR contracts (or an explicit unexportable
  outcome), a sequence/token data path, and the `sequence_mlp`/`sequence_attn`
  presets.
- **Event runtime and energy (WS-D).** A sparse/event-driven runner with a
  dense-parity acceptance test, SOP/MAC/AC counting, and an `spikeforge-energy` report
  that maps op counts to a declared, per-target cost table (all tables are
  `"measured": false` with a source).
- **Operational maturity (WS-E).** Opt-in persisted metrics under
  `SPIKEFORGE_METRICS_DIR`, optional TensorBoard/W&B tracking sinks behind extras,
  determinism tooling, and a MkDocs Material docs site generated from
  `plans/` by `scripts/build_docs.sh`.
- **Interop fold-ins (WS-F).** Event-dataset training, an ONNX export/import
  bridge, `nirtorch` extraction of third-party PyTorch modules, weight-level
  quantization, non-square sensor geometry, and per-step hidden-layer
  animation.
- `CHANGELOG.md`, the community files, and `.github/` issue/PR templates.
- A PEP 561 `py.typed` marker, shipped as package data.

### Changed

- Bumped the version from `0.1.0` to `0.2.0`.
- `Development Status` classifier from `3 - Alpha` to `4 - Beta`: the planned
  capabilities are delivered, but hardware and energy results remain
  unmeasured and are reported as estimates.
- `python_requires` raised from `>=3.8` to `>=3.10`, with classifiers for
  Python 3.10–3.13 to match what is tested.
- Reconciled `setup.py` dependency floors with `requirements.txt`
  (`torch>=2.5`, `torchvision>=0.20`, `snntorch>=1.0`, `matplotlib>=3.8`,
  `Pillow>=10.0`, `numpy>=1.26`) and declared `psutil`.
- CI now runs the docs check, an optional-extras matrix, a blocked-optional-
  dependency degradation run, a Python 3.10–3.13 matrix, and the client build
  on every push and pull request.

### Fixed

- `python -m spikeforge.cli.records_cli` and
  `python -m spikeforge.cli.target_cli` now execute under module
  invocation, matching the other console-script modules.
- Splitting `server/messages.py` keeps it within the project's 250-line style
  contract; the hub message helpers moved to `server/hub_messages.py`.

## [0.1.0] - 2026-09-09

### Added

- The initial spike-encoding playground: MNIST-subset loading, rate/latency/
  delta/random spike encoders, matplotlib/MP4 exports, and a fully-connected
  LIF `SpikingNet` trained with a surrogate-gradient loss.
- The FastAPI + React dashboard: a dark-themed WebSocket UI with a training
  panel, live charts, model management, and a CPU/GPU device picker with a
  live resource monitor.
- Interpreter spine (Phase 1): `TopologySpec` presets, a neuron registry, NIR
  export, an independent NIR interpreter, and numerical drift validation.
- Dual-mode introspection (Phase 2) and the unified dashboard (Phase 3).
- Event datasets (Phase 4) through Tonic, behind the `events` extra.
- Deployment targets and interop (Phase 5): a capability matrix, per-target
  reports, external NIR import/export, and a round-trip fidelity guarantee.
- Production workflows (Phase 6): a reproducibility manifest and config hash,
  a searchable checkpoint registry, opt-in scale-ups, a stored benchmark
  suite, JSON logging, packaged console scripts, and Docker CPU/GPU profiles.

[Unreleased]: https://github.com/capsize-games/spikeforge/compare/spikeforge-v0.3.0...HEAD
[spikeforge-targets-v0.1.1]: https://github.com/capsize-games/spikeforge/compare/spikeforge-v0.3.0...spikeforge-targets-v0.1.1
[0.3.0]: https://github.com/capsize-games/spikeforge/compare/v0.2.0...spikeforge-v0.3.0
[0.2.0]: https://github.com/capsize-games/spikeforge/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/capsize-games/spikeforge/releases/tag/v0.1.0

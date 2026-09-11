# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project has **not** been published to any package index; the versions below
describe the local source tree.

## [Unreleased]

### Added

- Open-source readiness scaffolding: [`CONTRIBUTING.md`](CONTRIBUTING.md),
  [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), [`SECURITY.md`](SECURITY.md),
  [`NOTICE.md`](NOTICE.md), issue and pull-request templates, and a
  `py.typed` marker.
- A hub catalog curation policy,
  [`snn_hub/CURATION.md`](snn_hub/CURATION.md): remote
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
  source file**: `snn-interpreter` `0.3.0` (core, import root `snn_interpreter`)
  and `snn-interpreter-server` `0.1.0` (import root `server`, which depends on
  core). `packages/*/pyproject.toml` is now the packaging authority.
- [`compatibility.json`](compatibility.json) at the repository root, recording
  the `protocol_version` (`"1.0"`), the released distribution versions, and the
  pinned dashboard bundle version (`dashboard: "0.1.0"`, now built in
  `w4ffl35/snn-dashboard`).
- The dashboard extraction (ARCH-0001 Phase 2): the browser UI now lives in its
  own repository,
  [`w4ffl35/snn-dashboard`](https://github.com/w4ffl35/snn-dashboard), created
  from this repository's history with `git subtree split --prefix=client`.
  `client/` remains here for one release as a read-only mirror.
- `SNN_DASHBOARD_DIST` (see [`snn_interpreter/config.py`](snn_interpreter/config.py))
  lets the server serve a pinned prebuilt dashboard bundle instead of the
  in-repo `client/dist`; [`server/web.py`](server/web.py) prefers it when
  present and otherwise keeps today's resolution, covered by
  [`tests/test_dashboard_bundle.py`](tests/test_dashboard_bundle.py).

- The `snn-targets` distribution (ARCH-0001 Phase 3): the deploy layer —
  `targets/` (plus `backends/`), `energy/`, `event_runtime/`, and the
  `target_cli` entry point — moved out of core into the top-level `snn_targets`
  import root, packaged as `packages/snn-targets` (`snn-targets` `0.1.0`,
  depending on `snn-interpreter~=0.3.0`). The `norse` and `lava` extras moved
  with the code they gate, and the server pins `snn-targets~=0.1.0`. The
  `snn_interpreter.{targets,energy,event_runtime}` import paths remain as
  deprecated re-export shims that emit a `DeprecationWarning` and raise a clear
  `ImportError` when `snn-targets` is not installed.
- The standalone [`w4ffl35/snn-targets`](https://github.com/w4ffl35/snn-targets)
  satellite repository (ARCH-0001 Phase 3): created (private) and populated from
  this repository's history with `git subtree split --prefix=snn_targets`, it
  lays out the `snn_targets/` package at its root alongside `pyproject.toml`
  (`snn-targets` `0.1.0`, core pin `snn-interpreter~=0.3.0`), `README.md`,
  `LICENSE`, `.gitignore`, a Python 3.10–3.13 CI workflow, and the 19 test
  modules that exercise `snn_targets`. Core has not been pushed and no
  distribution is on PyPI yet, so the satellite CI installs core from its git
  remote as a temporary stopgap until `snn-interpreter` `0.3.0` is published.
- The `snn-hub` distribution (ARCH-0001 Phase 4): the model hub — the curated
  catalog, cache, isolated download worker, and the inspect → compat → promote
  import funnel — moved out of core into the top-level `snn_hub` import root,
  packaged as `packages/snn-hub` (`snn-hub` `0.1.0`, depending on
  `snn-interpreter~=0.3.0`). `huggingface_hub` is now a base dependency of this
  distribution rather than a core `hub` extra, and the server pins
  `snn-hub~=0.1.0`. The `snn_interpreter.hub` import path remains a deprecated
  re-export shim that emits a `DeprecationWarning` and raises a clear
  `ImportError` when `snn-hub` is not installed.
- The standalone [`w4ffl35/snn-hub`](https://github.com/w4ffl35/snn-hub)
  satellite repository (ARCH-0001 Phase 4): created (private) and populated
  from this repository's history with `git subtree split --prefix=snn_hub`, it
  lays out the `snn_hub/` package at its root alongside `pyproject.toml`
  (`snn-hub` `0.1.0`, core pin `snn-interpreter~=0.3.0`, a `dev` extra),
  `README.md`, `LICENSE`, `.gitignore`, a Python 3.10–3.13 CI workflow, and the
  hub test modules that exercise `snn_hub`. Its CI uses the same PyPI-or-git
  core-install stopgap as `w4ffl35/snn-targets` until `snn-interpreter` `0.3.0`
  is published.
- Removed the stray, unreferenced `packages/snn_interpreter/` duplicate (263
  tracked files) that shadowed the real `snn_interpreter/` package; no
  `pyproject.toml`, CI job, test, or build script referenced it (the builds use
  the `packages/snn-interpreter/snn_interpreter` symlink to the real package).

### Changed

- Console-script ownership of `snn-energy` and `snn-targets` moved to the
  `snn-targets` distribution and `snn-hub` moved to the `snn-hub` distribution,
  resolving to `snn_targets.energy.cli:main`, `snn_targets.cli.target_cli:main`,
  and `snn_hub.cli:main`; core keeps the five remaining scripts
  (`snn-interpreter`, `snn-interpreter-encodings`, `snn-verify`, `snn-records`,
  `snn-benchmark`). The `snn_targets` and `snn_hub` import roots, not
  `snn_interpreter.{targets,hub}`, are now the canonical paths; the old paths
  are deprecated shims.
- The maintainer contact address is now `contact@capsizegames.com` in
  [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), [`SECURITY.md`](SECURITY.md), and
  the `packages/` packaging metadata; the `<maintainer@example.com>` placeholder
  is gone.
- The catalog schema rejects free-text licenses: an entry's `license` must be a
  concrete SPDX-style id or the explicit `unverified-candidate` marker.
- Core is now version `0.3.0` and no longer packages `server/`: a headless
  `pip install snn-interpreter` ships only `snn_interpreter/`, `main.py`, and
  `main_encodings.py`. The core distribution excludes the server root in its
  package discovery, and the server's FastAPI stack is no longer a core extra.
- The core `web` extra is removed; its `fastapi`, `uvicorn[standard]`,
  `websockets`, and `pydantic` dependencies become the base dependencies of
  `snn-interpreter-server`.
- [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml)
  install the `packages/snn-interpreter-server` distribution instead of the
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
  `w4ffl35/snn-dashboard`.

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

- **`snn-hub` extraction shipped.** ARCH-0001 Phase 4 moved the model hub to the
  `snn_hub` import root and the `packages/snn-hub` distribution, and published
  the private satellite repository
  [`w4ffl35/snn-hub`](https://github.com/w4ffl35/snn-hub) (see Added/Changed
  above). The hub extraction was executed per the Phase 4 directive; the T3
  catalog-stability clause still reads *unavailable* because no release baseline
  exists yet (`scripts/topology_metrics.py`).
- **`snn-server` extraction: no-go — `w4ffl35/snn-server` was not created.** Per
  [`plans/arch-0001-decision-metrics.md`](plans/arch-0001-decision-metrics.md)
  the server is extracted only when trigger **T4** fires, which requires *all*
  of: (a) server-only release demand ≥ 3 in the trailing 90 days, (b) ≥ 2 core
  pin conflicts in the window, and (c) isolated `server/` change sets ≥ 20% of
  core commits over a quarter. On 2026-09-11 only clause (b) held;
  `scripts/topology_metrics.py` reported verbatim:

  ```text
  T4 snn-server (Phase 4 conditional): not fired
      [         no] server-only release demand >= 3
      [        yes] core pin conflicts >= 2 in window
      [         no] isolated server change sets >= 20% of core commits
  ```

  With T4 not fired, the no-go stands and the server stays in the monorepo as
  the `snn-interpreter-server` distribution. The trigger is re-measured each
  release cycle.

## [0.2.0] - 2026-09-10

The professionalization program (workstreams WS-A…WS-F) plus the first
open-source-readiness pass. Every capability below is additive; the
`TopologySpec` single source of truth, the legacy `_fc1/_lif1/_fc2/_lif2`
checkpoint keys, and the existing WebSocket payload keys are preserved.

### Added

- **Model hub (WS-A).** A bundled, offline-first curated catalog
  (`snn_interpreter/hub/models.json`, 10 verified entries across five
  frameworks),
  optional live Hugging Face access behind the `hub` extra, an isolated
  download worker with progress/cancel and checksum verification, and an
  inspect → compat → promote import funnel. Surfaces: the `snn-hub` CLI, six
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
  dense-parity acceptance test, SOP/MAC/AC counting, and an `snn-energy` report
  that maps op counts to a declared, per-target cost table (all tables are
  `"measured": false` with a source).
- **Operational maturity (WS-E).** Opt-in persisted metrics under
  `SNN_METRICS_DIR`, optional TensorBoard/W&B tracking sinks behind extras,
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

- `python -m snn_interpreter.cli.records_cli` and
  `python -m snn_interpreter.cli.target_cli` now execute under module
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

[Unreleased]: https://github.com/w4ffl35/snn_interpreter/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/w4ffl35/snn_interpreter/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/w4ffl35/snn_interpreter/releases/tag/v0.1.0

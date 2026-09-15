# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Version tags shaped `<distribution>-vX.Y.Z` (e.g. `spikeforge-v0.3.1`) mark
what was actually published to PyPI; the bare `[X.Y.Z]` headers below predate
that and describe the local source tree only.

## [Unreleased]

### Added

- **Heidelberg audio splits are read directly, bypassing tonic's decode.**
  `spikeforge/events/hsd_reader.py` reads the SHD/SSC HDF5 itself and scales
  seconds to microseconds in `float64`, which recovers every timestamp tonic
  loses (see **Fixed**). It replaces exactly one thing — the per-sample decode.
  Tonic still downloads the dataset, extracts it, owns the cache layout, and
  supplies the sensor geometry; the reader is handed the constructed tonic
  dataset and consults it for all of that. The emitted stream keeps tonic's own
  `(t, x, p)` structured layout, so nothing downstream can tell the two paths
  apart.

  Routing is declared on the dataset, not hard-coded in the loader:
  `DatasetSpec.native_reader` names the reader, and only `ssc` sets it — the
  DVS datasets keep tonic's decode, which is correct for them. `h5py` arrives
  with tonic itself and is imported defensively, so this module is to `h5py`
  what `tonic_api` is to `tonic`: the only one that imports it.

  Pinning `numpy<2` would also have worked and was rejected: a published number
  whose reproduce command silently yields single-bin garbage on numpy 2 is not
  reproducible. This fix is correct on any numpy, and
  `tests/test_hsd_reader.py` asserts the broken promotion *is* still broken, so
  the workaround cannot quietly outlive its reason.

  It also opens the HDF5 once rather than per `__getitem__` as tonic does,
  which is what lets the source's open-once caching actually pay off here:
  0.50 ms/sample against 2.29 ms through tonic.

- **Auditory event sensors load.** A cochlea has channels, not pixel rows:
  tonic's SHD and SSC declare `sensor_size = (700, 1, 1)` and their streams
  carry `(t, x, p)` with no `y` field at all, so `events_to_sample` raised
  "event stream has no 'y' field" and no audio dataset could be read. A sensor
  whose declared height is 1 now places every event on row 0. A missing `y` on
  a taller sensor is still an error — there the field is absent *data* rather
  than an axis the recording does not have, and defaulting it would silently
  flatten a 2-D recording onto one row.

- **Impossible timestamps are refused by name.** `EventTimestampError` rejects
  a stream whose timestamps are negative, which recording times cannot be.
  This is what a non-finite float cast to an integer looks like, and it is not
  hypothetical: see **Fixed** below. The check sits before binning, because
  binning is deliberately forgiving of a zero-width time span — a real sample
  can have every event at one instant — and so cannot tell a genuine instant
  from timing that arrived destroyed.

- **An SSC reference configuration**, `ssc-fc-legacy`: `fc_legacy` at
  `input_size=700` (the flat cochlea area), `num_classes` left to the
  registry's 35. Registered so the target and its geometry are recorded, and
  so running it reports the upstream blocker at the point of use. **No
  checkpoint or catalog entry is published**, because no honest number can be
  produced from it yet.

- **The reference-training script can train and score an event dataset.**
  Three things blocked it, all now closed. `_train` hard-coded
  `TrainingEngine`, so it now picks the engine from the registry's `modality`
  and a new event dataset needs no change there. `_full_test_accuracy` went
  through `build_loader`, which refuses an event dataset **by design**, so
  events get their own walk over the test source's full length — every
  sample, not the four batches `EvalMixin` caches for the dashboard, so an
  event row means what the image rows mean. `_shrink_progress_evaluation`
  likewise no longer needs an image loader to size its in-training probe.

- **`epoch_samples` on `EventTrainingEngine`, so an epoch can be a real pass.**
  `event_batches` caps an epoch at `EPOCH_BATCHES` (10) batches, which is
  right for the live dashboard and is a **cap, not a fraction**: even at
  `subset=1` — documented everywhere else as "the whole training split" — an
  event epoch visited only `10 * batch_size` samples however large the split
  was. Every reference entry's notes claim training "on the full <dataset>
  training split", so publishing an event row on the old default would have
  shipped a false claim. Naming the split's own size makes it true, and the
  extent is recorded in the reproducibility manifest because it changes what
  an epoch means. The default is unchanged, so the dashboard is unaffected,
  and no such row was ever published.

- **A DVS128 Gesture reference configuration**, `dvs128-gesture-conv-net`, is
  registered in the script's table: `conv_net` at `in_channels=2`,
  `input_size=128`, batch 16 (one bridged sample is ~3.3 MB, so the image
  rows' batch of 128 would be hundreds of megabytes of input tensor alone).
  `num_classes` is deliberately absent — the engine injects the registry's 11.
  **No checkpoint or catalog entry is published yet:** the dataset could not
  be downloaded (see below), so there is no number, and inventing one is the
  thing this project exists not to do.

- **Every trained hub entry now records what its training data permits, and
  who to credit.** An entry's `license` describes the **weights** — this
  project's own artifact, BSD-3-Clause — and nothing recorded the terms of the
  data they encode. That gap was already live: `reference/kmnist-fc-legacy`
  shipped marked `BSD-3-Clause` while KMNIST is CC BY-SA 4.0 with a specific
  attribution CODH asks for, which appeared nowhere in the artifact.

  `HubEntry` gains `dataset_license` and `dataset_attribution`, both required
  for `source: "reference"` alongside `weights`/`sha256`/`dataset`/
  `test_accuracy`, and `dataset_license` is held to exactly the rule `license`
  is held to — a concrete SPDX-style id or the `unverified-candidate` marker,
  never free text. Values come from one table beside the dataset registry
  ([`spikeforge/data/dataset_provenance.py`](spikeforge/data/dataset_provenance.py))
  so the catalog and the registry cannot disagree; a test fails if a registry
  dataset is missing from it. The public page at
  <https://spikeforge.net/hub/> gains a Training data column, and
  [`NOTICE.md`](NOTICE.md) carries the attributions.

  Whether trained weights are "adapted material" under a ShareAlike licence is
  unsettled — the prevailing ML norm says they are not, and Creative Commons
  state their licences are not designed to govern model weights. This project
  takes no position. It records the provenance and lets a reader judge, which
  is the same move it makes everywhere else.

  Licences were read from each publisher's own page: Fashion-MNIST MIT,
  KMNIST CC BY-SA 4.0 (CODH's requested wording, verbatim), N-MNIST
  CC BY-SA 4.0, DVS128 Gesture CC BY 4.0, Spiking Speech Commands CC BY 4.0.
  **MNIST and CIFAR10-DVS are recorded as `unverified-candidate`**: their
  primary sources were unreachable, and the widely cited licences for both are
  secondary, so they are not asserted. Four shipped MNIST checkpoints
  therefore disclose an unverified data licence rather than a guessed one.
  Unlike `license`, `dataset_license` does not gate availability: what the
  data permits is disclosure for a reader, not a claim about whether the
  weights load.

- **`--sync-provenance`** on `scripts/train_reference_models.py` refreshes
  those two fields from the provenance table without retraining. A licence
  gets verified, or changes upstream, long after the bytes were published, and
  republishing a checkpoint to correct a citation would replace the artifact
  its numbers were measured on. The four fields that describe the bytes are
  never touched by it.

### Changed

- **`cifar10_dvs` can no longer be trained through `EventTrainingEngine`.**
  It ships upstream as one undivided pool, so it now declares only a train
  split, and asking it for held-out data raises the typed
  `EventSplitMissingError` naming the dataset and the split. Because
  evaluation runs from the first training step, refusing to score it also
  refuses to train it — the engine fails at construction rather than part-way
  through. That is deliberate: the alternative is a partition this project
  invented, which no published number elsewhere would be comparable to. One
  line in the registry reverses it if that trade stops being the right one.
  The other three event datasets are unaffected.

### Fixed

- **Event training never shuffled, so a class-ordered dataset trained one class
  per batch.** The image path builds its loader with `shuffle=train`; the event
  path read `range(start, stop)` strictly in order and had no shuffle anywhere.
  Real event datasets ship grouped by class — SSC's training split is exactly
  **35 contiguous runs over 75,466 samples** — so every batch contained a
  single class in class order, and the network learned only to name whichever
  class it was currently being shown. Measured on SSC: **3.84%** unshuffled
  against a 2.86% chance baseline on 35 classes, rising to **11.88%** once the
  training split is visited in a seeded random order.

  The synthetic fixture hid this completely. Its labels are
  `index % num_classes`, so consecutive indices cycle through every class —
  accidentally perfect interleaving, which is why no existing test caught it
  and why it only surfaced against a real recording.

  `event_batches` now takes `shuffle` and `seed`; the engine passes
  `shuffle=train`, mirroring the image loader exactly, and the order is seeded
  so a published row stays reproducible. The default is sequential, so the
  dashboard and every existing caller are unchanged.

- **Tonic destroys every SHD and SSC timestamp, and the pipeline would have
  trained on it silently.** The Heidelberg files store `spikes/times` as
  `float16`, whose maximum is 65504. Tonic's reader converts seconds to
  microseconds with `times * 1e6`; under NumPy 2's NEP 50 promotion that stays
  in `float16`, where `1e6` itself does not fit — so the scale factor becomes
  `inf`, every product is `inf` (or `NaN` where the timestamp is 0), and the
  cast to `int64` yields `INT64_MIN` for **every timestamp in every sample**,
  in both SHD and SSC, on tonic 1.4.3. Our binning then saw a zero-width span
  and collapsed all events into the first time step: a spiking network trained
  on that has had all of its timing removed, and would still report a
  plausible-looking accuracy. This is upstream's bug, not ours, but publishing
  a number through it would have been ours. The new `EventTimestampError`
  refuses the stream instead; the audio row stays unpublished until upstream
  is fixed.

- **The loader's named field error never fired for real tonic data.**
  `_field` caught `KeyError`/`IndexError`/`TypeError`, which is what a mapping
  raises for a missing key — but a numpy structured array raises `ValueError`,
  so every actual tonic stream reported numpy's wording instead of this
  project's message naming the absent field. Both container shapes now get the
  named error. The existing test passed because it used a dict.

- **Reading an event dataset re-opened it once per sample.**
  `EventSampleSource.load` called `load_event_pair`, which constructed the
  tonic dataset on every call — and constructing one indexes the split's
  files. A 1000-sample epoch therefore indexed the split 1000 times, which
  would have dominated any event training run and any timing measured from
  one. The source now opens its split at most once and holds it; opening
  stays lazy, so constructing a source still touches neither disk nor
  network, and the download still happens in the isolated child.
  `open_event_dataset` and `sample_from` are the two halves, split so a
  caller reading many samples pays the indexing cost once. `load_event_pair`
  keeps its behaviour for one-off reads and its docstring now says which is
  which.

- **The catalog's license validation let one-word free text through.**
  `CURATION.md` has always said that `"unknown"` and `"TBD"` are rejected;
  they were not. The shape rule refused `"see upstream"` only because it
  contains a space, so any single id-shaped token passed — `unknown`, `TBD`,
  `NOASSERTION`, `none`, `proprietary` all validated as concrete licences,
  which is the exact escape the rule exists to close. Those words are now
  refused by name, case-insensitively and whole-string, so real ids that
  merely resemble one (`Unlicense`) are unaffected. No shipped entry used one;
  the documented guarantee simply was not true.

- **Event-dataset "test accuracy" was training accuracy wearing a test
  label.** The event training path had no train/test split anywhere in it.
  `EventTrainingEngine._epoch_batches` accepted a `train` flag and never read
  it, so `_load_test_batches` — whose docstring promised "held-out scoring" —
  returned the first batches of the *training* stream, and the registry
  hardcoded `{"train": True}` for `n_mnist` and `dvs128_gesture`,
  `{"split": "train"}` for `ssc`, and no split at all for `cifar10_dvs`. Every
  accuracy the event path could report was measured on data it had just
  trained on.

  A split is now threaded through all five layers. `DatasetSpec` carries
  explicit per-split constructor arguments rather than a boolean, because
  tonic disagrees per dataset — `NMNIST` and `DVSGesture` take
  `train=True/False` while `SSC` takes `split="train"/"test"`. The engine
  holds one `EventSampleSource` per split, mirroring the image path's two
  loaders, so `event_batches` needs no split argument and the split is settled
  by the object that opened the dataset. The download worker warms every
  declared split, so the held-out fetch happens in the cancellable child
  rather than inside the training process.

  A dataset that declares no test split now raises the new typed
  `EventSplitMissingError` naming the dataset and the split, rather than
  quietly returning training data; see **Changed** for what that means for
  `cifar10_dvs`. The synthetic offline backend gained a genuinely disjoint
  held-out pool — its own index window *and* its own column band, because the
  generator ignores its seed when jitter is off, so index 64 would otherwise
  have produced a sample pixel-identical to training index 16 under a mere
  relabelling. The training stream's own samples are left byte-identical, and
  the stream still refuses to call itself a recording.

  **Blast radius: no published number is affected.** `tonic` is absent from
  both `requirements.txt` and the `Dockerfile`, so on the deployed dashboard
  the event path raised `EventsExtraMissingError` rather than reporting a
  wrong accuracy, and the public demo is read-only besides. No model hub
  entry, benchmark row, or README figure was ever produced from an event
  dataset. Affected: anyone who installed the `events` extra locally and
  trained on an event dataset, whose reported held-out score was
  meaningless. `tests/test_event_split.py` now fails if the split is removed:
  it asserts the two splits return *different tensors*, not merely that a
  flag is accepted.

## [spikeforge-v0.3.5] - 2026-09-14

Released together as one combination, recorded in
[`compatibility.json`](compatibility.json): `spikeforge` 0.3.5,
`spikeforge-hub` 0.2.0, `spikeforge-targets` 0.1.2, `spikeforge-server` 0.3.1,
`spikeforge-serve` 0.2.1, `spikeforge-io` 0.1.1, `spikeforge-clients` 0.1.1.

Every distribution changed in this release, so every one is bumped.
`spikeforge-hub` takes a minor bump rather than a patch because its wheel now
carries trained weights and a new catalog source; `spikeforge` takes one
because `TrainingEngine` joins the public API. The remaining five gained
`__version__` and a `py.typed` marker.

### Added

- **Trained weights in the model hub.** A new catalog source,
  `"source": "reference"`, carries checkpoints this project trained itself:
  they ship inside the `spikeforge-hub` wheel under `weights/`, are loaded
  rather than rebuilt (`spikeforge_hub.inspect.reference_path`), and are
  verified against the checksum the catalog pins. Each entry records the
  dataset it trained on and what it scores on that dataset's complete held-out
  split; validation rejects an entry that claims trained weights without
  naming its file, checksum, dataset, and accuracy. `spikeforge-hub list
  --trained` filters to them, and every entry card gains a `trained` flag.
  `scripts/train_reference_models.py --publish` regenerates the checkpoints and
  their catalog entries together so the pinned checksum, size, and accuracy
  cannot drift from the bytes that shipped.
- **Published benchmark numbers.** `documentation/benchmarks.md` carries one
  row per reference configuration — test accuracy on the *complete* held-out
  split, epochs, time steps, wall-clock, hardware, and the exact command that
  reproduces it — generated by `scripts/train_reference_models.py`. They are
  explicitly reference configurations with stock hyperparameters, not tuned
  attempts at state of the art.
- **`__version__` on all seven distributions**, read from installed metadata
  so it cannot drift from the wheel, and a `--version` flag on every console
  script. The flag prints the core version, every installed satellite, and
  whether the combination is one `compatibility.json` records — which makes
  the compatibility matrix visible at runtime instead of a file you had to
  know to go and read. `compatibility.json` now ships inside the core wheel.
- **An API reference.** `scripts/build_docs.sh` generates
  `docs/api-reference.md` from the shipped docstrings and type hints via
  `mkdocstrings`, so constructor signatures, parameter types, and return types
  are looked up rather than read out of the source. Added to the `docs` extra.
- **`TrainingEngine` is exported from the package root.** The README's
  flagship snippet reached four modules deep into a path that reads like
  internals while `spikeforge.__all__` advertised a different entry point;
  `from spikeforge import TrainingEngine` now works and is what the README,
  the quickstart, the cookbook, and `examples/01` use.
- **Two CI gates for the surfaces nothing was watching**
  (`published-surfaces` job, torch-free): `scripts/check_readme_links.py`
  renders the packaged README through `readme_renderer` — the library PyPI
  itself uses — and fails on any relative `href`/`src` or any repository link
  naming a path that does not exist; `scripts/check_wiki_links.py` builds the
  GitHub wiki and fails on any dead link. The landing page's i18n suite now
  runs in CI too.
- **`py.typed` on the remaining five distributions** (`spikeforge-hub`,
  `spikeforge-targets`, `spikeforge-serve`, `spikeforge-clients`,
  `spikeforge-server`), declared in their package data, plus the
  `Typing :: Typed` classifier on all seven — **and a mypy gate** in the
  `lint` job so that marker is a promise something checks. `mypy.ini`
  documents the gate as a deliberate ratchet: green today over 398 files, with
  `attr-defined` disabled (the mixin composition defeats it, 59 false
  positives) and 26 modules carrying the remaining 28 "Optional not narrowed"
  errors listed individually so the list can be shrunk one line at a time.
- **A Model hub submission issue template** (`.github/ISSUE_TEMPLATE/`),
  asking for exactly what `CURATION.md` requires: a resolvable locator, a
  concrete SPDX license, a checksum where one is published, and a confirmation
  that the submitter read the upstream license themselves.
- **`plans/ecosystem_listings.md`** — the two directories this field searches
  (NIR's framework support table, Open Neuromorphic's software guide), what
  each currently lists, and drafted submissions for both.

### Fixed

- **The PyPI project pages rendered 30 dead links and a broken hero image, on
  all seven distributions.** Every distribution's `long_description` is the
  repository-root README by symlink, and PyPI does not resolve relative links.
  The README's links are now absolute; verified by rendering it through
  `readme_renderer`, which now reports zero relative references.
- **The published documentation site had 896 dead links across 50 of its 61
  pages.** `docs.spikeforge.net` serves the GitHub wiki, and
  `scripts/build_wiki.py` rewrote only `.md` links that landed on another wiki
  page — every source reference (`spikeforge/topology/spec.py:16`), directory,
  image, and non-page document shipped verbatim, and GitHub silently serves
  the wiki Home page for an unknown wiki path rather than a 404. The generator
  now rewrites every non-page relative link to an absolute repository URL,
  turning a `file.py:16` reference into a working `blob/main/file.py#L16` deep
  link. Twenty-four genuinely stale references in `plans/` were fixed at the
  source: `setup.py` (retired by the packaging migration) and two unbuilt
  client components are now unlinked prose, and `target_cli.py` was repointed
  to its current home in `spikeforge_targets`.
- **`scripts/check_docs_links.py` was gating an artifact nobody publishes.**
  It validates the MkDocs tree in `docs/`, which is built, strict-validated,
  and published nowhere; the site readers actually land on had no gate at all.
  Both are now checked, and each checker's docstring says which is which.
- **The core distribution's PyPI metadata still described an MNIST
  experiment.** The `description` shown in PyPI search results and by
  `pip show` now describes the shipped toolkit, `keywords` drops `mnist` for
  the neuromorphic set the GitHub topics already carried, and all seven
  distributions gained `Documentation`, `Changelog`, and `Issues` URLs with
  `Homepage` pointing at spikeforge.net.
- **The landing page told visitors to `git clone` a package it advertised as
  released on PyPI.** The hero now leads with `pip install spikeforge` and the
  same five-line snippet the README leads with, with the clone path demoted to
  a footnote; the copy is translated into all 17 locales rather than falling
  back to English.
- **The hub page shipped but nothing linked to it.** `/hub/` is now in the
  landing page's navigation and footer, and `sitemap.xml` lists four URLs
  instead of one.
- **`CITATION.cff` emitted a citation with no version.** It now carries
  `version` and `date-released`.
- **The CPU-only install path was documented only for Docker.** The README and
  quickstart now give it for `pip` too: measured, a default install is 5.5 GB
  against 1.1 GB for the CPU wheels, and the order matters —
  `--extra-index-url` on a single command still lets pip prefer the CUDA build.

## [spikeforge-v0.3.4] - 2026-09-14

### Fixed

- **The `spikeforge` console script crashed on a clean, minimal install**
  (`ModuleNotFoundError: No module named 'pandas'`) because
  `spikeforge.exporters.presentation_exporter` and seven sibling exporters
  imported `snntorch.spikeplot` (which itself imports `pandas`, undeclared by
  either snnTorch or spikeforge) at module level. Those imports are now lazy,
  scoped to the function that needs them.
- **`spikeforge` and `spikeforge-encodings` ignored `--help`** and instead
  silently downloaded MNIST and started training. Both now have a real
  `argparse` parser; `--help` prints help and exits without touching the
  network.
- **`spikeforge-verify` and `spikeforge-benchmark` also crashed on a plain
  `pip install spikeforge`** (no extras): both unconditionally imported
  `spikeforge_targets` at module load, even though it is not a core
  dependency. `spikeforge-verify` now degrades honestly (the deployment
  subcommands are simply absent, named in `--help`'s description) and the
  benchmark's energy accounting imports `spikeforge_targets` lazily, only
  when `--energy` is actually requested, via the new
  `spikeforge.targets_extra` module.
- Added `scripts/check_console_scripts.py` and wired it into the CI
  `headless` job so every declared console script is smoke-tested with
  `--help` against a minimal (non-`[all]`, non-`[dev]`) install going
  forward.
- Fixed a broken anchor link (`documentation/dashboard.md` →
  `usage.md#access-control--rate-limiting`, which mkdocs renders as
  `#access-control-rate-limiting`) that made `scripts/build_docs.sh --check`
  fail under strict mode.
- `OPEN_SOURCE_CHECKLIST.md` stated the repository was unpublished, local
  only, with no PyPI release; all of that is now false. The banner, the
  publish-decision item, the dashboard-screenshot item, and several stale
  packaging facts (`setup.py` references, version, test counts) are
  corrected.
- `documentation/dashboard.md`'s screenshot section claimed no image was
  committed; `images/dashboard.png` has existed for a while and is the
  README hero image. The section now reflects that one of five planned
  captures is done.

### Changed

- **Breaking:** renamed the `SSNTrainer` class (a typo) to `SNNTrainer`
  across the public API (`spikeforge.SNNTrainer`), `LatencyTrainer`'s base
  class, and `SNNTrainerLogger`'s base class. No compatibility alias, per
  this project's no-back-compat-shim policy.
- README: added a Docker-free "five lines of Python" quickstart ahead of the
  Docker path, a positioning section comparing spikeforge to snnTorch/Norse/
  Lava/SpikingJelly, a capability matrix (shipped/experimental/spec-only per
  headline feature), and the compatibility-matrix/version-skew explanation
  in the Packages table (now listing all seven distributions, not four).
  `documentation/quickstart.md` mirrors the pip-first path.
- `spikeforge_hub/CURATION.md` now documents that every shipped entry is
  `"source": "bundled"` (a NIR graph rendered fresh from a topology preset,
  never trained weights) and that a trained-checkpoint artifact kind is a
  deliberate follow-up, not something to fake.
- Added `scripts/build_hub_page.py`, a stdlib-only static-site generator for
  the model-hub catalog, deployed alongside the landing page
  (`spikeforge.net/hub/`) so it is browsable and indexable without
  installing anything. `CONTRIBUTING.md` documents the PR-based path for
  proposing a new catalog entry.

## [spikeforge-v0.3.3] - 2026-09-13

### Added

- SNN-native `AffectiveRegulator` with slow-decay broadcast state and spike
  output for downstream module modulation.

## [spikeforge-v0.3.2] - 2026-09-13

### Added

- **Few-shot teaching for the held-out-digit memory protocol.**
  `OneShotAssociativeMemory.teach_many()` averages several hidden-layer
  spike-count traces into one Hebbian write, and the evaluation protocol now
  accepts `teach_examples` so representation capacity and exemplar count can
  be measured independently.

## [spikeforge-server-v0.3.0] - 2026-09-12

### Added

- **`SPIKEFORGE_DASHBOARD_TOKEN`: optional bearer/query-param gate on `/ws`
  and `GET /api/bundle/<name>`.** New `server/auth.py`, mirroring
  `spikeforge-serve`'s existing `/metrics` bearer-token pattern. A
  WebSocket connection is rejected before `accept()` (close code 1008)
  when the token is missing or wrong; the bundle route 401s the same way.
  Both accept the token as a query param (`?token=...`) as well as an
  `Authorization: Bearer` header, since a browser can attach neither a
  custom header to a WebSocket handshake nor one to a plain `<a
  download>` link. Unset (the default) leaves both routes exactly as
  open as before — no change for `docker compose up` on localhost.
- **`SPIKEFORGE_DASHBOARD_MAX_CONCURRENT_JOBS`: a server-wide cap on
  concurrent training/pipeline jobs.** New `server/concurrency.py`.
  `TrainingService`/`PipelineService` already refused a second run
  *within one session*; this caps it *across* sessions too (default 2),
  so many WebSocket connections can't each start their own run and hang
  the shared server. A request past the cap gets a clear `server busy`
  `error` message instead of queuing silently.
- **Dashboard: confirm-before-destroy.** A themed `ConfirmDialog`
  now guards every one-click action that discarded state without
  asking — the Pipeline tab's New/Delete, unloading the current model,
  and cancelling a dataset or hub download.

## [spikeforge-serve-v0.2.0] - 2026-09-12

### Added

- **`spikeforge-serve` CLI restructured into subcommands.** `serve` is the
  original HTTP/WebSocket service, now explicit rather than the bare
  `--bundle` flag. New: `run <name-or-path>` (one-shot inference — a JSON
  request on stdin or `--file`, a JSON response on stdout, no server, so a
  module behaves identically served or run standalone); `install
  <bundle.spkf> --name <name>` (registers a bundle under
  `~/.local/share/spikeforge/modules/<name>/` and writes an executable
  `<name>` wrapper to `~/.local/bin/`, so a trained model becomes its own
  command); `uninstall`/`list`. Installed modules chain over Unix pipes —
  `digit-classifier | jq ... | risk-scorer` — since each reads one JSON
  object on stdin and writes one on stdout. New `spikeforge_serve.modules`
  and `module_runner` modules.
- **Pipeline execution: chain checkpoints into a DAG and run it as one
  program.** New `spikeforge_serve.pipeline` (the DAG data model —
  `PipelineGraph`/`Node`/`Edge`, topological order via Kahn's algorithm,
  raising on a cycle, an unknown node reference, or fan-in),
  `pipeline_runner` (executes a graph through `ServingService`, one node
  at a time, shaping each edge's source output into the target's next
  input via a fixed `extract` mode: `mean_logits`, `predicted_class`, or
  `one_hot`), and `pipeline_store` (save/load/list/delete, mirroring
  `spikeforge.network.model_store`). Deliberately a DAG, not a state
  machine — no cycles, no conditional branching, no fan-in — see
  `documentation/model-deployment.md`.

## [spikeforge-server-v0.2.0] - 2026-09-12

### Added

- **`GET /api/bundle/<name>`.** Downloads a saved checkpoint as a `.spkf`
  deployment bundle, built on demand via `spikeforge.serving.bundle.build`
  into a temp file that's cleaned up after the response. 404s an unknown
  checkpoint name. The dashboard's Model panel gained a matching "Bundle"
  tab.
- **Six new WebSocket actions for the dashboard's Pipeline tab:**
  `list_pipelines`, `save_pipeline`, `load_pipeline`, `delete_pipeline`,
  `run_pipeline`, `stop_pipeline`. A run spawns a background worker
  (`server/pipeline_service.py`, structurally identical to
  `TrainingService`) that streams `pipeline_node_result`/
  `pipeline_run_state` messages without blocking the WebSocket read loop.
  `spikeforge-server` gains `spikeforge-serve` as a dependency to reuse
  `ServingService` directly — a pipeline node's bundle is built in memory
  from its checkpoint at run time, never written to disk.
- New `protocol/payloads/pipeline_graph.schema.json` and matching
  `client_message`/`server_message` schema and Pydantic additions, kept
  in lockstep by `tests/test_protocol_schema_parity.py`.

## [spikeforge-v0.3.1] - 2026-09-12

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
- **`spikeforge.memory.stdp_synapse.STDPSynapse`.** A feedforward
  synapse written by pair-based spike-timing-dependent plasticity:
  exponentially-decaying pre/post eligibility traces update every
  weight from genuine spike order (causal pairs potentiate,
  anti-causal depress, effect decays with `|delta_t|`), unlike
  `HebbianSynapse`'s single-shot rate-coded write. Verified against
  the textbook STDP learning-window shape in
  `tests/test_stdp_synapse.py`; `examples/14_stdp_learning_window.py`
  reproduces the curve.

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
- **Training with a stale checkpoint reference crashed with a raw
  `FileNotFoundError`.** A checkpoint name set by loading a model
  persists in the dashboard's local storage across sessions; if the
  server's model directory was since cleared or rebuilt, a plain
  "Train" click resent that now-missing name and
  `model_store.load()` let the bare OS path leak to the client.
  `load()` now raises a named "checkpoint not found" error, and the
  dashboard drops a checkpoint reference once the server's own model
  list shows it no longer exists.

### Dashboard

- **Prediction moved to the Viewer tab.** "Prediction (displayed
  sample)" lived in Training & Analysis, away from the sample it
  scores; it's now in the Viewer tab's right column.
- **Loaded-model summary moved under the tab strip.** The chip
  naming the active checkpoint only showed in the Viewer tab's side
  column; it's now in the fixed header under the tabs, visible
  regardless of which tab is open.

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

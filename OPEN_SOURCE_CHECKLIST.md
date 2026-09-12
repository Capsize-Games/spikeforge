# Open-source readiness checklist

An actionable pre-release checklist for `spikeforge`, grouped and marked
with what is **already done** (`[x]`) versus **outstanding** (`[ ]`). Every
"done" item was verified in the repository; every "outstanding" item names the
concrete gap. Nothing here changes product code.

Verification snapshot: `version 0.2.0`, `Development Status :: 4 - Beta`,
`python_requires >=3.10` (classifiers for 3.10–3.13), **744 passed / 1 skipped**
with the optional extras, `ruff` clean, docs site builds, BSD-3-Clause.

> **Not published.** This repository is still local. Nothing has been pushed to
> a remote, no release has been created, and no package has been published.

---

## 1. Licensing & authorship

- [x] **`LICENSE` present and coherent.** [`LICENSE`](LICENSE) is the standard
  BSD 3-Clause text; its copyright line points at the AUTHORS file.
- [x] **`AUTHORS` present.** [`AUTHORS`](AUTHORS) lists one contributor
  (`Capsize LLC <contact@capsizegames.com>`), matching the
  `LICENSE` copyright reference. This is the identity recorded in
  `git config user.name` / `user.email`.
- [x] **License declared in packaging metadata.** [`setup.py`](setup.py:34)
  sets `license="BSD-3-Clause"`, `license_files=["LICENSE"]`, and
  `License :: OSI Approved :: BSD License`.
- [x] **The `setup.py` author fields are filled.** `author`, `author_email`,
  and `url` are set from the git identity and the `origin` remote
  (`https://github.com/capsize-games/spikeforge`) — no longer empty.
- [x] **Confirm the intended license.** BSD-3-Clause is the deliberate choice
  for the project itself, and it is now the declared `license` plus
  `license_files`.
- [x] **State the metadata-only weights policy in-repo.** The policy is stated
  in the README, the roadmap (§12 decision 1), and now a top-level
  `NOTICE.md` so a redistributor sees it without reading the README.
- [x] **Third-party license audit for catalog entries — resolved by removal.**
  [`spikeforge_hub/models.json`](spikeforge_hub/models.json) now ships
  **10 entries, all `"license": "BSD-3-Clause"`** (this project's own NIR preset
  graphs). The four former `"license": "see upstream"` entries were invented
  "seed" ids under a fictional `snn-community/*` namespace with no real
  upstream; rather than ship unverifiable ids they were **removed**. The schema
  now rejects free-text licenses (see `spikeforge_hub/CURATION.md`), so an
  unverified candidate must be marked `"unverified-candidate"` and is reported
  `available: false`.

---

## 2. Community files

- [x] `CONTRIBUTING.md` — setup (`scripts/dev.sh setup`), the style contract
  in [`rules.md`](rules.md), and the test/lint gates.
- [x] `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1.
- [x] `SECURITY.md` — a private vulnerability-reporting path and supported
  versions.
- [x] `CHANGELOG.md` — keep-a-changelog style, seeded with the `0.1.0` and
  `0.2.0` work.
- [x] `.github/ISSUE_TEMPLATE/` — bug-report and feature-request templates,
  plus `config.yml` (which routes security reports to `SECURITY.md`).
- [x] `.github/PULL_REQUEST_TEMPLATE.md` — a PR checklist (ruff + pytest +
  client build + docs check + the honesty rule).

---

## 3. CI/CD

- [x] **CI workflow exists.** [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
  runs on push to `main` and on every pull request, with a `concurrency` group
  that cancels superseded runs.
- [x] **Lint step.** `ruff check .` (whole tree, not a hand-maintained list).
- [x] **Test step.** `pytest`, across a Python **3.10 / 3.11 / 3.12 / 3.13**
  matrix consistent with `python_requires`.
- [x] **Client build step.** `npm ci && npm run build` with Node 22 and the
  committed lockfile ([`client/package-lock.json`](client/package-lock.json)).
- [x] **Docs check on PRs.** A `docs` job runs
  `bash scripts/build_docs.sh --check`, so a broken documentation link fails
  the pull request.
- [x] **Optional-extras matrix.** An `extras` job installs each of `events`,
  `onnx`, `hub`, `norse`, `tracking`, and `docs` and exercises it. (`norse` is
  install + import smoke only: the suite deliberately asserts the SDK-gated
  backends are *absent* and drives them through fakes, so a present `norse`
  would flip those absence assertions.)
- [x] **Blocked-deps degradation run.** A `blocked-deps` job installs the
  optional extras, then makes them unimportable via the checked-in
  `scripts/blocked_deps/sitecustomize.py`, proving the honest-degradation paths
  (typed unavailable errors, `available: false`, reported reasons).
- [x] **Version-matrix coverage.** The floor was raised to `>=3.10` and CI now
  tests 3.10–3.13, so the advertised support matches what is run.

---

## 4. Repo hygiene

- [x] **`.gitignore` covers generated output.** [`.gitignore`](.gitignore)
  ignores `build/` (datasets, checkpoints, `docs/`, hub cache, benchmark
  records, media), `docs/`, `__pycache__/`, the pytest/ruff/mypy caches,
  `.coverage`, `client/node_modules/`, `client/dist/`, IDE directories
  (`.idea/`, `.vscode/`), and checkpoint files (`*.pt`, `*.pth`, `*.ckpt`).
- [x] **No committed secrets or credentials.** No `.env`, `.pem`, `.key`, or
  credential-like paths are tracked.
- [x] **No committed large binary artifacts.** The largest tracked files are
  text (`README.md` ≈ 86 KB, `client/package-lock.json` ≈ 62 KB); no model
  weights or datasets are tracked.
- [x] **No leftover TODO/FIXME/XXX/HACK markers** in `spikeforge/`,
  `server/`, `main.py`, `main_encodings.py`, or `tests/`.
- [x] **Files are within the style contract** ([`rules.md`](rules.md): files
  ≤ 250 lines, functions ≤ 20 lines).
- [x] **The one over-limit source file was split.** `server/messages.py` is now
  under the 250-line contract; its hub helpers moved to
  `server/hub_messages.py`, mirroring the existing `server/energy_messages.py`
  pattern (the hub senders are re-imported by `hub_handlers.py` and
  `hub_downloads.py`, so no caller changes).
- [x] **`.dockerignore` excludes the generated `docs/` tree** and the Python
  caches, keeping the image context tight.
- [x] **Module entry points work.** `records_cli.py` and `target_cli.py` gained
  `if __name__ == "__main__":` guards, so `python -m
  spikeforge.cli.records_cli` and `... target_cli` work like the other
  console-script modules.

---

## 5. Packaging & release

- [x] **Extras are declared with version floors.** `dev`, `web`, `nir`,
  `events`, `onnx`, `hub`, `norse`, `lava`, `tracking`, `tracking-wandb`,
  `docs` are all in [`setup.py`](setup.py:46).
- [x] **Console scripts are declared.** Eight entry points
  ([`setup.py`](setup.py:88)).
- [x] **Version bump and maturity label decided.** Bumped `0.1.0` → `0.2.0`
  and `Development Status :: 3 - Alpha` → `4 - Beta`: the planned capabilities
  are delivered, but hardware and energy results remain unmeasured (reported
  as estimates), so Beta is the honest label.
- [x] **`python_requires` reconciled with what is tested.** Raised from `>=3.8`
  to `>=3.10`, with `Programming Language :: Python ::` classifiers for
  3.10–3.13 to match CI.
- [x] **Dependency floors reconciled.** `setup.py` now matches
  `requirements.txt` (`torch>=2.5`, `torchvision>=0.20`, `snntorch>=1.0`,
  `matplotlib>=3.8`, `Pillow>=10.0`, `numpy>=1.26`) and declares `psutil`.
- [x] **`py.typed` added.** The package ships a PEP 561 marker, included in
  `package_data`.
- [x] **Release-notes process defined.** `CHANGELOG.md` follows Keep a
  Changelog; new work goes under `## [Unreleased]`.
- [ ] **Decide whether to publish to PyPI.** Metadata, extras, entry points,
  and the long description are all in place, and the author fields are filled.
  Nothing has been published. Publishing remains a deliberate, separate human
  decision; a release workflow should be added only once that decision is made.

---

## 6. Docs

- [x] **Quickstart exists.** The README install/usage sections and the
  developer script cover setup.
- [x] **Cookbook exists and is linked.** [`COOKBOOK.md`](COOKBOOK.md) has
  runnable recipes for every shipped capability; it is linked from the README,
  the docs landing page ([`plans/index.md`](plans/index.md)), and the MkDocs
  nav.
- [x] **Honest-limits narrative exists.** The README's [Implications and
  boundaries](documentation/implications-and-boundaries.md) section states the *why*
  and *consequence* of each deliberate limit.
- [x] **Architecture diagrams exist.** The workstream plans and the
  professionalization roadmap carry Mermaid diagrams rendered by the docs site,
  and the top-level README now carries its own end-to-end pipeline diagram.
- [x] **A README architecture diagram.** The README's "Architecture" section
  renders a Mermaid diagram of the whole pipeline (datasets/events → encoder →
  `TopologySpec` → snnTorch module + NIR graph → simulator/training →
  validation → hub/backends/energy → dashboard).
- [x] **`examples/` directory with runnable scripts.** [`examples/`](examples/)
  ships ten small, offline-safe, importable scripts — one per core journey —
  plus an index ([`examples/README.md`](examples/README.md)) recording the real
  output each prints. Every script was executed for this checklist; the two
  extra-gated ones degrade with a clear printed message. The index is linked
  from the README, `plans/index.md`, and the MkDocs nav, and is included by the
  docs generator.
- [ ] **Dashboard screenshots/GIF.** Deliberately not captured: the repository
  is prepared in a headless environment with no browser, and no image is
  faked. A clearly-marked placeholder in the README's Dashboard section lists
  exactly what to capture and from where, so **a maintainer with a real
  browser must fill this in** (see §9).
- [x] **Link the checklist from the README.** Kept current as items close.

---

## 7. Roadmap decisions

From [`plans/professional_roadmap.md`](plans/professional_roadmap.md:454),
now recorded as resolved in the roadmap's §12:

- [x] **Docs stack — resolved.** MkDocs Material generated from `plans/` and the
  README ([`mkdocs.yml`](mkdocs.yml), [`scripts/build_docs.sh`](scripts/build_docs.sh)).
- [x] **First backend — resolved.** `norse` shipped first (pure-PyTorch,
  pip-installable), with `lava_loihi2` as the first hardware path gated on its
  SDK.
- [x] **Energy cost tables are declared estimates — resolved.** Every table is
  `"measured": false` with a `source`; the report carries `estimate: true`.
- [x] **Sequence scope is experimentation, not production LLM training —
  resolved** and stated in the README.
- [x] **Metrics persistence root — resolved.** Written under
  `SPIKEFORGE_METRICS_DIR` (default `<DATA_DIR>/metrics`) with opt-in
  `SPIKEFORGE_METRICS_PERSIST`.
- [x] **Hub catalog licensing — resolved: metadata-only, verified only.** No
  weights are redistributed; remote artifacts are fetched on demand. The four
  fabricated `"see upstream"` entries were removed, so the shipped catalog now
  contains only entries with a concrete license (tracked in §1 and `NOTICE.md`).
- [x] **Live Hugging Face scope — resolved: allow-list first.** The
  catalog/allow-list path is the default; live search is opt-in behind the
  `hub` extra and gated honestly when absent.

---

## 8. Out of scope (deliberately deferred)

- [x] **Multi-user sessions and authentication** — explicitly out of scope
  ([`plans/professional_roadmap.md`](plans/professional_roadmap.md:476),
  [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md)).
- [x] **Remote telemetry** — no phone-home, no remote metrics; the metrics
  registry is per-process and local by default.
- [x] **Measured (rather than estimated) energy** — deferred until a physical
  device is present; the single `measurement=` integration point is ready.

---

## 9. Remaining human decisions (not blocking local work)

These need a person, not a script, and are recorded rather than invented.
They are the **only** open items on this checklist:

- **A monitored contact address — resolved.** `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, and `setup.py` now point at the monitored inbox
  `contact@capsizegames.com`; the `<maintainer@example.com>` placeholder is gone.
- **The four `"see upstream"` catalog licenses — resolved by removal** (§1 /
  `NOTICE.md`): the four fabricated `snn-community/*` entries were removed from
  the shipped catalog, which now contains only verified, concretely-licensed
  entries.
- **The PyPI publish decision** (§5) and, if chosen, a release workflow and the
  matching repository remote/release settings.
- **Dashboard screenshots/GIF** (§6): capture the five items listed in the
  README's Dashboard placeholder from a real browser session and link them.

---

## Definition of done for a public release

A release is ready when: the community/CI/packaging items above are closed; the
catalog ships only verified, concretely-licensed entries; the version and
`python_requires` decisions are consistent with `requirements.txt`; the two
formerly-open roadmap decisions are recorded; and the remaining human decisions
in §9 are answered.
The repository must also be deliberately made public and published — neither of
which this checklist performs.

# Spikeforge adoption-evaluation fixes — report

A pass/fail assessment of the work below should be measured against
`EVALUATION_REPORT.md`'s findings. Everything below is a change made in this
working tree (uncommitted — see "State of the working tree" at the end);
each claim is followed by how it was verified so it can be spot-checked
without re-deriving it from scratch.

Section numbers below match `EVALUATION_REPORT.md`'s numbering.

---

## 1. 🔴 Blocker — `spikeforge --help` crash — **Fixed**

Root-caused and fixed all three layers, not just the one command named in the
report:

- **The pandas crash.** Eight exporter modules
  (`spikeforge/exporters/{presentation,delta,raster,latency_raster,video,
  latency_video,random_spike_raster,random_spike_video}_exporter.py`)
  imported `snntorch.spikeplot` at module level, which imports `pandas` —
  undeclared by both snnTorch and spikeforge. Changed to function-local
  imports in every case (the exact fix shape the report asked for).
- **`--help` silently trained instead of printing help.** `main.py` and
  `main_encodings.py` (backing the `spikeforge` and `spikeforge-encodings`
  console scripts) took no arguments and ignored argv entirely. Both now
  have a real `argparse.ArgumentParser`; `--help` exits 0 after printing
  help, touching neither the network nor `SNNTrainer`. The heavy exporter
  imports inside `main()` are also now lazy, so `--help` never pays for them.
- **A second instance of the same root cause the report didn't test for.**
  Reproducing with `pip install "spikeforge[all]"` (as the report did) hid a
  second bug: `spikeforge-verify` and `spikeforge-benchmark` — the two CLIs
  the report called "well-built" — also crash with `ModuleNotFoundError` on
  a *genuinely* minimal `pip install spikeforge` (no `[all]`), because
  `spikeforge/cli/verify.py` and `spikeforge/benchmark/energy.py`
  unconditionally imported `spikeforge_targets` at module level, even though
  it is not a core dependency. Fixed the same way the codebase already
  fixes this class of problem elsewhere (`spikeforge/onnx_bridge/api.py`'s
  `require()` pattern): `spikeforge-verify` now degrades honestly — the
  deployment subcommands are simply absent from `--help` when
  `spikeforge-targets` isn't installed, named in the parser's description —
  and the benchmark's energy accounting imports `spikeforge_targets` lazily,
  only when `--energy` is requested, via a new `spikeforge/targets_extra.py`
  shim (`TargetsExtraMissingError`, mirroring the project's existing
  `OnnxExtraMissingError` / `EventsExtraMissingError` convention).
- **CI gate, exactly as requested.** Added `scripts/check_console_scripts.py`
  (reads `[project.scripts]` from `packages/spikeforge/pyproject.toml`, runs
  every one with `--help`, asserts exit 0 and no hang) and wired it into the
  `headless` job in `.github/workflows/ci.yml`, which already builds a
  no-extras wheel into a clean venv — the exact "minimal, not `[all]`, not
  `[dev]`" scenario the report asked for.

**Verified:**
- Built a genuinely clean venv, installed only `pip install
  ./packages/spikeforge/dist/spikeforge-*.whl` (no extras), ran all five
  core console scripts (`spikeforge`, `spikeforge-encodings`,
  `spikeforge-verify`, `spikeforge-records`, `spikeforge-benchmark`) with
  `--help`: all exit 0, all print help, none touch the network.
  `scripts/check_console_scripts.py` automates exactly this and passes.
- Confirmed no regression: ran `spikeforge` with no args end-to-end (real
  training run) against the cached local MNIST data — it still trains and
  writes all five artifacts (`build/spike_mnist_test.{gif,mp4}`,
  `build/spike_presentation.gif`, `build/spike_raster.png`,
  `build/spike_reconstruction.png`), same behavior as before the fix.
- With `spikeforge-targets`/`spikeforge-hub` also installed,
  `spikeforge-verify --help` shows all 14 subcommands exactly as before —
  no behavior change for the `[all]` install path.
- Full test suite: **1185 passed, 8 skipped** (all 8 are pre-existing,
  expected skips for uninstalled `onnx` extra). `ruff check .` clean.
  `scripts/check_core_boundary.py` clean (242 modules, no forbidden
  imports — confirms the fix didn't create a new core-boundary violation).

## 2. 🟠 Model hub has an excellent funnel and almost no content — **Partially addressed**

Did **not** attempt to populate the catalog with genuinely trained weights.
Investigated first: `spikeforge_hub/inspect.py`'s `_materialize()` rebuilds
every `"bundled"` entry fresh from its topology preset (random init) on every
access — there is currently no code path for a bundled entry to carry stored,
trained weights at all. Adding one is a real feature (new entry kind, a
materialization path that loads a checkpoint instead of always rebuilding,
schema changes in `spikeforge_hub/entry.py`, updated `CURATION.md` policy,
new tests) inside a subsystem that is already deliberately strict and
well-tested (dedicated `test_hub_catalog.py` / `test_hub_cache.py` /
`test_hub_verify.py` / `test_hub_inspect.py` / `test_cli_hub.py`, a CI `hub`
job). Judgment call: implementing that properly, with tests, is a
right-sized follow-up project, not a same-session addition on top of
everything else here — attempting it hastily risked shipping a half-built
feature into a subsystem whose whole design point is rigor. Documented the
gap explicitly and honestly instead of quietly leaving it unexplained:
`spikeforge_hub/CURATION.md` now has a "why every entry is bundled" section
naming this exact limitation.

Did implement the other two asks from the report's fix list:

- **A public, static, browsable hub page.** New `scripts/build_hub_page.py`
  (stdlib-only — deliberately does not import `spikeforge_hub`/`spikeforge`,
  so it doesn't drag torch into the landing-page deploy job) renders
  `spikeforge_hub/models.json` to a standalone HTML page: one row per entry,
  HTML-escaped, with a visible "verified source" vs. "unverified candidate"
  badge and an explicit metadata-only notice linking to `CURATION.md`. Wired
  into `.github/workflows/docs-deploy.yml` (the workflow that already
  publishes `landing/` to spikeforge.net) so it deploys to
  `spikeforge.net/hub/` alongside the landing page, without touching the
  landing page's own content or its 17-language i18n system.
- **A documented contribution path.** `CONTRIBUTING.md` has a new "Model-hub
  entries" bullet under "Ways to contribute" pointing at `CURATION.md`'s
  existing verification bar (real source, verified license or the explicit
  `unverified-candidate` marker, checksum where available) and naming the
  tests a proposed entry must still pass.

**Verified:** `python scripts/build_hub_page.py --out <path>` produces a
valid HTML page listing all 10 current entries; new `tests/test_hub_page.py`
(5 tests: loads the real catalog, HTML-escapes untrusted fields, includes
every entry exactly once, `main()` writes the file, the unverified-candidate
marker is the only thing that flips the badge) passes. Ran the exact command
the CI step now runs (`python scripts/build_hub_page.py --out
build/pages/hub/index.html`) end to end.

## 3. 🟠 No "why not just snnTorch/Norse/Lava/SpikingJelly" answer — **Fixed**

Added a "Why not just snnTorch, Norse, Lava, or SpikingJelly?" section to
`README.md`, right after the quickstart: a short paragraph plus a 7-row
comparison table. Before writing it, used web search to check current facts
rather than asserting from memory (this table is exactly the kind of claim
that would undermine the project's own credibility if wrong) — notably, this
corrected an assumption: NIR *export* itself is **not** a spikeforge-only
capability (snnTorch and Norse both support it too — see the "Corrections
after independent review" section below for the one cell this check first
got wrong, on Lava specifically), so the table doesn't claim NIR export in
general as a differentiator; it instead identifies spikeforge's actual
differentiators
(independent-interpreter drift validation of the exported graph, a per-target
deployment capability matrix, a built-in energy estimator, a live dashboard)
and is explicit that the model-hub content gap from §2 is real, in the table
itself, rather than glossing over it. Ends with an honest "you may not need
this" paragraph rather than a hard sell.

**Verified:** every link in the new section resolves to a real file in the
repo (checked programmatically — see "Verification" below); the claims
about NIR-framework support are sourced from
[the NIR paper via Open Neuromorphic](https://open-neuromorphic.org/neuromorphic-computing/software/data-tools/neuromorphic-intermediate-representation/)
(links seven simulators including snnTorch, Norse, and Lava; SpikingJelly is
not on that list, which is why the table hedges on that cell specifically).

## 4. 🟠 Documentation built for the roadmap, not the reader — **Partially addressed, with a correction**

Investigated the specific "two near-duplicate doc trees" claim before
acting on it, and it needed correcting: **`docs/` is entirely
`.gitignore`d** (`.gitignore:5`) — it is `scripts/build_docs.sh`'s local
build output (regenerated from `plans/` + `documentation/` + `README.md` on
every run) and **never appears on the live GitHub repository at all**. A
GitHub visitor never sees two folders; only a contributor who has run the
docs build locally would. That doesn't make the underlying tooling
duplication irrelevant, but it does mean the specific "confuses every
visitor" framing overstated the audience. Given that correction, did not
attempt a risky structural merge/delete of a generated build directory (it
feeds `mkdocs.yml`'s `docs_dir`, `scripts/check_docs_links.py`, and the CI
`docs` job — renaming or removing it without care would break more than it
fixes), and instead:

- Added one clarifying paragraph to the top of `documentation/README.md`
  (the real, hand-maintained source) explaining that a locally-seen `docs/`
  folder is generated, git-ignored, and not a second thing to maintain.
- Fixed the **actually false** claim the report flagged:
  `OPEN_SOURCE_CHECKLIST.md`'s banner said *"Not published. This repository
  is still local... no package has been published"* — false today (public
  GitHub repo, PyPI package). Rewrote the banner, the PyPI-publish-decision
  item (was `[ ]`, now `[x]` with the actual release workflow named), the
  dashboard-screenshot item (was `[ ]`, now `[x]` — `images/dashboard.png`
  exists and is the README hero image), and several stale facts throughout
  (`setup.py` references — that file doesn't exist anymore, packaging moved
  to `pyproject.toml`; version `0.2.0` → `0.3.3`; test count `744/1` →
  `1180/8`; "eight entry points" → thirteen across seven distributions).
- Found and fixed a **pre-existing, unrelated bug** while verifying the docs
  build still passed after these edits: `documentation/dashboard.md` linked
  to `usage.md#access-control--rate-limiting`, but mkdocs generates the
  anchor as `#access-control-rate-limiting` (single hyphen). This was
  silently failing `bash scripts/build_docs.sh --check` (the exact command
  the CI `docs` job runs) before any of my changes — worth flagging since it
  means that CI job would very likely have started failing on the next push
  regardless of this session (`mkdocs-material` is pinned floor-only,
  `>=9.0`, and pulled in a version with stricter anchor-mismatch warnings).

**Verified:** `bash scripts/build_docs.sh --check` (strict mkdocs build +
the project's own link checker) passes clean, both before I started fixing
the anchor bug (confirmed it was failing) and after.

## 5. 🟡 Install/first-run friction vs. the Hugging Face bar — **Fixed**

Added a Docker-free "five lines of Python" path as the *first* thing in
`README.md`'s Quickstart (`pip install spikeforge` + a 5-line
`TrainingEngine` snippet, sourced from and verified against
`examples/01_train_image_model.py`'s actual API), with the Docker dashboard
now presented as "the richer, second path" rather than the default. Mirrored
the same restructuring in `documentation/quickstart.md`, and fixed that
file's stale "once the distributions are published" framing (already
published) and its inaccurate example-script count ("ten runnable
journeys" — actually fifteen; see the correction note at the end of this
report for the arithmetic error a reviewer caught here).

**Verified:** ran the exact snippet now in the README, standalone, from
`/tmp` (not even inside the repo) with nothing but a fresh `pip install
spikeforge` — it downloads MNIST once and trains, printing
`{'loss': ..., 'train_accuracy': ..., 'test_accuracy': ...}`, confirming
the "5 lines, no clone, no Docker" claim is literally true.

## 6. 🟡 Fragmented, independently-versioned distributions, no compatibility story surfaced — **Fixed**

`README.md`'s Packages table was itself stale (claimed "four distributions,"
there are seven) — fixed to list all seven, and added a paragraph
explaining that `spikeforge-hub`/`spikeforge-targets` sitting at `0.1.x`
next to core's `0.3.x` is deliberate independent versioning, not neglect,
pointing at `compatibility.json` as the source of truth for which versions
go together. This was the report's suggested "fold the version-skew
explanation into the README's package table" option, chosen over a runtime
`spikeforge --version` check as the smaller, equally-effective fix.

## 7. 🟡 Delivered vs. aspirational not obvious at a glance — **Fixed**

Added a "What's implemented vs. experimental vs. spec-only" table to
`README.md` (13 rows: training, NIR export/validation, each deployment
target split out — `reference` always-available vs. `norse`/`lava_loihi2`
SDK-gated vs. `spinnaker2`/`speck`/`xylo` spec-only-with-no-installable-SDK
— energy accounting explicitly marked "estimate, not hardware-measured,"
hub catalog infrastructure vs. hub catalog *content* called out as two
separate rows with two separate honest statuses, sequence/attention marked
experimental, ONNX marked single-step, and the production-use-case toolkit
split UC-1 (shipped) from UC-2–10 (spec-only)). Every row links to the
detailed page rather than asserting the claim in isolation. The
target-availability data came from actually calling
`spikeforge_targets.summary.target_summaries()` in a real venv, not from
reading docs — it turned up three more registered targets
(`spinnaker2`/`speck`/`xylo`) than the evaluation report's "Only `reference`
is available" mentioned, and confirmed `spinnaker2`/`speck`/`xylo` have no
corresponding pip extra in `spikeforge-targets`' `pyproject.toml` at all
(no installable path to `available: true` today), which the table now says
plainly.

## 8. 🟡 No social proof / citation positioning / bus factor — **Not addressed (not code-actionable)**

As the evaluation report itself says, this isn't something a work session
fixes directly — it's earned by items 1–4 being fixed, not built. No action
taken; noted here only for completeness against the original list.

## 9. 🟢 Minor polish — **Mostly fixed**

- **`SSNTrainer` typo → `SNNTrainer`.** Confirmed it was a typo, not a
  documented convention (nothing in the docs referenced it as intentional).
  Renamed across the public API (`spikeforge.SNNTrainer`, was
  `spikeforge.SSNTrainer`), `spikeforge/training/trainer.py`,
  `LatencyTrainer`'s base class, `SNNTrainerLogger`'s base class and
  docstrings, `spikeforge/exporters/exporter.py`'s docstring, a comment in
  `spikeforge/memory/held_out_digit_poc.py`, and
  `documentation/project-layout.md`. **This is a breaking public-API
  change** — no compatibility alias was added, matching this project's own
  documented no-back-compat-shim policy (`plans/arch-0001-migration-plan.md`).
  Flagged prominently in `CHANGELOG.md` under `[Unreleased]` / "Changed" as
  breaking. Not renamed: two historical planning documents
  (`INTEGRATION_PLAN.md`, `EVALUATION_REPORT.md`) that reference the old
  name alongside other already-stale file paths from an earlier phase of the
  project — fixing the class name there without fixing the surrounding
  stale paths would have made those documents more, not less, misleading;
  out of scope for this pass.
- **`images/dashboard.png` undersold.** Fixed in both places the report
  named: `OPEN_SOURCE_CHECKLIST.md` (see §4) and
  `documentation/dashboard.md`, which claimed "no image is committed" —
  now shows the real image inline and accurately states one of five
  planned captures is done, not zero.
- **Landing page translated before the CLI worked.** A sequencing
  observation about past prioritization, not a defect to fix now — no
  action taken, noted for completeness.

---

## Bonus fixes found while verifying the above

Not named in the evaluation report, found by actually running the tools
the report's own fixes depend on:

- `CONTRIBUTING.md` had the same class of staleness as
  `OPEN_SOURCE_CHECKLIST.md`: a test-count baseline of "739 passed, 1
  skipped" and an extras list (`hub`, `norse`, `lava` as core extras) that
  predates the `spikeforge-targets`/`spikeforge-hub` package split. Fixed
  both.
- The broken anchor link in `documentation/dashboard.md` (§4 above) — was
  already failing `bash scripts/build_docs.sh --check` before this session.

## What was not attempted, and why

- **Real trained model-hub checkpoints** (§2) — requires a real feature
  addition (new entry kind + materialization path + schema + tests) inside
  a deliberately strict, already-well-tested subsystem; judged as a
  right-sized follow-up rather than something to rush. See
  `spikeforge_hub/CURATION.md`'s new section for the exact technical reason
  (`_materialize()` always rebuilds from the preset with fresh random
  init — there is no code path today for a bundled entry to carry stored
  weights).
- **Hosted demo / Colab / Binder** (§5) — a notebook committed without ever
  actually running in Colab risks shipping something broken and untested,
  which would be worse than the current gap; did not fabricate one.
- **Social proof, citations, bus factor** (§8) — not code-actionable, as
  the evaluation report itself notes.

## Corrections after independent review

An independent reviewer reproduced the work in this report from scratch
(fresh wheel build, clean venv, full suite, ruff, core-boundary scanner,
strict docs build) and fact-checked the new comparison table against
primary sources. §1's fixes held up entirely under that review — no changes
needed there. Two real errors were found and are now fixed:

- **The positioning table's Lava row (§3).** It marked Lava ✅ for "NIR
  export" without distinguishing read from write. Checked NIR's own
  framework-support table
  ([neuromorphs/NIR](https://github.com/neuromorphs/NIR)): Lava-DL reads
  NIR but does not write it (only jaxsnn and SpiNNaker2 share that
  read-only status; snnTorch, Norse, Nengo, Rockpool, Sinabs, and Spyx all
  support both directions). `README.md`'s table now reads "NIR export
  (write)" with Lava's cell corrected to "import-only," and the prose above
  the table no longer claims Lava exports NIR.
- **The example-script count (§5).** Reported as "fourteen" in
  `documentation/README.md` and `documentation/quickstart.md`. The
  reasoning was wrong, not the source: `examples/` has two files both
  numbered `11` (`11_held_out_digit_memory.py`,
  `11_streaming_timeseries.py`), and that got misread as "one fewer file
  than the highest number" instead of just counting the files
  (`ls examples/*.py | wc -l` = 15). Both docs now correctly say fifteen —
  restoring what `documentation/README.md` said before this session's edit
  touched it.

## Verification summary

Everything below was actually run in this session, not asserted:

| Check | Result |
|---|---|
| Full test suite (`pytest`) | **1185 passed, 8 skipped** (up from 1180/8 — the 5 new tests are `tests/test_hub_page.py`; all 8 skips are pre-existing, expected: missing `onnx` extra) |
| `ruff check .` | clean |
| `scripts/check_core_boundary.py` | clean, 242 modules scanned |
| `bash scripts/build_docs.sh --check` (strict mkdocs + link check) | clean |
| `scripts/check_console_scripts.py` against a genuinely minimal wheel install (no extras) | all 5 core console scripts OK |
| Same, with `spikeforge-targets`/`spikeforge-hub` also installed | no regression — all 14 `spikeforge-verify` subcommands still present |
| `spikeforge` with no args (real training run, cached local MNIST) | still produces all 5 output artifacts, unchanged behavior |
| The README's new 5-line quickstart snippet, standalone from `/tmp` | runs, trains, prints metrics |
| Headless-wheel boundary check (no forbidden deps, no leaked satellite roots) | clean, rebuilt from a fresh wheel |
| Every relative link added to `README.md` | resolves to a real file (checked programmatically) |

## State of the working tree

Nothing has been committed. `git status` shows modifications to 30 tracked
files plus 4 new source files (`spikeforge/targets_extra.py`,
`scripts/check_console_scripts.py`, `scripts/build_hub_page.py`,
`tests/test_hub_page.py`), this report, and `EVALUATION_REPORT.md` (the
source document for this work, which predates this session and was left
untouched). `CHANGELOG.md`'s `[Unreleased]` section has a full, categorized
account of every change for whoever writes the actual release notes.

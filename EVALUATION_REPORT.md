# Spikeforge — adoption evaluation

**Question asked:** what stands between this project and being the default choice
for anyone who wants to touch spiking neural networks — hobbyist, academic, or
professional? What causes silent refusal, what causes head-scratching, and what
would make the answer "obviously yes"?

**Method:** read the README, all of `documentation/`, `docs/`, `plans/`,
`COOKBOOK.md`, `examples/`, the packaging metadata, `OPEN_SOURCE_CHECKLIST.md`,
CI config, and GitHub issues; checked the live GitHub repo, PyPI, the landing
page, and the wiki; and — most valuably — actually did what a new user does:
`pip install "spikeforge[all]"` into a clean virtualenv and ran the first
command the README tells you to run. Findings below are graded by how many
people they turn away, not by engineering elegance.

**Bottom line:** the engineering underneath is unusually rigorous for a
project this young — typed error paths, an honesty-over-convenience design
philosophy, a real NIR interop spine, 145 test files, green CI. That
rigor is invisible to a newcomer, because the first five minutes are broken
in ways Hugging Face-caliber projects never let ship, and the documentation
is written for the roadmap's own contributors, not for the person deciding
whether to adopt it. Fix the first five minutes, give the hub actual content,
and answer "why not just use snnTorch" — those three things matter more than
any new feature.

---

## 0. Severity key

- 🔴 **Blocker** — a new user hits this in the first 10 minutes and leaves.
- 🟠 **Major** — doesn't block the first run, but blocks "I'll use this for
  real work" or "I'll cite/recommend this."
- 🟡 **Medium** — friction and head-scratching; survivable, costs trust.
- 🟢 **Minor / polish** — small, but visible, and cheap to fix.

---

## 1. 🔴 The flagship CLI command is broken on a clean install

The README's Quickstart says, in the "prefer a local install" branch:

```
spikeforge --help         # the library CLI
```

This is the single most likely first command a new user runs after
`pip install`. Reproduced from scratch:

```bash
python -m venv env && source env/bin/activate
pip install "spikeforge[all]"
spikeforge --help
```

Result on a genuinely clean install:

```
ModuleNotFoundError: No module named 'pandas'
```

Root cause, traced end to end:

- The `spikeforge` console script is wired to `main:main`
  ([`packages/spikeforge/pyproject.toml`](packages/spikeforge/pyproject.toml)
  → `spikeforge = "main:main"`).
- [`main.py`](main.py) unconditionally imports
  `spikeforge.exporters.presentation_exporter` at module level — before
  `main()` is even called, and regardless of what argv contains.
- [`spikeforge/exporters/presentation_exporter.py:5`](spikeforge/exporters/presentation_exporter.py)
  does `import snntorch.spikeplot as splt`.
- `snntorch.spikeplot` does `import pandas as pd` at module level, but
  `snntorch`'s own PyPI metadata declares **no dependency on pandas**
  (confirmed via `pip show snntorch` → `Requires:` empty). That's an upstream
  snntorch packaging gap, but spikeforge inherits it silently because nothing
  in spikeforge's own dependency list mentions pandas either.

Installing `pandas` by hand and retrying exposes the second, worse problem:
**`main()` takes no arguments and never looks at `sys.argv`.** `spikeforge
--help` doesn't print help — it silently ignores `--help`, downloads MNIST
over the network, and starts training:

```
MNIST training subset size: 6000
Converted vector: tensor([1., 0., 0., 0., 1., 1., 1., 0., 0., 0.])
The output is spiking 40.00% of the time
...
```

So the advertised "library CLI" is actually the pre-professionalization
`main.py` demo script (its own docstring: *"train a rate-coded MNIST subset
and export all visuals"*), wearing the `spikeforge` command name. Meanwhile
the *real*, well-built CLIs in this repo — `spikeforge-hub --help`,
`spikeforge-targets --help`, `spikeforge-verify --help` — all use `argparse`
properly, show real subcommands, and behave exactly as expected. The
flagship-named entry point is the one exception, and it's the one everyone
tries first.

**Why this matters more than its size suggests:** this is exactly the kind of
thing a Hugging Face-caliber library would never ship — a first command that
crashes on a stock install, and even when it doesn't crash, silently does
something unrelated to what was asked and reaches out to the network without
being asked to. One bad first command is enough for a lot of evaluators to
close the tab.

**Fix shape (not full design, just direction):**
- Make `spikeforge.exporters.presentation_exporter` (and anything else that
  drags in `snntorch.spikeplot`/matplotlib/pandas-adjacent stacks) a lazy
  import inside the function that needs it, not a module-level import.
- Give `main.py`/the `spikeforge` script real `argparse` with a `--help` that
  actually prints help and does nothing else — matching the discipline
  already present in the sibling CLIs.
- Longer-term: consider whether the bare `spikeforge` command should exist at
  all, versus deprecating it in favor of the already-consistent
  `spikeforge-verify`/`spikeforge-hub`/`spikeforge-targets` family, so there's
  one CLI design language instead of two.
- Either way, add a CI smoke check that runs every declared console script
  with `--help` (or no args) against a **minimal** install (not `[all]`, not
  `[dev]`) and asserts exit 0. Nothing in the current CI extras matrix
  (`.github/workflows/ci.yml`) would have caught this, because it always
  installs the extras together.

---

## 2. 🟠 The model hub has an excellent funnel and almost no content

This is very likely what prompted "we're missing a hub," even though a
`spikeforge_hub` package already exists and is genuinely well designed:
curated catalog, offline-first, a three-gate inspect → compat → promote
pipeline, checksum verification, cancellable downloads, an honest
`unverified-candidate` state, and an opt-in live Hugging Face search. The
plumbing is there. The problem is what's inside it.

Look at [`spikeforge_hub/models.json`](spikeforge_hub/models.json): **all 10
entries** are `"source": "bundled"` NIR graphs re-rendered from this
project's own four tutorial presets (`fc_legacy`, `fc_small`, `conv_net`,
`recurrent_net`), labelled by which *framework* they nominally represent
(snnTorch, SpikingJelly, Norse, Lava). None of them carry trained weights
from anywhere — the notes field says so explicitly: *"no third-party weights
are redistributed"* / *"ecosystem provenance only."* A user who opens the hub
expecting a ResNet-for-SNNs, a pretrained N-MNIST classifier, or a Loihi-ready
gesture-recognition model finds ten copies of the same four untrained
architecture skeletons.

Compare to what "a model hub" means to literally everyone who's used Hugging
Face: a place to get **weights that already work**, with a model card, a
license, and a one-liner to load them. Right now spikeforge's hub is a
*catalog of shapes*, not a *catalog of models*. That gap — not the missing
UI — is what will make people say "there's no hub here."

There is also no *website*. `huggingface.co/models` is a public, indexable,
browsable, shareable surface; spikeforge's hub is a JSON file plus a CLI plus
a panel inside a dashboard you have to run locally. Nothing about it is
discoverable by someone who isn't already running the project. There is also
no path for outside contributors to add a model — `CURATION.md`'s policy
(verified source + concrete license, or `unverified-candidate`) is sound, but
there's no documented *process* (PR template, issue template, or upload
mechanism) for someone to propose a new entry.

**What would close this gap:**
- Populate the catalog with real trained checkpoints for the datasets and
  topologies already shipped (MNIST/Fashion-MNIST/KMNIST/N-MNIST/DVS128 ×
  `fc_small`/`conv_net`/`recurrent_net`) — even a handful of honestly-labelled
  "reference checkpoints, not SOTA" entries beats zero.
- A public, static, browsable hub page (can be generated from
  `models.json` — same instinct as the existing docs-site generator) so a
  search engine and a casual visitor can find it without installing anything.
- A documented contribution path for community-submitted entries (a PR
  against `models.json` plus the CURATION.md checklist, wired into the
  bug/feature issue templates), so the hub can grow past what one maintainer
  can personally verify.

---

## 3. 🟠 No answer to "why not just use snnTorch/Norse/Lava/SpikingJelly directly"

Spikeforge is explicitly *built on* snnTorch (`snnTorch>=1.0` is a hard
dependency, and the README leads with "built on snnTorch and PyTorch"). That
means the very first question any academic or professional who already knows
this space will ask is: what does this give me that snnTorch doesn't? The
repo actually has a strong answer — the NIR interpreter spine with drift
validation, the honest capability/deployment-matrix approach to hardware
targets, event-driven energy accounting, the unified dashboard, cross-library
NIR interop — but **that answer exists nowhere as a comparison**. There is no
"spikeforge vs. snnTorch vs. Norse vs. Lava vs. SpikingJelly" table anywhere
in the README, the docs, or the landing page. `CITATION.cff` and the landing
page both describe *what it does* in isolation, never *relative to the
alternatives a reader already has installed*.

For an academic deciding what to cite or build on, and for a professional
deciding what to depend on, this is a real gap, not a nice-to-have: without
it, the honest and rigorous engineering underneath (interop spine, energy
accounting, honest capability matrix) is invisible unless the reader already
digs through `plans/professional_roadmap.md`.

**What would close this gap:** a short, concrete positioning section near the
top of the README — one table, four or five rows (training, NIR export,
cross-framework interop, hardware deployment reporting, energy accounting,
dashboard) — naming where snnTorch/Norse/Lava/SpikingJelly stop and where
spikeforge picks up. This is a documentation task, not an engineering one,
and it's probably the single highest-leverage paragraph missing from the
project.

---

## 4. 🟠 Documentation is comprehensive but built for the roadmap, not the reader

There is a *lot* of documentation — genuinely more than most projects this
young — but almost all of it is written in "engineering design record" voice:
phase status blocks, ADRs, workstream tables, acceptance bars, Mermaid
dependency graphs. That's a legitimately good way to keep a fast-moving,
heavily AI-assisted codebase honest and consistent (and it clearly works —
the "honesty rule" shows up everywhere in the actual code paths, not just the
docs). But it is not how a newcomer — student, academic, or professional
evaluating the library for the first time — wants to be greeted.

Concrete symptoms:

- **Two near-duplicate doc trees.** [`docs/`](docs/) and
  [`documentation/`](documentation/) both exist at the repo root, with
  overlapping filenames (`quickstart.md`, `usage.md`, `architecture.md`,
  `model-hub.md`, `backend-execution.md`, `introspection.md`, …). Nothing
  signals which is canonical at a glance; `docs/index.md` explains that
  `docs/` is roadmap/plans-derived and `documentation/` is meant for users,
  but a first-time visitor has no way to know that without opening both.
- **The README explicitly defers everything.** "This README stays short on
  purpose" is a defensible choice, but it means the *first* document a GitHub
  visitor sees has almost no task-oriented content — no "train your first
  model in 5 lines," no "here's what you get," just a features list and a
  pointer to `documentation/`.
- **Stale, actively wrong status claims.** `OPEN_SOURCE_CHECKLIST.md` states,
  in its own banner: *"Not published. This repository is still local. Nothing
  has been pushed to a remote, no release has been created, and no package
  has been published."* This is false today — the repo is public on GitHub
  (`capsize-games/spikeforge`, live CI, a real landing page at
  spikeforge.net) and `spikeforge` is published on PyPI at `0.3.3` with real
  download counts. A document a newcomer is pointed to for "is this
  trustworthy/production-ready" telling them the opposite of reality is worse
  than not having the document — it signals that the docs aren't kept in
  sync with what actually shipped, which invites doubt about every other
  claim in the same file (test counts, "verified" statuses, etc.).
- **Roadmap documents describe finished work in the past tense, right next to
  documents describing unstarted work in the future tense, with the same
  formatting** — `docs/ecosystem_roadmap.md` and `docs/professional_roadmap.md`
  both open with "ROADMAP COMPLETE / DELIVERED" banners for their own scope,
  while `docs/index.md`'s production-use-case table shows 8 of 10 use cases
  as merely "scoped" (spec only, issues #13–#21). A skimming reader can very
  easily come away thinking more is implemented than actually is, because the
  visual weight (headers, checkmarks, "delivered" banners) is identical
  whether something shipped or is still a design doc.

**What would close this gap:** pick one canonical docs location (delete or
fully merge the other), lead the README with a real 5-minute task rather than
a features list, and add a single, honestly-labelled "what's implemented vs.
what's a design doc" status page instead of scattering that signal across
many similarly-formatted files. None of this requires new engineering.

---

## 5. 🟡 Install/first-run friction relative to the Hugging Face bar

Directly comparing against the bar named in the prompt:

| | Hugging Face (`transformers`) | spikeforge today |
|---|---|---|
| Install | `pip install transformers` | `pip install "spikeforge[all]"` — works, ~40s, pulls full CUDA stack (inherent to any torch project, not spikeforge-specific) |
| First working code | `pipeline("sentiment-analysis")("I love this")` — 1 line, no clone needed | No equivalent ships in the wheel. The runnable "journeys" live in [`examples/`](examples/) at the repo root, which **is not included in the PyPI package** — a `pip`-only user has nothing to run without separately fetching the GitHub repo. |
| Primary quickstart | pip + Python snippet | `git clone` + `docker compose up --build`, opening a browser dashboard. Docker-as-default-path is real friction next to "pip install and go." A pip-only path exists (`pip install spikeforge-server`) but is presented as the secondary option. |
| Try before installing | Spaces / hosted demos everywhere | No hosted demo, no Colab notebook, no Binder link anywhere in the repo or landing page. |

None of this is disqualifying on its own — a training/experimentation
toolkit built on PyTorch is never going to be as light as a single inference
call — but taken together, the *shape* of the onboarding funnel (clone →
Docker → browser) is inverted from what the audience named in the prompt
(developers who want to drop a dependency into an existing project, fast)
expects. The fix isn't to drop the dashboard, it's to put a "5 lines of
Python, no Docker" path first in the README, with Docker/dashboard offered as
the richer second option — and to ship 2-3 of the existing `examples/`
scripts as importable, `pip`-reachable code (or at least document `pip
download`-plus-copy as the intended path) rather than requiring a full clone.

---

## 6. 🟡 Fragmented, independently-versioned distributions with no compatibility story surfaced to the user

The project ships **seven** separate PyPI-shaped distributions from one
monorepo: `spikeforge`, `spikeforge-targets`, `spikeforge-hub`,
`spikeforge-server`, `spikeforge-serve`, `spikeforge-clients`,
`spikeforge-io` (see [`packages/`](packages/) and the split-out sibling repos
`spikeforge-hub`, `spikeforge-targets`, `spikeforge-dashboard`). That's a
defensible architecture — it mirrors how mature ecosystems eventually split —
but it's happening very early (issue #1, "Phased repo split," and the
`arch-0001-*` ADRs are all about this), and the version numbers currently
visible to a user don't inspire confidence: after installing
`spikeforge[all]`, `pip list` shows `spikeforge-0.3.3` next to
`spikeforge-hub-0.1.0` and `spikeforge-targets-0.1.1`. A `compatibility.json`
matrix exists to reconcile which versions of which package go together, but
it isn't surfaced anywhere a `pip`-only user would see it (not in the README,
not printed by any CLI, not checked at import time). To someone unfamiliar
with the project, "core at 0.3.3, satellites stuck at 0.1.x" reads as
"abandoned sub-packages," not "deliberately independent versioning" — even
though the latter is true.

**What would help:** either print/expose the compatibility matrix at
runtime (e.g. `spikeforge --version` or a warning if an incompatible
combination is detected), or fold the version-skew explanation into the
README's package table so it's not left to be inferred from `pip list`.

---

## 7. 🟡 Feature breadth is described uniformly; delivered vs. aspirational is not always obvious at a glance

The project is candid, in the fine print, about what's real: "Only
`reference` [target] is available," energy numbers are "estimates," ONNX
export is "single-step," ~~8 of the 10 documented production use cases~~ are
listed as "scoped" rather than implemented in `docs/index.md`'s own table.
This honesty is a genuine strength and should be preserved — it's rare and
valuable. The issue is purely one of **visual hierarchy**: a roadmap doc
describing a shipped phase and a roadmap doc describing an unstarted phase
use the same headers, the same Mermaid diagrams, and similar-looking status
banners, so a reader has to read carefully to tell "this runs today" from
"this is a spec for issue #19." For a professional deciding whether a
capability (say, spiking-transformer sequence support, or Lava/Loihi
deployment) is usable *now*, that distinction is exactly what they came to
find, and it currently costs more reading effort than it should.

**What would help:** a single, prominent "capability matrix" (implemented /
experimental / spec-only, one row per headline feature) near the top of the
README or docs index, so the honesty the project already practices is
scannable in ten seconds instead of requiring a read of several
roadmap documents.

---

## 8. 🟡 No social proof, no positioning for citation, thin bus factor

Not a code gap, but directly relevant to "what makes someone refuse to
adopt": the GitHub repo currently has **1 star, 0 forks**, all commits trace
to a single human identity (plus an AI pair-programmer co-author), and there
are no external contributors. For a professional this reads as "unproven";
for an academic deciding what to cite or build a thesis on, it reads as
"risky to depend on" — there's a `CITATION.cff` (good) but no paper, no
benchmark numbers against published SNN results, and no third-party usage to
point to. This isn't something the next work session can fix directly (it's
earned, not built), but it interacts with everything else in this report:
fixing items 1–4 (a working CLI, real hub content, clear positioning, honest
capability matrix) is the actual lever for earning the stars/forks/citations
that are currently missing — polishing further before that won't move this
number.

---

## 9. 🟢 Minor polish items (cheap, visible)

- **`SSNTrainer`** — the flagship class exported from
  [`spikeforge/__init__.py`](spikeforge/__init__.py) and defined in
  [`spikeforge/training/trainer.py:13`](spikeforge/training/trainer.py) reads
  as a typo of `SNNTrainer` to any newcomer skimming the public API (there's
  a separate, correctly-named `SNNTrainerLogger` right next to it in the same
  `__all__`, which makes the inconsistency more visible, not less). If it's
  intentional it should be a documented convention; if not, it's the kind of
  thing that makes an evaluator wonder what else wasn't proofread.
- **`images/dashboard.png`** is a real screenshot (verified: genuine 915×925
  PNG, not a placeholder), but the checklist and README both flag dashboard
  media as incomplete/missing in places — worth reconciling so the README
  doesn't undersell an asset that already exists.
- The landing page was translated into 17 languages before the primary CLI
  command reliably works from a clean install. Not wrong, but worth naming
  as a sequencing question for future prioritization: polish that widens
  reach matters less than polish that prevents bounce on the very first
  command.

---

## 10. What would make it "impossible to say no" — priority order

1. **Fix `spikeforge --help`** (§1) and add a CI smoke test that runs every
   console script's `--help` against a minimal (non-`[all]`, non-`[dev]`)
   install. This is small and urgent — it's actively costing first
   impressions today.
2. **Write the positioning section** (§3): one table, "spikeforge vs.
   snnTorch/Norse/Lava/SpikingJelly." Pure documentation, highest
   leverage-per-hour item in this whole report.
3. **Add a real capability matrix** (§7) near the top of the docs: shipped /
   experimental / spec-only, one glance, no digging through roadmap docs.
4. **Give the hub real content** (§2): even a small number of genuinely
   trained, honestly-labelled checkpoints beats ten copies of untrained
   preset shapes, plus a static browsable page and a documented community
   contribution path.
5. **Put a Docker-free, clone-free "5 lines of Python" quickstart at the top
   of the README** (§5), with the Docker dashboard offered as the richer
   second path rather than the first.
6. **Consolidate `docs/` and `documentation/`** into one canonical tree (§4),
   and fix or delete the stale "not published" claim in
   `OPEN_SOURCE_CHECKLIST.md` before anyone else reads it.
7. Everything else in this report (versioning-skew story, hosted demo/Colab,
   community proof) compounds on top of 1–6 rather than substituting for
   them.

None of the above requires new SNN capability — the interpreter spine,
honesty discipline, and deployment/energy accounting are already
differentiated and worth keeping exactly as they are. What's missing is
almost entirely: one broken command, one missing paragraph, one empty
catalog, and two folders that should be one.

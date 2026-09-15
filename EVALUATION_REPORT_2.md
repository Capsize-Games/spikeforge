# Spikeforge — adoption evaluation, cycle 2

**Question asked:** same as cycle 1 — what still stands between this project and
being the default choice for anyone who wants to touch spiking neural networks?
What is left that causes silent refusal or head-scratching? And: are we now at
the point where the remaining work is purely promotion?

**Method:** re-read the README, `documentation/`, `plans/`, `COOKBOOK.md`,
`examples/`, packaging metadata, CI config, `CHANGELOG.md`, the cycle-1
`EVALUATION_REPORT.md` and `FIXES_REPORT.md`; checked open GitHub issues, CI
run history, and all seven live PyPI projects; fetched the live landing page,
the live hub page, the live dashboard demo, and the live documentation site;
rendered the README through **PyPI's own renderer** (`readme_renderer`) to see
what a PyPI visitor actually gets; built the GitHub wiki locally from current
`main` with the project's own `scripts/build_wiki.py` and audited every link it
emits; and — as in cycle 1 — did what a new user does: `python -m venv` →
`pip install spikeforge` from PyPI → ran the console scripts and the README's
own quickstart snippet.

**Bottom line:** cycle 1's work held up. Every 🔴/🟠 fix from that report is
real, shipped, and verified against the *published* 0.3.4 wheel, not just the
working tree — the first five minutes are genuinely fixed now. But the answer
to "is it purely promotion from here?" is **no, not yet**, and for a reason
worth naming precisely: cycle 1 fixed the *repository*, and the repository is
not where most people meet this project. The two surfaces that a stranger
actually lands on — **the PyPI project page** and **the published documentation
site** — are both visibly broken right now, in ways that are cheap to fix and
that no CI gate can currently see. Those are items 1 and 2 below and they
outrank everything else. After those, the ranking is: the hub still has no
weights (unchanged, and it is the thing you originally named), the landing page
still sends people to `git clone` while the README sends them to `pip install`,
and there is no published accuracy number anywhere for an academic to check.
The pure-promotion work is real and is item 6 — but doing it *before* 1 and 2
means driving traffic to broken pages.

---

## 0. Severity key

- 🔴 **Blocker** — a stranger hits this within the first few minutes on a
  surface we control, and it costs the impression outright.
- 🟠 **Major** — doesn't break first contact, but blocks "I'll use this for real
  work" / "I'll cite this" / "I'll recommend this."
- 🟡 **Medium** — friction and head-scratching; survivable, costs trust.
- 🟢 **Minor / polish** — small, visible, cheap.

---

## Cycle 1: what verifiably holds up (do **not** redo this work)

Confirmed against the published `spikeforge 0.3.4` wheel in a clean venv, and
against the live sites — not against the working tree:

| Cycle-1 finding | Status now | How it was verified this pass |
|---|---|---|
| §1 🔴 `spikeforge --help` crashed / silently trained | **Fixed and shipped** | Clean `python -m venv`, `pip install spikeforge` (0.3.4, no extras). All five console scripts (`spikeforge`, `spikeforge-encodings`, `spikeforge-verify`, `spikeforge-records`, `spikeforge-benchmark`) run `--help` → exit 0, print real help, touch no network. `spikeforge-verify --help` degrades honestly, naming the `spikeforge-targets` install for the absent deployment subcommands. |
| §5 🟡 No pip-only, Docker-free first run | **Fixed** | Ran the README's exact 5-line snippet in the clean venv from an empty directory: trains in **13.6 s**, prints `{'loss': 1.60, 'train_accuracy': 0.83, 'test_accuracy': 83.45}`. The claim is literally true. |
| §3 🟠 No "why not just snnTorch/Norse/Lava" answer | **Fixed** | Present in `README.md`, 7-row table, with the Lava import-only correction applied and an honest "you may not need this" close. |
| §7 🟡 Delivered vs. aspirational not scannable | **Fixed** | `README.md`'s 13-row implemented/experimental/spec-only table is present and each row links to detail. |
| §6 🟡 Version-skew story not surfaced | **Fixed (docs half)** | README's Packages table lists all seven distributions and explains the `0.1.x` / `0.3.x` split, pointing at `compatibility.json`. (The runtime half was deliberately skipped — see item 7 below.) |
| §9 🟢 `SSNTrainer` typo | **Fixed** | `spikeforge.SNNTrainer` in the published wheel; old name gone, breaking change recorded in `CHANGELOG.md`. |
| §2 🟠 Hub had no public browsable surface | **Fixed** | <https://spikeforge.net/hub/> is live, renders all 10 entries, and is honest in its own copy about being metadata-only. |
| §2 🟠 No documented hub contribution path | **Fixed** | `CONTRIBUTING.md` has the Model-hub bullet pointing at `CURATION.md`'s bar. |
| CI gate for the above | **In place and green** | `scripts/check_console_scripts.py` wired into the `headless` job; latest CI run on `main` is green across all jobs. |

Two cycle-1 items were **explicitly deferred** and remain open by design — they
reappear below as items 3 and 12, not as criticism of that judgment call.

---

## 1. 🔴 The PyPI project page — the storefront for the `pip install` audience — is visibly broken

Cycle 1 correctly moved the README to lead with `pip install spikeforge`. That
makes PyPI the front door for exactly the audience that change was aimed at.
That front door is currently in worse shape than the GitHub one, and nothing in
CI looks at it.

All seven `packages/*/pyproject.toml` set `readme = "README.md"`, and in every
case that file is a **committed symlink to the repository-root README**
(verified: `packages/spikeforge/README.md -> ../../README.md`, same for
`spikeforge-hub`, `spikeforge-targets`, and the rest). The root README is
written for GitHub, which resolves relative links against the repo. **PyPI does
not.** Rendered through `readme_renderer` — the exact library PyPI uses:

- **30 dead relative links** on the page, including every single pointer to the
  documentation (`documentation/usage.md`, `documentation/quickstart.md`,
  `documentation/interpreter-spine.md`, …), `COOKBOOK.md`, `examples/`,
  `plans/`, `LICENSE`, `CITATION.cff`, `compatibility.json`, and
  `spikeforge_hub/models.json`.
- **The hero image does not render.** `![spikeforge dashboard](images/dashboard.png)`
  is the only relative `src` among nine images (the eight badges are absolute
  and fine). A PyPI visitor gets a broken-image icon where the dashboard
  screenshot should be — the single strongest visual asset the project has, on
  the page where it would do the most work.
- Because all seven distributions symlink the same README, this is **seven
  broken project pages**, not one.

Compounding it, the metadata itself is stale in a way that actively undersells:

- `description` (the one line shown in PyPI search results and by `pip show`)
  reads *"Spiking neural network trainer/experiments for **MNIST** built with
  snnTorch and PyTorch."* The project trains on eight image datasets plus four
  neuromorphic event datasets, exports NIR, validates drift, deploys to three
  executable backends and estimates energy. The storefront one-liner says
  "MNIST experiments."
- `keywords` still lists `mnist` and omits `neuromorphic`, `nir`,
  `spike-encoding`, `lif`, `loihi`, `energy` — note the *GitHub* repo topics
  already carry the right set, so the two are out of sync.
- `[project.urls]` has only `Homepage` and `Repository`, both pointing at
  GitHub. There is no `Documentation` (→ <https://docs.spikeforge.net/>), no
  `Changelog`, no `Issues`, no `Homepage` pointing at spikeforge.net — so
  PyPI's sidebar, which is prime real estate, is nearly empty.
- Classifiers omit `Typing :: Typed` even though the core distribution ships
  `py.typed`.

**Fix shape:** make the packaged README absolute-linked. Two workable routes —
either (a) rewrite the root README's relative links to absolute
`https://github.com/capsize-games/spikeforge/blob/main/...` /
`https://raw.githubusercontent.com/.../images/dashboard.png` URLs (simplest,
one file, and it costs nothing on GitHub where absolute links work fine), or
(b) generate a PyPI-specific README at build time from the root one. Then fix
`description`, `keywords`, `urls`, and the `Typing :: Typed` classifier in all
seven `pyproject.toml` files. **Add a CI gate** in the same spirit as
`check_console_scripts.py`: render the packaged README with `readme_renderer`
and fail on any relative `href`/`src`. That gate is what keeps this from
regressing the next time the README is edited.

---

## 2. 🔴 The published documentation site has 896 dead links, and the CI link checker cannot see them

`docs.spikeforge.net` redirects to the **GitHub wiki** (verified: 302 →
`github.com/Capsize-Games/spikeforge/wiki/`). The wiki is generated from
`documentation/` and `plans/` by `scripts/build_wiki.py` on every push to
`main`. That generator rewrites *only* relative links that end in `.md` **and**
resolve to another page in the wiki set (`rewrite_links()`,
[`scripts/build_wiki.py`](scripts/build_wiki.py)). Everything else is emitted
verbatim.

The documentation is dense with exactly the links that don't survive that
rule — source-code references like
`[TrainingEngine](spikeforge/training/training_engine.py:1)`, which are
excellent on GitHub and meaningless on a wiki.

Reproduced by running the project's own builder against current `main`:

```
wiki pages generated: 61
DEAD relative links on the published docs site: 896 across 50 pages
    documentation/-sourced pages:  124 dead links across 17 pages
    plans/-sourced pages:          755 dead links across 31 pages
    other:                          17 dead links across  2 pages
```

Worst offenders: `professional-roadmap` (116), `model-hub-plan` (55),
`production-toolkit-plan` (55), `production-use-cases` (44).

**What the reader experiences is worse than a 404.** Spot-checked live:
`https://github.com/Capsize-Games/spikeforge/wiki/setup.py` and
`.../wiki/spikeforge/topology/spec.py:16` both return HTTP 200 and silently
render the wiki **Home** page (page title `Home · Capsize-Games/spikeforge
Wiki`), while a genuine page like `.../wiki/architecture` renders as itself. So
a reader clicking a source reference in the docs is teleported back to the
documentation home page with no error and no explanation. The Home page itself
has three such links (`../rules.md`, `../protocol/`,
`../OPEN_SOURCE_CHECKLIST.md` — confirmed 404/406 live).

**Why CI is green anyway:** `scripts/check_docs_links.py` states its contract in
its own docstring — *"The plan documents intentionally reference repository
source files … which are not documentation pages"* — and therefore checks only
`.md`-suffixed relative links, resolved against the **mkdocs** tree in `docs/`,
where they do resolve. That contract was written for a mkdocs site. The
published site is now a wiki, and the checker was never re-pointed at it. The
gate is passing on an artifact nobody publishes.

Which surfaces a second, related problem: **the mkdocs site is built,
strict-mode-validated and link-checked in CI (`docs` job,
`scripts/build_docs.sh --check`) and then published nowhere.** `spikeforge.net/docs/`
is a 404; `docs.spikeforge.net` goes to the wiki. `mkdocs.yml`, `docs/`, and
`scripts/build_docs.sh` are maintained infrastructure producing no artifact,
while the artifact people actually read has no gate.

Third, smaller, but it sets the tone for every visitor: the **front page of the
public documentation site** is `documentation/README.md`, whose second sentence
reads *"It is written for contributors and LLM coding agents that need the full
picture."* A developer who lands on docs.spikeforge.net is told in the opening
paragraph that these docs are not for them.

**Fix shape:** decide which site is canonical, then make the gate match it.
Either (a) teach `build_wiki.py` to rewrite non-page relative links into
absolute `blob/main` GitHub URLs (this single change fixes all 896 at once,
turns every source reference into a *working* one, and is the smallest possible
diff), or (b) publish the mkdocs site at docs.spikeforge.net and retire the
wiki. Either way, extend the link check to run against the **published**
output. And rewrite the docs homepage opening for a reader who is evaluating
the library, not contributing to it.

---

## 3. 🟠 The model hub still has no trained weights — unchanged since cycle 1

This is the item you originally named ("we're missing a hub"), and it is the
one substantive gap that both cycles have now left open. Cycle 1 made a
defensible call to defer it, delivered the two surrounding asks (public page,
contribution path), and documented the blocker honestly in
`spikeforge_hub/CURATION.md`. All of that is real. But the gap itself is
untouched:

`spikeforge_hub/models.json` still contains exactly 10 entries, all
`"source": "bundled"`, all NIR renderings of the same four shipped presets
(`fc_legacy`, `fc_small`, `conv_net`, `recurrent_net`), re-labelled by which
framework they nominally represent. Zero carry trained weights. The blocker
named in cycle 1 still stands: `spikeforge_hub/inspect.py`'s `_materialize()`
rebuilds every bundled entry from its preset with fresh random init, so there
is no code path for a catalog entry to carry stored weights at all.

The project's own surfaces now say this plainly, which is to its credit and
also makes the gap more conspicuous than it was:

- <https://spikeforge.net/hub/> opens with *"This catalog is metadata-only …
  structure, not third-party trained weights."*
- The landing page status table has a row reading **"Model hub — pretrained
  weights: not started."**

So a visitor who follows "Model hub" from the feature list reaches a public page
that tells them there are no models. That is honest, and it is still a visitor
who leaves.

Two things also worth fixing while in here, both cheap:

- **The hub page is orphaned.** The landing page links Features, Status,
  Documentation, Dashboard, GitHub — but **not** `/hub/`. And `sitemap.xml`
  contains exactly one URL (`https://spikeforge.net/`). So the hub page is
  neither linkable from the site nor advertised to search engines. The work
  shipped but nothing points at it.
- There is still no issue/PR template for a hub submission — `CONTRIBUTING.md`
  describes the path in prose, but `.github/ISSUE_TEMPLATE/` has only
  `bug_report.yml` and `feature_request.yml`.

**Fix shape:** this is the one item on this list that is a real feature, and it
should be scoped as one — an entry kind that carries a checkpoint, a
materialization path that loads it instead of rebuilding, schema and
`CURATION.md` updates, tests alongside the existing `test_hub_*` suite — then
seeded with a handful of honestly-labelled *reference* checkpoints (not SOTA
claims) for datasets and topologies already shipped. Even three real ones
changes what the hub *is*. Link `/hub/` from the landing page and add it to
`sitemap.xml` regardless, since that part is a five-minute fix.

---

## 4. 🟠 The landing page still sends people to `git clone`; only the README was fixed

Cycle 1 inverted the README's funnel to lead with `pip install spikeforge`.
The landing page at spikeforge.net — the surface that ranks in search, gets
shared, and carries the 17-language translation investment — was not changed to
match. Its hero terminal block still reads:

```
$ git clone https://github.com/capsize-games/spikeforge.git
$ cd spikeforge && ./install.sh
```

The string `pip install` does not appear anywhere on the page. Meanwhile the
page's own status table has a row reading **"PyPI release — done."** So the
landing page simultaneously advertises that the package is on PyPI and tells
you to clone a git repository and run a shell script to get it. For the
"developers who want to drop a dependency into an existing project" audience
named in the original brief, `curl | clone | ./install.sh` is the highest-
friction possible opening, and it is inconsistent with the README the same
visitor sees one click later on GitHub.

**Fix shape:** make the landing hero's primary command `pip install spikeforge`
followed by the same five-line snippet the README now leads with, with
clone/`install.sh` demoted to the contributor path. Note this touches the i18n
system (`landing/locales.js`, `landing/cta-locales.js`) — the copy change needs
to propagate to the translated strings, or the page will show English fallback
text in 17 locales.

---

## 5. 🟠 No published accuracy numbers anywhere — the academic audience has nothing to check

Searched `documentation/`, `README.md`, `COOKBOOK.md`, `plans/`, and the
landing page: there is **no results table, no baseline accuracy, no comparison
against published SNN numbers** for any dataset the project ships. The closest
thing is the number that scrolls past during a quickstart run.

This is a real barrier for the two audiences named in the brief. An academic
choosing what to build a thesis on, or a professional choosing what to depend
on, asks "what does this actually achieve on MNIST / N-MNIST / DVS128 Gesture,
in how long, on what hardware, and how does that compare to what's published?"
The project currently cannot answer that question with a link.

What makes this cheap rather than expensive: **the machinery already exists.**
`spikeforge-benchmark` has a full harness (fixtures, repeats, warmup, a
`BenchmarkStore`, `--compare`, `--fail-on-regression` as a CI gate). It measures
latency/throughput, not task accuracy — but the training and eval paths that
produce accuracy are equally shipped, and the datasets are already wired.

**Fix shape:** a single `documentation/benchmarks.md` (and a short README
section linking it) with one row per dataset × topology: test accuracy, epochs,
wall-clock, hardware, seed, and the exact command to reproduce it — plus the
honest caveat that these are reference configurations, not tuned SOTA attempts.
Generating it from the existing tooling and committing the results is the whole
job. This pairs naturally with item 3: the checkpoints you train to produce the
numbers are exactly the checkpoints the hub needs.

Adjacent and equally cheap, same audience: **`CITATION.cff` has no `version`,
no `date-released`, and no DOI.** GitHub's "Cite this repository" button
therefore emits a citation with no version, and there is no Zenodo archive, so
there is nothing stable for a paper to point at. Adding `version`,
`date-released`, and minting a Zenodo DOI (it auto-archives on GitHub release)
is a one-time setup that materially changes how citable this is.

---

## 6. 🟠 The project is absent from the two directories this field actually searches

This is the "pure promotion" part of your question, and it has a concrete,
high-leverage answer rather than a vague one. Two checks:

- **NIR's own framework support table** (<https://github.com/neuromorphs/NIR>)
  lists ten frameworks with read/write status: hxtorch, jaxsnn, Lava-DL, Nengo,
  Norse, Rockpool, Sinabs, snnTorch, SpiNNaker2, Spyx. **spikeforge is not on
  it** — despite supporting both directions *and* doing something none of the
  ten do (independent re-execution of the exported graph with a drift report).
  The README's entire positioning argument rests on the NIR spine, and the NIR
  project itself doesn't know spikeforge exists.
- **Open Neuromorphic's software guide**
  (<https://open-neuromorphic.org/neuromorphic-computing/software/>) catalogues
  27 SNN frameworks and data tools and is the de-facto directory for this
  field. **Neither "spikeforge" nor "Capsize" appears.** The page explicitly
  invites submissions: *"Suggest new frameworks, data tools, or corrections by
  opening an issue on our GitHub repository."*

Search discoverability for the *name* is fine — a search for "spikeforge
spiking neural network" returns the GitHub repo and all seven PyPI projects at
the top. The problem is entirely categorical discovery: someone searching "SNN
framework comparison" or browsing the NIR ecosystem never encounters it.

**Fix shape:** open a PR against NIR adding spikeforge to the support table, and
an issue against Open Neuromorphic's guide proposing the entry. Both are free,
both are explicitly invited, and both put the project in front of precisely the
audience it is built for. **Sequence these after items 1 and 2** — both
submissions will be evaluated by people who click straight through to the PyPI
page and the docs site.

---

## 7. 🟡 No `__version__`, no `--version`, and `compatibility.json` is still invisible at runtime

`import spikeforge; spikeforge.__version__` → **`AttributeError`** (verified in
the clean venv against the published 0.3.4 wheel). None of the seven
distributions expose a version attribute. No console script accepts
`--version` (`spikeforge --help` shows only `-h`).

This is a small thing that shows up constantly: it's the first line of every
bug report, the thing a paper's methods section needs, and the thing a user
checks before filing an issue. Every mainstream Python library has it.

It also leaves cycle-1 item §6 half-done. That report offered two options —
surface the compatibility matrix at runtime, or explain it in the README — and
cycle 1 chose the README. That was reasonable, but it means a user with a
genuinely mismatched combination (`spikeforge 0.3.4` + a future
`spikeforge-targets 0.2.0`) still gets no signal, and `compatibility.json`
remains a file you have to know to go read.

**Fix shape:** `__version__` on all seven roots (read from installed metadata
via `importlib.metadata`, so it can't drift from the wheel), plus a
`--version` flag that prints the core version and the installed satellite
versions, flagging any combination not present in `compatibility.json`.

---

## 8. 🟡 There is no API reference documentation at all

Verified: no mkdocstrings, no Sphinx autodoc, no pdoc, nothing, anywhere in the
repo. `mkdocs.yml`'s nav is 60-plus hand-written prose pages and zero generated
API pages.

Everything is narrative documentation. There is no page where a user can look
up `TrainingEngine`'s constructor signature, what `dataset=` accepts, what
`train()` yields, what the encoder classes expose, or what exceptions any of it
raises — short of reading the source. For a library whose audience includes
academics and professionals integrating it into their own code, this is a
significant, standard-expectation gap: snnTorch, Norse, Lava and PyTorch all
ship a generated API reference.

The groundwork is unusually good — the codebase has consistent docstrings and
ships `py.typed` — so this is mostly a matter of wiring
`mkdocstrings[python]` into the existing mkdocs build (which also gives the
currently-unpublished mkdocs site a reason to exist, tying into item 2's
"decide which site is canonical").

---

## 9. 🟡 The README's flagship quickstart bypasses the public API

The README's first code block — the one cycle 1 correctly made the headline —
is:

```python
from spikeforge.training.training_engine import TrainingEngine
```

But `spikeforge/__init__.py` exports six names, and `TrainingEngine` is not one
of them: `SNNTrainer`, `SNNTrainerLogger`, `LatencyTrainer`, `DeltaTrainer`,
`RandomSpikeGenerator`, `convert_to_time`.

So the single most prominent code sample in the project reaches four modules
deep into a path that reads like internals, while the package's advertised
public surface offers a *different* entry point (`SNNTrainer`) that the README
never mentions. A reader who types `import spikeforge` and hits tab-complete
does not find the thing the README just told them to use, and has no way to
tell which of the two is the intended one.

**Fix shape:** export `TrainingEngine` from `spikeforge/__init__.py`, add it to
`__all__`, and make the README snippet `from spikeforge import TrainingEngine`.
Then state, in one line in the docs, how `TrainingEngine` and `SNNTrainer`
relate and which one a new user should reach for.

---

## 10. 🟡 `pip install spikeforge` costs 5.5 GB, and the CPU-only path is documented only for Docker

Measured in the clean-room install: `pip install spikeforge` produces a
**5.5 GB** virtualenv in 90 seconds, because default `torch` wheels drag the
entire CUDA stack (`nvidia-cublas`, `nvidia-cudnn`, `nvidia-cusolver`,
`cuda-toolkit`, `triton`, …).

Most of that is inherent to PyTorch, not to spikeforge. What *is* actionable:
the project clearly knows the cheap path — CI sets
`TORCH_INDEX_URL: https://download.pytorch.org/whl/cpu`, and `documentation/usage.md`
and `COOKBOOK.md` both document CPU-only builds — but **only for the Docker
route**. The `pip install` path that the README now leads with never mentions
it. A user evaluating the library on a laptop, in a container, or in CI pays
5.5 GB to run a 13-second MNIST demo that never touches a GPU.

**Fix shape:** one line under the README quickstart —
`pip install spikeforge --index-url https://download.pytorch.org/whl/cpu` for
CPU-only — and the same in `documentation/quickstart.md`. Pure documentation.

---

## 11. 🟡 `py.typed` on 2 of 7 distributions, and no type-check gate

`spikeforge` and `spikeforge_io` ship `py.typed`. `spikeforge_hub`,
`spikeforge_targets`, `spikeforge_serve`, `spikeforge_clients` and `server` do
not — neither on disk nor declared in their `pyproject.toml` package data. So a
user running mypy or pyright against code that touches the deployment or hub
APIs — the differentiated surfaces — gets `module is installed, but missing
library stubs or py.typed marker` and every symbol silently degrades to `Any`.

Separately, there is **no type checker in CI at all** (verified across
`ci.yml`, `ruff.toml`, `pytest.ini` — ruff only). The core distribution ships a
`py.typed` marker, which is a promise to type checkers that nothing verifies.

**Fix shape:** add `py.typed` to the remaining roots and declare it in their
package data; add a mypy or pyright job to CI. Given the codebase is already
annotated throughout, this is likely a small diff plus whatever the first clean
run turns up.

---

## 12. 🟢 Minor / polish

- **The `spikeforge` command is still the tutorial demo.** Cycle 1 fixed the
  crash and gave it honest help text — which now reads, accurately, *"this is
  the tutorial demo, not a general-purpose training CLI -- see
  spikeforge-verify, spikeforge-benchmark, and documentation/usage.md for the
  rest of the toolkit."* That is honest, and it is also the command carrying
  the project's name telling every new user to go look somewhere else. Cycle 1
  flagged the consolidation as longer-term; it is still open. A single
  `spikeforge <subcommand>` dispatcher (with `demo`, `train`, `verify`,
  `benchmark`, `hub`, `--version`) would give one CLI design language instead
  of six sibling binaries. **Judgment call for whoever picks this up:** this is
  the one item here that is a refactor, and it is legitimately deferrable — but
  if item 7's `--version` work happens, that is the natural moment to decide.
- **`examples/` still doesn't ship in the wheel.** Confirmed: the 0.3.4 wheel
  contains 249 files, top-level `main.py`, `main_encodings.py`, `spikeforge/`,
  and the dist-info — no examples, no `.md` files. So a pip-only user still has
  nothing runnable to copy without fetching the repo separately. Cycle 1
  addressed the README half of §5 but not this half. Cheapest fix is
  documentation (say plainly where the examples live and how to get them);
  shipping two or three as `spikeforge.examples` is the fuller one.
- **`docs/` is confirmed git-ignored** — cycle 1's correction of the cycle-1
  "two doc trees" finding was right (`git ls-files docs/` → 0 files). No action
  needed beyond what item 2 covers.
- **Social proof is still 1 star, 0 forks, 0 external contributors.** As cycle 1
  said, this is earned rather than built. Items 1, 2, 5 and 6 are the levers.

---

## 13. Priority order

1. **Fix the PyPI storefront** (§1) — absolute links + working hero image +
   honest `description`/`keywords`/`urls` across all seven distributions, with a
   `readme_renderer` CI gate. Highest leverage-per-hour on this list; it is the
   front door for the audience cycle 1 aimed the README at.
2. **Fix the published docs site** (§2) — rewrite non-page links in
   `build_wiki.py` (or publish mkdocs and retire the wiki), re-point the link
   checker at whatever is actually published, and rewrite the docs homepage
   opening for readers rather than contributors. 896 dead links is the largest
   single defect found this pass.
3. **Landing page: lead with `pip install`** (§4), and link `/hub/` from the
   site + add it to `sitemap.xml` (§3). Small, and it removes a direct
   contradiction between two front doors.
4. **Publish benchmark numbers** (§5) and fill in `CITATION.cff`'s
   `version`/`date-released`/DOI. Unblocks the academic audience with tooling
   that already exists.
5. **Give the hub real weights** (§3). The largest piece of genuine engineering
   on this list, and the one you named first. Scope it properly; the
   checkpoints from step 4 feed straight into it.
6. **Then promote** (§6) — NIR support table, Open Neuromorphic software guide.
   Free, invited, and aimed exactly right — but only worth spending once 1–3
   are done, since both audiences will click straight through.
7. `__version__`/`--version` (§7), API reference (§8), `TrainingEngine` export
   (§9), CPU-only install line (§10), `py.typed` + type gate (§11), polish
   (§12) — each independently cheap, none blocking.

---

## 14. Explicitly out of scope — do not do these

Named because the brief was explicit about not adding or refactoring for its
own sake, and because a fresh session is prone to it:

- **Do not redo any cycle-1 fix.** The table at the top of this report lists
  what was verified working against the published wheel and the live sites.
- **Do not add new SNN capability.** No new neuron models, encoders,
  topologies, or targets. UC-2 through UC-10 (issues #13–#21) are correctly
  scoped as spec-only and should stay that way.
- **Do not restructure the package split.** The seven-distribution layout and
  ARCH-0001 are deliberate and working; `compatibility.json` is maintained.
  Item 7 asks only to *surface* it, not to change it.
- **Do not rewrite the documentation's voice wholesale.** Item 2 asks for the
  links to work and for one homepage paragraph to address readers. The
  design-record style of `plans/` is a genuine asset for this codebase.
- **Do not fabricate a Colab notebook or hosted demo.** Cycle 1 declined this
  for good reason, and note that `dash.spikeforge.net` already serves a live,
  read-only dashboard demo — the "try before installing" gap is more filled
  than cycle 1's §5 assumed. Link it more prominently rather than building
  something new.
- **Do not chase stars.** Item 6 is the legitimate version of that work.

---

## 15. What "done" looks like for this cycle

A reviewer should be able to verify, without re-deriving anything:

- The rendered PyPI README has **zero** relative `href`s and `src`s
  (`readme_renderer` + a CI gate proving it stays that way), and the dashboard
  image renders on <https://pypi.org/project/spikeforge/>.
- `python scripts/build_wiki.py <tmp>` followed by a link audit reports **zero**
  dead links, and a CI gate runs that audit.
- <https://spikeforge.net/> shows `pip install spikeforge` as the primary
  command, links `/hub/`, and `sitemap.xml` lists more than one URL.
- A documented benchmark page exists with reproducible commands, and
  `CITATION.cff` carries a version and a DOI.
- `python -c "import spikeforge; print(spikeforge.__version__)"` prints the
  installed version, and `spikeforge --version` works.
- Whatever is claimed as done was verified against a **clean install of the
  published artifact or the live site**, not against the working tree — that
  distinction is what this pass found cycle 1's remaining gaps hiding behind.

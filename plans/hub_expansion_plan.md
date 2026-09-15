# Model-hub expansion — architecture and implementation plan

**Audience:** the engineer or agent implementing this, in a fresh session, with
no memory of the conversation that produced it. Everything needed is either
here or named by file and line.

**Goal:** move the hub from six static-image reference checkpoints to a catalog
that demonstrates spiking networks on the workloads that justify them — event
vision first, then audio — without weakening a single honesty guarantee the
project already makes.

**Read first:** [`spikeforge_hub/CURATION.md`](https://github.com/Capsize-Games/spikeforge/blob/main/spikeforge_hub/CURATION.md) is
the policy this plan operates under and does not amend. Where this plan and
that document disagree, that document wins and this plan is wrong.

---

## 0. The strategic frame, so the work is not mis-scoped

The hub's differentiator is **not size**. It cannot out-scale Hugging Face and
must not try. What nothing else in this field offers is a catalog where every
entry is NIR-exportable, drift-validated, deployment-reported, energy-estimated,
checksum-pinned, and reproducible by a printed command. Ten entries where every
claim is checkable beat a thousand where none are.

So: **breadth across modalities, not across architectures.** The current six
entries are all MNIST-family static images — breadth on the axis that matters
least, and on the one workload where a reviewer knows a conventional network is
the better answer. One DVS128 Gesture checkpoint changes what the hub
demonstrates more than twenty more MNIST variants would.

Keep the reference-configuration philosophy exactly as
[`CURATION.md`](https://github.com/Capsize-Games/spikeforge/blob/main/spikeforge_hub/CURATION.md) states it: stock
hyperparameters, modest epochs, one seed, honest numbers, a reproduce command,
and explicitly not a state-of-the-art claim.

---

## 1. 🔴 Blocking discovery: event "test accuracy" is currently train accuracy

**This must be fixed before any event-dataset checkpoint or benchmark row is
produced. Nothing else in this plan is safe to start first.**

The event training path has no train/test split anywhere in it. The `train`
flag exists at the top of the chain, is threaded nowhere, and is silently
ignored at the bottom. Traced end to end:

| Layer | File | What it does |
|---|---|---|
| Dataset spec | [`spikeforge/data/datasets.py`](../spikeforge/data/datasets.py) | `kwargs` hardcodes `{"train": True}` for `n_mnist` and `dvs128_gesture`, `{"split": "train"}` for `ssc`. `cifar10_dvs` passes no split at all. |
| Tonic construction | [`spikeforge/data/event_loader.py`](../spikeforge/data/event_loader.py) `_dataset()` | `cls(_root(spec, save_to), **spec.kwargs)` — no split parameter exists |
| Sample source | [`spikeforge/events/event_source.py`](../spikeforge/events/event_source.py) `EventSampleSource.__init__` | no split parameter |
| Batch assembly | [`spikeforge/training/event_batches.py`](../spikeforge/training/event_batches.py) `event_batches()` | signature is `(source, spec, subset, batch_size)` — no split parameter |
| Engine | [`spikeforge/training/event_engine.py`](../spikeforge/training/event_engine.py) `_epoch_batches(self, train=True)` | **accepts `train` and never reads it** — always returns the same batches |

The consequence: `EventTrainingEngine._load_test_batches()`, whose docstring
says *"Cache a few bridged event batches for held-out scoring"*, returns the
first batches of the **training** stream. Every accuracy the event path can
currently report is train accuracy wearing a test label.

For a project whose entire positioning rests on numbers a reader can trust,
publishing an event benchmark on top of this would be the worst possible
unforced error. It is also a real bug independent of this plan — the dashboard
reports the same misleading number today.

### 1.1 Fix shape

Thread a split through all five layers. Two design decisions worth making
deliberately rather than discovering:

**(a) Split selection differs per dataset.** `NMNIST` and `DVSGesture` take
`train: bool`; `SSC` takes `split: "train"|"test"`; `CIFAR10DVS` has no split at
all (it is a single pool and needs a deterministic partition, or it must be
declared unsupported for held-out scoring — say which, do not quietly use all of
it). Do not paper this over with a boolean. Extend
[`DatasetSpec`](../spikeforge/data/dataset_spec.py) to carry explicit per-split
kwargs, e.g. a `splits: Dict[str, Dict[str, Any]]` mapping `"train"`/`"test"` to
the kwargs that select it, and make a missing `"test"` entry a typed, named
error rather than a silent fallback to train.

**(b) Two sources, not a threaded flag.** The image path already works by
holding separate loaders. Mirror it: `EventTrainingEngine` holds a train
`EventSampleSource` and a test `EventSampleSource`, and `_epoch_batches(train)`
picks between them. This keeps `event_batches()` unchanged and puts the split
where it belongs — in the source that knows which dataset it opened.

**(c) The synthetic backend needs a disjoint test stream.** `EventSampleSource`
falls back to a deterministic synthetic generator when tonic is absent, and the
test suite relies on it (`synthetic_only=True`). If both splits generate the
same samples, the regression test below passes vacuously. Give the synthetic
test stream a disjoint index range or a different seed, and state in the
docstring that synthetic "held-out" data is generated, not recorded.

### 1.2 Required regression test

A test that would have caught this, in `tests/` alongside the existing event
tests. It must assert the split is *real*, not merely that a flag is accepted:

- `_epoch_batches(train=True)` and `_epoch_batches(train=False)` return
  different tensors for the same engine and seed.
- `EventSampleSource(..., train=False)` opens a dataset whose length or content
  differs from `train=True` (under tonic), and a disjoint sample range (under
  synthetic).
- A dataset with no declared test split raises the typed error by name rather
  than returning training data.

### 1.3 Blast radius — already established, do not re-investigate

Checked while writing this plan, so it can be scoped calmly rather than
treated as an incident:

- **The public dashboard is not affected.** `tonic` is not installed in the
  deployed image (absent from both `requirements.txt` and the `Dockerfile`, and
  confirmed absent inside a built container), so `EventTrainingEngine._source()`
  raises `EventsExtraMissingError` rather than returning a number. The event
  path fails honestly there instead of reporting a wrong accuracy. The public
  demo is read-only on top of that.
- **Who is actually affected:** anyone who installs the `events` extra locally
  and trains on an event dataset. Real, but bounded — and nobody has published a
  number from it.

So this is a correctness bug to fix properly and in sequence, not a hotfix.

### 1.4 Honesty follow-through

`_load_test_batches`'s docstring currently states something untrue. Fix it.
Check whether any dashboard copy, `documentation/` page, or panel label presents
an event accuracy as held-out, and correct anything that does. Record the bug
and its blast radius in `CHANGELOG.md` under `Fixed` — the project's own
standard is that a wrong number gets named, not quietly corrected.

---

## 2. Phase 0 — dataset provenance in the catalog schema

Do this **before** producing new entries. Retrofitting attribution onto already
published entries is worse than shipping it with the first batch.

### 2.1 The gap

[`HubEntry`](../spikeforge_hub/entry.py) records `dataset` (a bare name) and
`license` — and that `license` describes **the weights**, not the data they
encode. Nothing records what the training data permits.

That is already live: `reference/kmnist-fc-legacy` ships marked
`"license": "BSD-3-Clause"`, but KMNIST is **CC BY-SA 4.0 with a requested
attribution** which appears nowhere in the artifact. N-MNIST — the obvious next
event target — is **CC BY-SA 4.0** as well.

Whether trained weights constitute "adapted material" under a ShareAlike licence
is genuinely unsettled; the prevailing ML norm says they do not, and Creative
Commons themselves state their licences are not designed to govern model
weights. **This plan does not take a position on that question.** It removes the
need to have one: record the provenance, carry the attribution, let a reader
judge. That is the same move the project already makes everywhere else.

### 2.2 Verified dataset licences

Each of these was checked against the dataset's own publisher during the
session that wrote this plan — not carried over from memory. Re-verify rather
than trusting this table: the authority is the publisher's page, and a licence
can change.

| Dataset | Licence | Attribution |
|---|---|---|
| DVS128 Gesture | CC BY 4.0 | required |
| SHD / SSC | CC BY 4.0 | required |
| N-MNIST | CC BY-SA 4.0 | required, ShareAlike |
| KMNIST | CC BY-SA 4.0 | required; CODH requests a specific wording |
| Fashion-MNIST | MIT | required |

**Not yet verified — the implementer must confirm and must not guess:** MNIST
itself (no explicit licence on the original distribution; commonly cited as
CC BY-SA 3.0) and CIFAR10-DVS. If a licence cannot be confirmed from a primary
source, record it with the project's existing `unverified-candidate` discipline
rather than inventing one — that marker exists precisely for this.

### 2.3 Schema change

In [`spikeforge_hub/entry.py`](../spikeforge_hub/entry.py):

- Add `dataset_license` and `dataset_attribution` to `_FIELDS` and the
  `HubEntry` dataclass.
- In `_check_reference()`, require both for `source: "reference"`, exactly as
  `weights` / `sha256` / `dataset` / `test_accuracy` are required today. The
  rationale is identical: a trained entry that cannot say what its data permits
  is not one the project can stand behind.
- Validate `dataset_license` with the **same** rule as `license` — reuse
  `_check_license`'s concrete-SPDX-or-marker logic rather than writing a second,
  looser one. Free-text escapes must be rejected in both fields.

Then:

- Extend `_catalog_entry()` in
  [`scripts/train_reference_models.py`](../scripts/train_reference_models.py) to
  emit both fields, sourced from a single provenance table — put it next to the
  dataset registry so it cannot drift from the datasets it describes.
- Surface both on the public page in
  [`scripts/build_hub_page.py`](../scripts/build_hub_page.py) (a Data column, or
  a line under the notes). The page is HTML-escaped already; keep it that way.
- Backfill all six existing entries via `--publish`, not by hand.
- Add the dataset attributions to [`NOTICE.md`](https://github.com/Capsize-Games/spikeforge/blob/main/NOTICE.md).

### 2.4 Tests

Extend `tests/test_hub_reference_weights.py`: a reference entry missing either
new field is rejected by name; a free-text dataset licence is rejected; the
shipped catalog has both fields populated on every reference entry.

---

## 3. Phase 1 — DVS128 Gesture reference checkpoint

The entry that changes what the hub demonstrates. Chosen over N-MNIST
deliberately: CC BY 4.0 rather than ShareAlike, a genuinely event-native task
rather than a re-recording of MNIST, and the canonical "why spikes" example.

### 3.1 What already works — do not rebuild it

- `conv_net` **already accepts the needed geometry**. See
  [`spikeforge/topology/presets.py`](../spikeforge/topology/presets.py):
  `conv_net(in_channels=1, channels=8, num_classes=10, input_size=28, ...)`, and
  `input_size` accepts an `int` square side or an explicit `(H, W)` pair. No new
  preset is required.
- `Reference` in
  [`scripts/train_reference_models.py`](../scripts/train_reference_models.py)
  already carries a `topology_params` field that reaches the builder.
- `EventTrainingEngine` already bridges events, validates sensor geometry, and
  reuses the shared loop, metrics and checkpointing.

So the configuration is:

```python
Reference(
    name="dvs128-gesture-conv-net",
    dataset="dvs128_gesture",
    topology="conv_net",
    topology_params={"in_channels": 2, "num_classes": 11, "input_size": 128},
    ...
)
```

Read `num_classes` and the sensor geometry from the dataset registry rather than
hardcoding them in two places; `DatasetSpec` already carries the class count.

### 3.2 What must change in the training script

Two concrete gaps, both in
[`scripts/train_reference_models.py`](../scripts/train_reference_models.py):

1. **Engine selection.** `_train()` does `from spikeforge import TrainingEngine`
   and instantiates it directly. Event datasets need `EventTrainingEngine`.
   Branch on the registry's `modality`, not on the dataset name.

2. **Full-split scoring.** `_full_test_accuracy()` calls `build_loader(...)`,
   and [`build_dataset`](../spikeforge/data/datasets.py) **raises by design** for
   event modality (*"train it through the event batch path instead of
   build_dataset"*). An event branch is required: score over the complete
   held-out set from the test source — every batch, not the `EVAL_BATCHES`
   truncation `_load_test_batches()` applies. The existing helper's contract
   ("the complete held-out split, not a sample of it") must hold identically for
   events, or the number is not comparable to the image rows.

Note that this scoring path is only meaningful **after** §1 lands. Implementing
it first against the unsplit source produces a confidently wrong number.

### 3.3 Cost and size — measure before committing

- **Wall clock.** Event samples are bridged per-sample, not read from a tensor
  file; expect this to be materially slower per epoch than the 249 s MNIST
  `conv_net` row. `_shrink_progress_evaluation()` already exists for exactly
  this reason and matters more here. **Time a small-subset run first and report
  the projection before launching a full one.** If a CPU run is implausible,
  stop and ask rather than quietly reducing the training split — `subset` is a
  divisor, and shrinking it changes what the published number means.
- **Checkpoint size.** `conv_net` at 128×128 pools to 32×32 with 16 channels →
  16 384 flattened features → ~180 k readout parameters, on the order of 1 MB.
  That likely fits `CURATION.md`'s "few megabytes" wheel budget, but **measure
  it**; the six current checkpoints total 1.6 MB.
- **If it does not fit**, `CURATION.md` already prescribes the answer:
  `"source": "url"` with a checksum, downloaded on demand. Be aware that **no
  catalog entry uses that path today** — `tests/test_hub_download_routing.py`
  exercises routing, but nothing real has ever been fetched through it. Treat a
  first use as a feature to verify end to end (download, checksum verify,
  cancel, cache reuse, corrupted-file rejection), not as a path you can assume
  works.

### 3.4 Outputs

`documentation/benchmarks.md` gains a row; the catalog gains an entry; the
README's "What it scores" table gains a row. All three are generated by
`--publish` — do not hand-edit any of them. Update the README's framing so the
table is no longer implicitly all-MNIST.

---

## 4. Phase 2 — an audio checkpoint (SSC or SHD)

Only after Phase 1 lands and is reviewed. Gives three modalities: static vision,
event vision, audio.

Higher risk, and the reasons are known in advance:

- **Split shape differs.** SSC uses `split: "train"|"test"`, not a boolean. If
  §1.1(a) was implemented as a boolean, this phase will require reworking it —
  which is why §1.1(a) specifies explicit per-split kwargs.
- **Geometry is not spatial.** SSC is 700 channels, 35 classes. The engine's
  `_check_features()` requires `input_size == height * width` against tonic's
  `sensor_size`. `fc_legacy` with `input_size=700` is the likely fit, but the
  implementer must confirm what tonic reports for SSC's `sensor_size` before
  assuming — `_shape()` in `event_loader.py` unpacks it as
  `(width, height, channels)`.
- SHD is the smaller, cleaner sibling and is a legitimate substitute if SSC
  proves awkward. Both are CC BY 4.0.

---

## 5. Explicitly out of scope

- **No state-of-the-art claims.** Not in the entry, the notes, the page, the
  README, or a commit message. The published framing is "a floor the shipped
  defaults reach, not a ceiling" and it stays.
- **No more MNIST-family variants.** More rows on the existing axis is the
  specific thing this plan exists to stop.
- **No relabelling `bundled` as trained.** `CURATION.md` names this as the
  fabrication the policy exists to prevent.
- **No softening of the spec-only target status.** `spinnaker2`, `speck` and
  `xylo` remain registered-but-uninstallable, and the energy figure remains an
  estimate.
- **No third-party weights redistributed.** Reference entries are this project's
  own artifacts.
- **No new SNN capability.** No new neuron models, encoders or targets; this is
  a data, provenance and correctness plan.

---

## 6. Sequencing

Strictly ordered. Each step is reviewable on its own.

1. **§1 — event train/test split**, with the regression test and the changelog
   entry. Nothing downstream is trustworthy without it.
2. **§2 — provenance schema**, with backfill of the existing six.
3. **§3 — DVS128 Gesture**, timing probe reported before the full run.
4. **§4 — audio**, only after 3 is reviewed.

Steps 1 and 2 are independent of each other and may be done in either order, but
both precede step 3.

---

## 7. Acceptance criteria

A reviewer should be able to check each of these without re-deriving anything:

- `EventTrainingEngine._epoch_batches(train=True)` and `(train=False)` return
  demonstrably different data, proven by a test that fails if the split is
  removed.
- A dataset with no declared test split raises a typed, named error instead of
  returning training data.
- Every `source: "reference"` entry carries a concrete `dataset_license` and a
  `dataset_attribution`; validation rejects an entry missing either, and rejects
  free-text in the licence field.
- The six existing entries are backfilled, and the attributions appear in
  `NOTICE.md` and on <https://spikeforge.net/hub/>.
- A DVS128 Gesture entry exists whose accuracy was measured on the **complete**
  held-out split, with the reproduce command in its notes, and whose `sha256`
  and `size_bytes` match the shipped bytes (re-verify by hashing the packaged
  file, not by trusting `--publish`).
- `documentation/benchmarks.md`, the README table and the catalog agree on every
  number, because all three were generated.
- Full suite green, `ruff` clean, `mypy` clean, `scripts/check_wiki_links.py`
  and `scripts/check_readme_links.py` clean, packaging guards clean.
- Any claim of "verified" is backed by a command in the work report that a
  reviewer can re-run.

---

## 8. Questions for the maintainer — ask, do not assume

1. **CPU or GPU for the event run?** If the timing probe in §3.3 projects an
   implausible CPU run, the options are a GPU run (which changes the
   hardware line in the benchmark table and makes the number less reproducible
   for a reader on a laptop) or a smaller configuration. That is a product
   decision, not an implementation one.
2. **MNIST's licence.** If it cannot be confirmed from a primary source, confirm
   that recording it as an unverified candidate is acceptable rather than
   asserting CC BY-SA 3.0 from secondary sources.
3. **Release batching.** `spikeforge-hub` went to 0.2.0 carrying weights. Should
   each phase cut a release, or should they batch into one? Note the tag-push
   caveat in [`ecosystem_listings.md`](ecosystem_listings.md)'s sibling history:
   push release tags **one at a time**, since GitHub suppresses workflow events
   when several tags arrive in one push.
4. **CIFAR10-DVS** has no upstream split. Deterministic partition, or declare it
   unsupported for published accuracy?

# Hub catalog curation policy

The shipped catalog [`models.json`](models.json) is a **curated allow-list**, not
a directory of everything that exists. Every entry must be something the project
can stand behind: a real source and a license that has actually been read. An
invented repository id or an unnamed license is a fabrication and must never be
committed here.

The record schema is enforced by
[`hub/entry.py`](entry.py:1) and validated on load by
[`hub/catalog.py`](catalog.py:1); anything that fails validation is reported
through `catalog.issues()` and dropped from `entries()` rather than silently
accepted.

## Adding a bundled entry

A bundled artifact is authored by this project (for example a NIR graph rendered
from a shipped topology preset). To add one:

1. Add an entry object to [`models.json`](models.json) with `"source":
   "bundled"`.
2. Set the concrete SPDX id for **your** artifact (`"license":
   "BSD-3-Clause"` for this project's own presets).
3. Set `"topology"` to the shipped preset the graph is rendered from, and
   `"input_shape"` to its expected input.
4. Describe the provenance in `"notes"` so a reader can tell why it is bundled.

## Adding a reference entry (trained weights)

A reference entry (`"source": "reference"`) is the one kind that carries
**trained** weights: a checkpoint this project trained itself, shipped inside
the `spikeforge-hub` distribution under `weights/` and loaded — never rebuilt —
by [`inspect.reference_path()`](inspect.py:82).

Do not hand-write these entries. They are generated, together with the
checkpoint they describe, by:

```bash
python scripts/train_reference_models.py --only <name> --publish
```

That script trains the configuration, scores it on the **complete** held-out
test split, copies the checkpoint into `spikeforge_hub/weights/`, and rewrites
the entry so its `sha256`, `size_bytes`, `test_accuracy`, and `test_samples`
describe the bytes that actually shipped. Those four fields are worse than
useless when they drift, which is why they are never edited by hand.

Validation enforces the honesty rule for this source: a reference entry must
name its `weights` file, pin a `sha256`, say which `dataset` it trained on, and
report a `test_accuracy`. An entry claiming trained weights that cannot say
what it scores is rejected on load.

**These are reference configurations, not state-of-the-art claims.** Stock
hyperparameters, modest epoch counts, one seed, CPU. Each entry's `notes` names
the exact command that reproduces it, and
[`documentation/benchmarks.md`](../documentation/benchmarks.md) carries the full
table. Label them that way in any copy that mentions them.

**Size budget.** These checkpoints ship in the wheel because the hub is
offline-first, so the budget is deliberately small: keep the total under a few
megabytes and prefer small topologies. Anything larger belongs behind
`"source": "url"` with a checksum, downloaded on demand.

## Adding a remote entry

A remote entry (`"source": "url"` or `"source": "hf_repo"`) points at bytes
somewhere else. To add one **you must verify the source first**:

1. **Name a real repository or reference.** Use the exact, resolvable locator —
   the full `url`, or the real `org/repo` id for `hf_repo`. Do not invent a
   namespace or a "seed" repository id. If you cannot point at a real upstream,
   there is no entry to add.
2. **State a verified license.** Read the upstream license and put its concrete
   SPDX-style id in `"license"` (for example `"Apache-2.0"`,
   `"BSD-3-Clause"`, `"MIT"`). A compound term such as `"MIT OR Apache-2.0"` is
   accepted. Free-text escapes — `"see upstream"`, `"unknown"`, `"TBD"` — are
   rejected by validation.
3. **Publish a checksum where one is available.** Prefer `"sha256"` (and
   `"size_bytes"` when known) so a download can verify as `verified`. When the
   source publishes no checksum, it is honest to omit it — the download then
   reports as `unverified`, never as verified.

## Unverified candidates must not be presented as ready

If a candidate is known-but-not-yet-verified, it may be recorded **only** with
the explicit marker:

```json
"license": "unverified-candidate"
```

That marker disables the entry: [`catalog.availability()`](catalog.py:106)
reports it `available: false` with the named reason *"unverified candidate:
upstream repository and license are not confirmed …"*, and it must never be
described as ready to use. Candidates are a record of work to do, not a
recommendation. When the real source and license are confirmed, replace the
marker with the concrete SPDX id (and add a checksum where available); if the
source cannot be verified, delete the entry.

An `hf_repo` or `url` entry that omits a real locator, or whose `"license"` is
free-text rather than a concrete id or the marker, fails validation and appears
in `catalog.issues()` instead of in `entries()`.

## Why the shipped catalog is small

The catalog ships only artifacts the project can stand behind. The Hugging Face
ingestion capability stays fully available — [`probe.py`](probe.py:1),
[`hf_api.py`](hf_api.py:1), [`download_cli.py`](download_cli.py:1), and the
`hub` extra let a user download any **vetted** repository they choose to add.
The catalog itself is not the downloader; it is the list of things already
checked.

The catalog now holds two different things, and the `source` field is what
tells them apart:

- `"source": "bundled"` — a NIR graph rendered on demand from one of this
  project's own topology presets. **Structure, with freshly-initialised
  weights.** Useful for comparing an exported graph against a known-good shape;
  useless as a model to run. Every such entry's `notes` says so.
- `"source": "reference"` — a checkpoint this project trained, shipped as bytes,
  with the accuracy it scores on the complete held-out split recorded in the
  entry itself.

The distinction is deliberate and must stay visible in the UI and the copy: a
catalog of untrained shapes and a catalog of trained models are different
products, and relabelling the former as the latter is exactly the fabrication
this policy exists to prevent.

Third-party trained weights are still not redistributed here. The reference
entries are this project's own artifacts, under this project's own license.

See also [`NOTICE.md`](../NOTICE.md) for the metadata-only weights policy.

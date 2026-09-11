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

See also [`NOTICE.md`](../../../NOTICE.md) for the metadata-only weights policy.

# Model hub (WS-A)

The hub discovers and obtains SNN models across the landscape and funnels
every artifact through an honest compatibility gate.

### Curated catalog + optional live Hugging Face

[`spikeforge_hub/models.json`](../spikeforge_hub/models.json) is the curated
catalog, and it holds **two different kinds of thing**. The `source` field is
what tells them apart, and the distinction matters:

- **`"source": "reference"` — trained weights.** Checkpoints this project
  trained itself, shipped inside the `spikeforge-hub` wheel under `weights/`
  and loaded (never rebuilt) by
  [`inspect.reference_path()`](../spikeforge_hub/inspect.py:82), verified
  against the checksum the catalog pins. Each entry records the dataset it
  trained on, what it scores on that dataset's **complete** held-out split,
  and — separately from its own `license` — that dataset's `dataset_license`
  and `dataset_attribution`. These are reference configurations with stock
  hyperparameters, not tuned attempts at state of the art —
  [Benchmarks](benchmarks.md) has the full table and the command that
  reproduces each row.
- **`"source": "bundled"` — structure only.** A NIR graph rendered on demand
  from one of this project's own topology presets, with freshly-initialised
  weights. Useful for checking an exported graph against a known-good shape;
  not a model to run. Every such entry's `notes` says so.

Regenerate the trained entries — checkpoint, checksum, size, and accuracy
together — with:

```bash
python scripts/train_reference_models.py --publish
```

Never hand-edit those four fields: they are worth nothing once they drift from
the bytes that shipped, and validation rejects a reference entry that cannot
name its weights file, pin a checksum, say which dataset it trained on, report
an accuracy, and declare that dataset's own licence and attribution.

### The weights' licence is not the data's licence

An entry's `license` covers the **weights**: this project's own artifact,
`BSD-3-Clause`. `dataset_license` covers the data those weights encode, which
has a different holder and different terms — KMNIST is CC BY-SA 4.0 with a
specific wording its publisher asks for, carried verbatim in
`dataset_attribution`. Both fields are held to the same rule: a concrete
SPDX-style id or the explicit `unverified-candidate` marker, with free text
rejected. Unlike `license`, `dataset_license` does not gate availability —
what the training data permits is disclosure for a reader to judge, not a
claim about whether the shipped weights load.

Whether trained weights are "adapted material" under a ShareAlike licence is
genuinely unsettled, and this project takes no position on it. Recording the
provenance removes the need to have one.

Both fields are looked up by dataset name from
[`dataset_provenance.py`](../spikeforge/data/dataset_provenance.py), which sits
beside the dataset registry so the two cannot drift. A licence recorded there
as `unverified-candidate` is one that could **not** be read from the
publisher's own page — MNIST and CIFAR10-DVS are both in that state — and is
deliberately not filled in from secondary sources. When one is later
confirmed, refresh the catalog without retraining:

```bash
python scripts/train_reference_models.py --sync-provenance
```

The catalog renders fully offline. Entries are validated into a
[`HubEntry`](../spikeforge_hub/entry.py:1); a malformed entry is *reported*
in `issues()` rather than silently skipped. The catalog ships **only verified
entries** — a remote entry must name a real repository/reference and a concrete
SPDX-style license, and a known-but-unverified candidate is marked
`"unverified-candidate"` and reported `available: false`. The full policy is in
`spikeforge_hub/CURATION.md`, which also documents how to propose a new
entry. `scripts/build_hub_page.py` renders the same catalog to a static,
publicly browsable page (deployed alongside the landing site) so it is
discoverable without installing anything.

Live Hugging Face search/download is provided by the **`spikeforge-hub`
distribution** (`packages/spikeforge-hub`, import root `spikeforge_hub`; ARCH-0001 Phase 4),
whose `huggingface_hub` dependency is isolated in
[`spikeforge_hub/hf_api.py`](../spikeforge_hub/hf_api.py:1) and
[`spikeforge_hub/probe.py`](../spikeforge_hub/probe.py:1). When `huggingface_hub` is absent,
`search` returns `available: false` with an explicit reason — never an error
and never a fabricated hit. There is no legacy core-relative hub import path:
the extraction shipped without a shim, so importers use `spikeforge_hub`
directly.

### Downloading

Downloads reuse the isolated child-process worker pattern so the FastAPI loop
never blocks: [`spikeforge_hub/download_cli.py`](../spikeforge_hub/download_cli.py:1)
fetches one entry into the offline cache and
[`spikeforge_hub/verify.py`](../spikeforge_hub/verify.py:1) checks its sha256 and size.
[`spikeforge_hub/downloads.py`](../spikeforge_hub/downloads.py:1) streams progress and
supports cancellation, exactly like the dataset downloader. The cache lives
under `HUB_CACHE_DIR` (`SPIKEFORGE_HUB_DIR`, default `<DATA_DIR>/hub`), kept separate
from the trained-model store. A source that publishes no checksum is reported
**unverified**, not passed silently.

### Inspect → compat → promote

[`spikeforge_hub/import_model.py`](../spikeforge_hub/import_model.py:1) runs a
three-gate funnel:

1. **Inspect** ([`spikeforge_hub/inspect.py`](../spikeforge_hub/inspect.py:1)) detects
   the artifact kind (`nir_graph`, `state_dict`, `framework_weights`) and
   describes its structure. Resolution depends on the source: a `bundled`
   entry is rendered from its preset, a `reference` entry is loaded from the
   packaged checkpoint, and a remote entry is read from the download cache.
2. **Compat** ([`spikeforge_hub/compat.py`](../spikeforge_hub/compat.py:1)) returns a
   verdict — `exact`, `mappable` (with a stage mapping), or `incompatible`
   (with the specific mismatches named).
3. **Promote** loads weights via
   [`spikeforge_hub/weight_map.py`](../spikeforge_hub/weight_map.py:1), runs a drift
   check, and only then saves into `MODEL_DIR` with hub provenance in `meta`.

A NIR-only artifact that matches no preset is still runnable through the
reference interpreter, so import is useful even without a weight mapping.

### `spikeforge-hub` CLI

```bash
spikeforge-hub list [--framework nir] [--kind nir_graph] [--available]
spikeforge-hub search <query> [--limit 20]
spikeforge-hub download <id> [--no-verify]
spikeforge-hub inspect <id>
spikeforge-hub import <id> [--topology conv_net]
```

Every command prints JSON; `download` and `import` exit non-zero on a failed
verification or an `incompatible` verdict, so they double as CI gates.

### WebSocket + client

Six additive actions: `hub_list`, `hub_search`, `hub_download`, `hub_cancel`,
`hub_inspect`, `hub_import` (replies `hub_list`, `hub_search`,
`hub_download_state`, `hub_inspect`, `hub_import`). The
[`HubPanel`](../client/src/components/HubPanel.tsx:1) browser renders entry cards,
a compat badge, a download progress row with cancel, and an inline import
verdict ([`HubVerdictView`](../client/src/components/HubVerdictView.tsx:1)) that
names every mismatch.

# NOTICE

## License

`spikeforge` is distributed under the **BSD 3-Clause License**; see
[`LICENSE`](LICENSE). The copyright is held by **Capsize LLC
<contact@capsizegames.com>**, the contributor listed in [`AUTHORS`](AUTHORS).

## Third-party models: metadata-only, fetched on demand

This repository does **not** redistribute third-party model weights, and that
has not changed. The model hub's catalog
([`spikeforge_hub/models.json`](spikeforge_hub/models.json)) ships two kinds of
entry, both authored here and both BSD-3-Clause:

- `"source": "bundled"` — a NIR graph derived from this project's own topology
  presets. Structure, with freshly-initialised weights.
- `"source": "reference"` — a **trained checkpoint produced by this project**,
  shipped inside the `spikeforge-hub` distribution under `weights/` and
  checksum-verified on load. These are this project's own artifacts, trained on
  publicly available datasets; they are not derived from, and do not
  redistribute, anyone else's weights.

The catalog ships **only verified entries**. Third-party weights a user chooses
to add are fetched on demand from their source into a local cache using the
Hugging Face ingestion mechanism
([`hub/hf_api.py`](spikeforge_hub/hf_api.py),
[`hub/download_cli.py`](spikeforge_hub/download_cli.py)) plus the `hub`
extra.

**Consequences for a redistributor and for users:**

- You are responsible for honouring each model's upstream license. The
  catalog's `license` field is the authoritative pointer and is now always a
  concrete SPDX-style id; the schema rejects free-text escapes such as
  `"see upstream"`.
- The catalog does not publish a checksum for every entry. An entry with no
  published checksum verifies as `unverified` rather than trusted, and integrity
  checking is best-effort.
- The only weights bundled here are this project's own reference checkpoints,
  released under the same BSD-3-Clause terms as the rest of the repository.
  This NOTICE grants no right to any third-party model a user chooses to add
  or download.
- The datasets those reference checkpoints were trained on (MNIST,
  Fashion-MNIST, Kuzushiji-MNIST) carry their own upstream terms, which are
  unaffected by this project's license and are not redistributed here.

### Curation

Adding an entry requires a **real** repository or reference plus a **verified**
license, and a checksum where one is available. A known-but-unverified
candidate must be marked `"license": "unverified-candidate"`, which loads the
record but reports it `available: false` with a named reason and never presents
it as ready. The full policy is in
[`spikeforge_hub/CURATION.md`](spikeforge_hub/CURATION.md). The hub
ships only verified entries plus an on-demand downloader for vetted repositories
the user chooses to add.

An earlier revision of this catalog carried four invented "seed" repository ids
under a fictional `snn-community/*` namespace with `"license": "see upstream"`.
They had no real upstream and could not be verified, so they were **removed**
rather than shipped; see [`CHANGELOG.md`](CHANGELOG.md).

## Third-party software dependencies

Runtime and optional dependencies (PyTorch, snnTorch, NIR, Tonic, ONNX,
`huggingface_hub`, Norse, Lava, TensorBoard, Weights & Biases, MkDocs Material,
and their transitive dependencies) are each governed by their own licenses.
Review them before redistributing a bundle that includes them.

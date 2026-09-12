# NOTICE

## License

`spikeforge` is distributed under the **BSD 3-Clause License**; see
[`LICENSE`](LICENSE). The copyright is held by **Capsize LLC
<contact@capsizegames.com>**, the contributor listed in [`AUTHORS`](AUTHORS).

## Third-party models: metadata-only, fetched on demand

The model hub ships a **curated catalog of metadata only**
([`spikeforge_hub/models.json`](spikeforge_hub/models.json)). This
repository does **not** redistribute third-party model weights. Every shipped
entry is a NIR graph derived from this project's own presets (BSD-3-Clause,
authored here); the catalog ships **only verified entries**. Weights a user
chooses to add are fetched on demand from their source into a local cache using
the Hugging Face ingestion mechanism
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
- No weights are bundled, so this NOTICE does not grant any right to any
  third-party model a user chooses to add or download.

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

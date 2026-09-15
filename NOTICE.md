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
- The datasets those reference checkpoints were trained on carry their own
  upstream terms, which are unaffected by this project's license and are not
  redistributed here. Each entry now records them in its own fields —
  `dataset_license` and `dataset_attribution`, separate from the `license`
  that covers the weights — and they are reproduced under
  [Training-data attribution](#training-data-attribution) below.

## Training-data attribution

A trained checkpoint's `license` describes **the weights**: this project's own
artifact, BSD-3-Clause. It says nothing about the data those weights encode.
Whether trained weights are "adapted material" under a ShareAlike licence is
genuinely unsettled — the prevailing ML norm says they are not, and Creative
Commons state their licences are not designed to govern model weights. This
project takes no position on that question. It records the provenance instead,
carries the credit each publisher asks for, and leaves the judgement to the
reader.

The table below is generated from the same source as the catalog's fields —
[`spikeforge/data/dataset_provenance.py`](spikeforge/data/dataset_provenance.py)
— so the two cannot disagree. A licence recorded as `unverified-candidate` is
one this project could **not** confirm from the publisher's own page; it is
deliberately not filled in from secondary sources.

| Dataset | Upstream licence | Credit |
|---|---|---|
| MNIST | `unverified-candidate` | The MNIST database of handwritten digits, Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. |
| Fashion-MNIST | MIT | Fashion-MNIST, Copyright (c) 2017 Zalando SE (https://tech.zalando.com), MIT licensed. |
| Kuzushiji-MNIST | CC BY-SA 4.0 | "KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341 |

The wording for Kuzushiji-MNIST is the attribution CODH asks for and is
reproduced verbatim.

Datasets the registry can load but no shipped checkpoint was trained on carry
their provenance in the same table. Confirmed from the publisher's own page:
N-MNIST (CC BY-SA 4.0, Orchard et al. 2015), DVS128 Gesture (CC BY 4.0, Amir
et al., CVPR 2017), Spiking Speech Commands (CC BY 4.0, Cramer et al., IEEE
TNNLS 2022), and QMNIST (BSD-3-Clause, Copyright (c) Facebook, Inc.).

The remaining entries carry `unverified-candidate`, and the table's `source`
field distinguishes two quite different reasons for it:

- **Checked; the publisher states no licence.** USPS, EMNIST, and CIFAR-10
  were each read at their own distribution point, and none declares a licence
  or terms of use — they ask only to be cited. The marker records that
  verified absence, which a guessed SPDX id would misrepresent.
- **Not checked, because the source will not serve.** MNIST
  (`yann.lecun.com` refuses connections) and CIFAR10-DVS (figshare answers
  403/202 with no body). The licences in circulation for both are secondary,
  so neither is asserted.

**Verify before publishing any checkpoint trained on one of them.**

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

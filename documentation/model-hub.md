# Model hub (WS-A)

The hub discovers and obtains SNN models across the landscape and funnels
every artifact through an honest compatibility gate.

### Curated catalog + optional live Hugging Face

[`spikeforge_hub/models.json`](../spikeforge_hub/models.json) bundles
**10 curated entries across five frameworks** (NIR, snnTorch, SpikingJelly,
Norse, Lava): ten NIR graphs rendered from this project's own presets. It
renders fully offline. Entries are validated into a
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
   describes its structure.
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

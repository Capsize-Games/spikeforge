# Spikeforge — WS-A: Model Hub and Import

> Focused design for workstream **A** of the
> [`professional_roadmap.md`](plans/professional_roadmap.md). Read the roadmap
> first for the vision, cross-workstream interfaces, and the packaging rules.

Design/spec only. Every claim about the current code is grounded in a file/line
reference so a Code-mode agent can execute this file-by-file.

**Objective.** Make the tool able to *find, download, inspect, and import* SNN
models from the wider neuromorphic landscape — in the dashboard, the CLI, and
the WebSocket API — with honest availability, reuse of the existing isolated
download worker, checksum verification, and an offline cache. A downloaded
model is either mapped to a shipped preset (structure + weights) or explicitly
rejected; it is never silently loaded wrong.

---

## 1. Approach decision

**Bundled curated catalog first, optional live Hugging Face Hub second.**

- A local JSON catalog (`spikeforge_hub/models.json`) enumerates known
  models across snnTorch, NIR, SpikingJelly, Norse, and Lava. It renders fully
  offline and ships only **verified** entries; the Hugging Face ingestion path
  (`hub/hf_api.py`, `hub/download_cli.py`, and the `hub` extra) downloads any
  vetted repo a user chooses to add.
- Live Hugging Face search/download is opt-in behind the `hub` extra
  (`huggingface_hub`), isolated in one module so its absence is reported, not
  raised. This mirrors the `tonic`/`events` precedent
  ([`datasets.py`](spikeforge/data/datasets.py:51)) and the isolated SDK
  probe precedent ([`probe.py`](spikeforge_targets/probe.py:14)).

Why not live-search-only: network dependence, no offline story, and no way to
keep the honesty rule (an uncurated hit is not a verified, runnable SNN). Why
not catalog-only: it forecloses the broader landscape the user explicitly
wants to reach. The hybrid keeps the default deterministic and the reach
open-ended.

Scope boundary: this workstream handles **model artifacts** (NIR graphs, SNN
checkpoints, framework weights in declared formats) and **metadata**. It does
not train downloaded models, and it does not claim any model runs until
compatibility is verified (Phase A3).

---

## 2. Current state and gap analysis

| Capability the goal requires | Current reality | Anchor |
|---|---|---|
| Model browser | None; only dataset pickers | [`datasets.py`](spikeforge/data/datasets.py:143) |
| Model download | None; no download path for weights | [`model_store.py`](spikeforge/network/model_store.py:34) |
| Isolated download worker | Exists for datasets, reusable pattern | [`download_cli.py`](spikeforge/data/download_cli.py:15), [`downloads.py`](server/downloads.py:41) |
| Progress + cancel UI | Exists for datasets | [`DownloadProgress.tsx`](client/src/components/DownloadProgress.tsx) |
| Checksum/size verification | None | n/a |
| Offline cache dir | Only `DATA_DIR` / `MODEL_DIR` | [`config.py`](spikeforge/config.py:10) |
| External NIR import | Partial, file-path only | [`ingest.py`](spikeforge/nir_bridge/ingest.py:21), [`serialization.py`](spikeforge/nir_bridge/serialization.py) |
| Weight import into a preset | None | [`model_store.py`](spikeforge/network/model_store.py:57) |
| Compatibility validation | None for foreign artifacts | [`registry.py`](spikeforge/topology/registry.py:129) |
| HF dependency | Absent | [`setup.py`](setup.py:46) |

### 2.1 Invariants that must not break

- The existing dataset `DownloadManager` and its `download_state` payload keys
  stay unchanged ([`downloads.py`](server/downloads.py:53)); the hub gets its
  own manager and its own state type.
- `model_store` remains the local registry for **trained** models; imported
  foreign models land in a separate hub cache and are only *promoted* into
  `MODEL_DIR` after a successful compat check
  ([`config.py`](spikeforge/config.py:17)).
- NIR import keeps raising the typed errors from
  [`errors.py`](spikeforge/nir_bridge/errors.py) — never a silent partial
  load.
- All new WebSocket fields are additive; the `ClientMessage`/`ServerMessage`
  literals only grow ([`client_message.py`](server/schemas/client_message.py:15),
  [`server_message.py`](server/schemas/server_message.py:11)).

---

## 3. Module layout

New package `spikeforge_hub/` (one class per file, files under 250 lines):

```
spikeforge_hub/
  __init__.py          public API: catalog, search, download, inspect, import_model
  entry.py             HubEntry dataclass: id, name, framework, kind, source...
  catalog.py           load/validate models.json; list + filter
  models.json          the bundled curated catalog (data file)
  probe.py             isolated huggingface_hub probe (hub extra)
  hf_api.py            isolated huggingface_hub search/download wrappers
  cache.py             offline cache dirs, path resolution, size accounting
  verify.py            checksum + size verification
  download_cli.py      isolated child-process downloader (mirrors data/download_cli.py)
  downloads.py         async HubDownloadManager: progress + cancel + verify
  inspect.py           structural report for a downloaded artifact
  compat.py            map an artifact to a preset or reject it, with a verdict
  weight_map.py        load compatible weights into a built preset module
  import_model.py      orchestrate inspect -> compat -> promote into MODEL_DIR
  errors.py            typed hub errors
  cli.py               spikeforge-hub entry point
```

Modified files:

| File | Change |
|---|---|
| [`setup.py`](setup.py:46) | add `hub` extra; add `spikeforge-hub` console script |
| [`config.py`](spikeforge/config.py:10) | add `HUB_CACHE_DIR` (env `SPIKEFORGE_HUB_DIR`, default `DATA_DIR/hub`) |
| [`server/protocol_handlers.py`](server/protocol_handlers.py:19) | route `hub_*` actions to `server/hub_handlers.py` |
| [`server/schemas/client_message.py`](server/schemas/client_message.py:15) | add `hub_list`, `hub_search`, `hub_download`, `hub_cancel`, `hub_inspect`, `hub_import` |
| [`server/schemas/server_message.py`](server/schemas/server_message.py:11) | add `hub_list`, `hub_search`, `hub_download_state`, `hub_inspect`, `hub_import` |
| [`client/src/useWebSocket.ts`](client/src/useWebSocket.ts) | dispatch the new replies |
| [`client/src/App.tsx`](client/src/App.tsx) | mount the `HubPanel` |

New server modules (kept out of [`handlers.py`](server/handlers.py:1) to respect
the 250-line limit, exactly like [`target_handlers.py`](server/target_handlers.py:1)):

```
server/hub_handlers.py    dispatch_hub + per-action handlers
server/hub_payloads.py    JSON payload builders (entry cards, inspect, compat)
server/hub_downloads.py   hub download manager singleton + state emitter
```

New client files: `client/src/hubTypes.ts`,
`client/src/components/HubPanel.tsx`, `HubEntryCard.tsx`,
`HubDownloadProgress.tsx`, `HubCompatBadge.tsx`, `client/src/hooks/useHub.ts`,
`client/src/styles/hub.css`.

---

## 4. The catalog schema

`hub/models.json` is a versioned list of entries. Each entry is validated into
a `HubEntry` ([`hub/entry.py`](spikeforge_hub/entry.py)). Required fields:

| Field | Type | Meaning |
|---|---|---|
| `id` | str | stable slug, e.g. `nir/conv_net` |
| `name` | str | human label |
| `framework` | str | `snntorch`, `nir`, `spikingjelly`, `norse`, `lava`, `hf` |
| `kind` | str | `nir_graph`, `state_dict`, `framework_weights` |
| `source` | str | `bundled`, `url`, or `hf_repo` |
| `url` | str? | for `source=url` |
| `hf_repo` | str? | for `source=hf_repo` |
| `sha256` | str? | expected checksum (per file) |
| `size_bytes` | int? | expected size |
| `topology` | str? | preset it matches, if any |
| `input_shape` | str? | declared input contract |
| `license` | str | declared license |
| `notes` | str | free text, honesty about provenance |

The catalog loader validates every entry and **reports** unknown frameworks
rather than dropping them (honesty rule). `catalog.list()` returns JSON-able
dicts with an additive `available` flag computed from
[`hub/probe.py`](spikeforge_hub/probe.py), exactly as
[`datasets.catalog()`](spikeforge/data/datasets.py:143) does.

Proposed seed entries (≥12 across ≥5 frameworks): the four shipped presets as
bundled NIR graphs, a small SpikingJelly reference, a Norse reference, a Lava
reference, and a handful of allow-listed HF SNN repos.

---

## 5. Download design

Reuse the **isolated child-process worker** pattern verbatim
([`download_cli.py`](spikeforge/data/download_cli.py:15),
[`downloads.py`](server/downloads.py:87)) so a download never blocks the
FastAPI event loop and can be terminated to cancel in flight.

- [`hub/download_cli.py`](spikeforge_hub/download_cli.py) downloads one
  entry (or HF repo) into the cache, then verifies checksum and size via
  [`hub/verify.py`](spikeforge_hub/verify.py). Non-zero exit on
  verification failure.
- [`hub/downloads.py`](spikeforge_hub/downloads.py) mirrors
  `DownloadManager`: `ensure`, `_poll`, `_finish`, `snapshot`, `cancel`, with a
  `hub_download_state` payload carrying `{id, status, bytes, total_bytes,
  verified}`. Terminal states stay `{idle, downloading, done, cancelled,
  error}` for consistency with [`downloads.py`](server/downloads.py:49).
- Progress is measured by cache-dir byte delta (same best-effort approach as
  [`_dir_bytes()`](server/downloads.py:29)); when `huggingface_hub` exposes a
  total, `total_bytes` is filled, else `null` (honest unknown).
- The offline cache dir is `HUB_CACHE_DIR` under `DATA_DIR`, gitignored like
  `MODEL_DIR`.

---

## 6. Import and inspect design

Import is a three-gate funnel; a model must pass each gate or be explicitly
rejected with a typed reason.

1. **Inspect** ([`hub/inspect.py`](spikeforge_hub/inspect.py)): detect the
   artifact kind (`nir_graph`, `state_dict`, `framework_weights`) and describe
   its structure without committing. NIR artifacts are summarized with
   [`graph_summary`](spikeforge/nir_bridge/exporter.py:86); state dicts are
   described by key/shape. Unknown or unreadable artifacts are rejected with
   `HubArtifactError`.
2. **Compat** ([`hub/compat.py`](spikeforge_hub/compat.py)): compare the
   inspected structure against the shipped presets in
   [`topology/registry.py`](spikeforge/topology/registry.py:78). Produce a
   `CompatibilityVerdict`: `exact`, `mappable` (with the stage mapping), or
   `incompatible` (with the specific mismatches named).
3. **Promote** ([`hub/import_model.py`](spikeforge_hub/import_model.py)):
   only on `exact`/`mappable`, build the preset via
   [`build_topology`](spikeforge/topology/registry.py:129), load weights via
   [`hub/weight_map.py`](spikeforge_hub/weight_map.py), run a drift check
   with [`validate`](spikeforge/nir_bridge/__init__.py:39), and only then
   save into `MODEL_DIR` with hub provenance recorded in `meta`.

This satisfies "reported and either mapped or explicitly rejected, never
silently loaded wrong". The weight loader reuses `load_state_dict` semantics
and refuses strict mismatches, reporting missing/unexpected keys.

NIR-only models that match no preset are still runnable: they are registered as
**imported NIR graphs** ([`ingest.py`](spikeforge/nir_bridge/ingest.py:21))
and executed by the reference interpreter, so import is useful even without a
preset match.

---

## 7. WebSocket surface

Request/response shapes (all `payload` additive; errors use the existing
`error` type per [`target_handlers.py`](server/target_handlers.py:87)).

| Action | Request fields | Reply | Reply payload |
|---|---|---|---|
| `hub_list` | `{framework?, kind?, available?}` | `hub_list` | `{entries: [{...entry, available}]}` |
| `hub_search` | `{query, limit?}` | `hub_search` | `{query, available, results: [...]}` |
| `hub_download` | `{name: id}` | `hub_download_state` (streamed) | download snapshot |
| `hub_cancel` | `{name: id}` | `hub_download_state` | terminal `cancelled` |
| `hub_inspect` | `{name: id}` | `hub_inspect` | `{kind, nodes, keys, notes}` |
| `hub_import` | `{name: id, topology?}` | `hub_import` | `{verdict, promoted, meta, validation}` |

`hub_search` reports `available: false` with a reason when the `hub` extra is
absent, rather than erroring. `hub_import` returns a verdict object even when
incompatible, so the client can render *why*.

---

## 8. CLI surface (`spikeforge-hub`)

[`hub/cli.py`](spikeforge_hub/cli.py), mirroring the argparse style of
[`target_cli.py`](spikeforge/cli/target_cli.py:166):

```
spikeforge-hub list [--framework snntorch] [--available]
spikeforge-hub search <query> [--limit 20]
spikeforge-hub download <id> [--no-verify]
spikeforge-hub inspect <id>
spikeforge-hub import <id> [--topology conv_net]
```

Every command prints JSON. `import` exits non-zero when the verdict is
`incompatible`, so it doubles as a CI gate (same convention as
[`deploy_exit`](spikeforge/cli/target_cli.py:86)).

---

## 9. Client panel

[`HubPanel.tsx`](client/src/components/HubPanel.tsx) is a browser + downloader:

- Filter bar: framework, kind, availability.
- Entry cards ([`HubEntryCard.tsx`](client/src/components/HubEntryCard.tsx)):
  name, framework badge, kind, size, license, `available` state, and a
  [`HubCompatBadge.tsx`](client/src/components/HubCompatBadge.tsx) showing the
  compat verdict.
- Download row reuses the existing progress UX
  ([`DownloadProgress.tsx`](client/src/components/DownloadProgress.tsx)) and
  adds cancel + verified state.
- Import action surfaces the inspect report and the compat verdict inline; an
  incompatible model shows the named mismatches rather than a generic failure.

Types live in `client/src/hubTypes.ts` (no `any`, 80-column).

---

## 10. Phases, deliverables, acceptance

### A1 — Catalog, probe, cache

- **Deliverables:** `hub/entry.py`, `hub/catalog.py`, `hub/models.json`,
  `hub/probe.py`, `hub/cache.py`, `hub/errors.py`, `hub/__init__.py`; `config.py`
  `HUB_CACHE_DIR`; `models.json` schema validation.
- **Acceptance:** `python -m spikeforge_hub.cli list` prints ≥10 entries with
  `available` flags; a malformed entry is reported (not silently skipped); no
  `huggingface_hub` import occurs without the extra.

### A2 — Downloader

- **Deliverables:** `hub/download_cli.py`, `hub/downloads.py`, `hub/verify.py`.
- **Acceptance:** `spikeforge-hub download <id>` fetches into the cache, verifies
  sha256 + size, and exits non-zero on a seeded checksum mismatch; cancellation
  terminates the child and reports `cancelled`.

### A3 — Import + inspect

- **Deliverables:** `hub/inspect.py`, `hub/compat.py`, `hub/weight_map.py`,
  `hub/import_model.py`.
- **Acceptance:** a shipped NIR artifact inspects and imports with verdict
  `exact`; a structurally perturbed artifact is rejected `incompatible` with the
  mismatched stage named; a matching preset artifact loads weights and
  `validate(...)` reports `within_tolerance=True` before promotion.

### A4 — Surfaces

- **Deliverables:** `server/hub_handlers.py`, `server/hub_payloads.py`,
  `server/hub_downloads.py`, schema additions, `hub/cli.py`, client panel.
- **Acceptance:** the six WS actions round-trip over a live connection;
  `spikeforge-hub import` gates on the verdict; the client builds and renders the panel
  from live payloads; existing dataset `download_state` payloads are unchanged.

---

## 11. Risks and deferred items

- **Licensing / redistribution:** whether weights may be bundled is a user
  decision (decision 1 in the roadmap). Default: metadata-only catalog; fetch
  weights on demand.
- **Network flakiness / HF API drift:** isolated in `hub/hf_api.py` behind the
  `hub` extra, exactly like [`api.py`](spikeforge/nir_bridge/api.py:1).
- **Checksum unknowns:** when a source publishes no checksum, `sha256` is
  `null` and the verifier reports "unverified" rather than passing silently
  (honesty rule).
- **Deferred:** model *push*/publish, model cards, and any training of
  downloaded models. Multi-user caches are out of scope.

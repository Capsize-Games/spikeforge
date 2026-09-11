# Spikeforge — WS-E: Operational Maturity

> Focused design for workstream **E** of the
> [`professional_roadmap.md`](plans/professional_roadmap.md). Read the roadmap
> first for the vision, cross-workstream interfaces, and packaging rules.

Design/spec only. Every claim about the current code is grounded in a file/line
reference so a Code-mode agent can execute this file-by-file.

**Objective.** Close the gap between "works" and "trustworthy in production":
persist the in-memory metrics registry, add optional external tracking sinks
(TensorBoard and/or Weights & Biases) behind extras **without changing the
local file-based default**, advance the reproducibility/bit-exactness story, and
generate a docs site from `plans/`. Multi-user sessions and auth are explicitly
out of scope.

---

## 1. Current state and gap analysis

| Capability | Current reality | Anchor |
|---|---|---|
| Metrics registry | In-memory only, surfaced via `system_stats` | [`metrics.py`](spikeforge/observability/metrics.py:14) |
| Metrics registry impl | `MetricsRegistry` counters/gauges/timers | [`registry.py`](spikeforge/observability/registry.py) |
| Structured logs | Opt-in JSON logging | [`logging_setup.py`](spikeforge/observability/logging_setup.py) |
| Reproducibility manifest | Config hash, seed, versions, history | [`manifest.py`](spikeforge/tracking/manifest.py:35) |
| Bit-exactness | Documented as **not** bit-exact | [`manifest.py`](spikeforge/tracking/manifest.py:70) |
| External tracking | Local `MODEL_DIR` only, by design | [`model_store.py`](spikeforge/network/model_store.py:34) |
| Benchmark store | File-based, regression gating | [`store.py`](spikeforge/benchmark/store.py), [`compare.py`](spikeforge/benchmark/compare.py) |
| Docs | Markdown plans only | [`ecosystem_roadmap.md`](plans/ecosystem_roadmap.md:1) |
| Persistence root | `DATA_DIR`, `MODEL_DIR` | [`config.py`](spikeforge/config.py:10) |

### 1.1 Invariants that must not break

- The local file-based manifest/checkpoint design remains the **default**; a
  sink is opt-in and its absence is never an error
  ([`ecosystem_roadmap.md`](plans/ecosystem_roadmap.md:372)).
- `system_stats` keeps its existing `cpu`/`gpu`/`device`/`metrics` keys
  ([`stats.py`](server/stats.py)).
- The benchmark store/suite/compare contracts are unchanged; energy is additive.
- `ReproducibilityManifest.to_dict` keys stay stable; new fields are additive
  ([`manifest.py`](spikeforge/tracking/manifest.py:60)).
- `ruff`, the test suite, and the client build stay green.

---

## 2. Persisted metrics

### 2.1 Design

The in-memory registry is useful live but lost on restart. Add a small,
file-based persistence layer that snapshots the registry to JSON on demand and
on shutdown, keyed by run/session, and can reload a previous snapshot for the
stats surface.

```
spikeforge/observability/
  store.py         read/write metric snapshots under METRICS_DIR
  persistence.py   hook the registry to the store; flush + load helpers
  snapshot.py      snapshot dataclass: timestamp, run id, metrics
```

- `METRICS_DIR` is a new setting in [`config.py`](spikeforge/config.py:10),
  defaulting to `DATA_DIR/metrics`, overridable with `SPIKEFORGE_METRICS_DIR`
  (roadmap decision 6).
- `persistence.flush()` writes the current
  [`metrics.snapshot()`](spikeforge/observability/metrics.py:42); `load()`
  returns the latest snapshot for a run id without mutating the live registry.
- The `system_stats` reply gains an additive `metrics_persisted` flag and the
  timestamp of the last flush; existing keys are untouched
  ([`stats.py`](server/stats.py)).
- The benchmark store ([`store.py`](spikeforge/benchmark/store.py)) is
  unchanged; it already persists runs, so energy/op-count blocks ride along.

---

## 3. External tracking sinks

### 3.1 Design

A sink is a tiny interface with one responsibility: receive a manifest-like
record and forward it to an external tracker. The default is **no sink**; the
local manifest is always written first, so a tracker outage never loses a run.

```
spikeforge/tracking/
  sink.py            Sink protocol: available(), log(record)
  sinks.py           registry + active-sink resolution from config
  tensorboard_sink.py  TensorBoard SummaryWriter wrapper (tracking extra)
  wandb_sink.py        W&B wrapper (tracking-wandb extra)
  sink_probe.py      isolated probes for tensorboard / wandb
```

| Sink | Extra | Module | Absent behavior |
|---|---|---|---|
| TensorBoard | `tracking` | `tensorboard` | `available()` false; note in manifest |
| W&B | `tracking-wandb` | `wandb` | `available()` false; note in manifest |

Selection: an additive `TrainConfig.tracking: Optional[str]` (values `null`,
`tensorboard`, `wandb`) defaulting to `null`. The manifest records
`tracking: {"requested": ..., "active": bool, "reason": ...}` so a run states
which sink it used and why, satisfying the honesty rule.

### 3.2 Wiring

[`CheckpointMixin._manifest`](spikeforge/training/checkpoint_mixin.py:55)
already builds the manifest; it calls `sinks.emit(manifest)` after the local
write. Because probes are isolated, an absent package is a recorded reason, not
a crash.

---

## 4. Reproducibility and bit-exactness

### 4.1 Design

The manifest already states `bit_exact: false` with the reasons
([`manifest.py`](spikeforge/tracking/manifest.py:70)). WS-E turns the
documented gap into a measurable path:

```
spikeforge/tracking/
  determinism.py   enable_deterministic(seed) + a bit-exactness check
```

- `enable_deterministic(seed)` sets the torch/cuDNN/Python/NumPy seeds and the
  deterministic-algorithm flags, returning a report of what it could and could
  not enforce (cuDNN nondeterminism on some hardware is reported, not asserted).
- A `bit_exactness_check(run_a, run_b)` helper reruns a tiny fixture twice under
  determinism and reports whether trajectories match exactly, upgrading the
  manifest's claim from "documented" to "verified on this fixture".
- The manifest gains an additive `determinism` block: `{enabled, exact_fixture,
  notes}`. Existing keys and `reproducible` are unchanged.

---

## 5. Docs site

### 5.1 Design

Generate a static docs site from `plans/` and `README.md` so the designs are
browsable, not buried.

```
mkdocs.yml                MkDocs Material config; nav from plans/
scripts/build_docs.sh     build + --check (fails on broken links)
```

- Nav mirrors the master → focused-doc structure
  ([`professional_roadmap.md`](plans/professional_roadmap.md) →
  `model_hub_plan.md`, `backend_execution_plan.md`, ...).
- `--check` runs `mkdocs build --strict` so a broken relative link fails CI,
  preserving the project-wide rule of clickable relative references.
- Docs generation is additive: the markdown source stays authoritative.

---

## 6. Surfaces

| Surface | Change |
|---|---|
| WebSocket | `system_stats` gains additive `metrics_persisted`/timestamp; no new action required |
| CLI | `spikeforge-benchmark` unchanged; new `spikeforge-docs` optional script wrapping `build_docs.sh` (roadmap lists it under packaging) |
| Client | [`ResourceMonitor.tsx`](client/src/components/ResourceMonitor.tsx) shows the persisted-metrics indicator; [`BenchmarkPanel.tsx`](client/src/components/BenchmarkPanel.tsx) shows the energy block when present |

---

## 7. Phases, deliverables, acceptance

### E1 — Persist metrics

- **Deliverables:** `observability/store.py`, `observability/persistence.py`,
  `observability/snapshot.py`, `config.py` `METRICS_DIR`, additive
  `system_stats` fields.
- **Acceptance:** a metric written, flushed, and reloaded from disk survives a
  simulated restart; `system_stats` still carries its existing keys; the default
  remains local and file-based.

### E2 — External tracking sinks

- **Deliverables:** `tracking/sink.py`, `tracking/sinks.py`,
  `tracking/tensorboard_sink.py`, `tracking/wandb_sink.py`,
  `tracking/sink_probe.py`, `TrainConfig.tracking`, manifest `tracking` block.
- **Acceptance:** with no extra installed, a run writes the local manifest and
  records `active: false` with a reason; with `tracking` installed, a
  TensorBoard event file is produced for a tiny run; the local manifest is
  always written first.

### E3 — Determinism and docs site

- **Deliverables:** `tracking/determinism.py`, manifest `determinism` block,
  `mkdocs.yml`, `scripts/build_docs.sh`.
- **Acceptance:** `enable_deterministic` returns a report; the bit-exactness
  fixture check is green on CPU; `build_docs.sh --check` passes and fails on a
  deliberately broken link.

---

## 8. Risks and deferred items

- **Optional-dependency weight:** `tensorboard`/`wandb` are extras; probes keep
  their absence non-fatal.
- **Determinism is hardware-dependent:** the report states what could not be
  enforced rather than promising bit-exactness universally.
- **Deferred (explicitly out of scope):** multi-user sessions, authentication,
  remote users, and hosted tracking. These are noted here so the boundary is
  unambiguous.

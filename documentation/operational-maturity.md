# Operational maturity (WS-E)

### Persisted metrics

The in-process registry can be snapshotted to disk. Persistence is **opt-in**
via `SPIKEFORGE_METRICS_PERSIST`; with it unset, `flush()` is a no-op and behaviour is
unchanged. Snapshots
([`MetricSnapshot`](../spikeforge/observability/snapshot.py:14)) are written
under `METRICS_DIR` (`SPIKEFORGE_METRICS_DIR`, default `<DATA_DIR>/metrics`).
`system_stats` gains additive `metrics_persisted` and `metrics_last_flush`
keys; existing keys are untouched.

### External tracking sinks

`TrainConfig.tracking` selects `tensorboard` (extra `tracking`) or `wandb`
(extra `tracking-wandb`), default `null`. The **local manifest is always
written first**, so a tracker outage never loses a run; an absent backend
becomes a recorded `reason` in the manifest's `tracking` block
([`sinks.describe()`](../spikeforge/tracking/sinks.py:68)). Probes are
isolated in [`tracking/sink_probe.py`](../spikeforge/tracking/sink_probe.py:1).

### Determinism

[`tracking.determinism.enable_deterministic()`](../spikeforge/tracking/determinism.py:82)
seeds Python/NumPy/torch and sets the deterministic-algorithm flags, returning
a report of what it could and could not enforce;
[`bit_exactness_check()`](../spikeforge/tracking/determinism.py:117) reruns a
fixture and reports exactness. `TrainConfig.deterministic` (default off) opts a
run in, and the manifest gains an additive `determinism` block. This narrows
the bit-exactness gap; it does not claim universal bit-exactness.

### Docs site

[`mkdocs.yml`](../mkdocs.yml) and [`scripts/build_docs.sh`](../scripts/build_docs.sh)
render `plans/` and this README into a browsable Material site:

```bash
scripts/build_docs.sh          # build into build/docs
scripts/build_docs.sh --check  # fail on broken documentation links
```

The markdown remains authoritative; the site is a rendering of it. The `docs`
extra provides MkDocs Material.

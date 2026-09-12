# Production workflows (Phase 6)

Phase 6 closes the gap between "works" and "trustworthy in production": a run
records how to reproduce itself, checkpoints become searchable, training
scales up behind opt-in flags, performance is tracked over time, and logs and
metrics become machine-readable. Every addition is opt-in and the default
workflow is unchanged.

### Reproducibility manifest and config hash

[`spikeforge/tracking/`](../spikeforge/tracking/__init__.py:1) records
what a run needs to be recreated and compared. A
[`ReproducibilityManifest`](../spikeforge/tracking/manifest.py:35) captures
the dataset, topology and params, encode config, hyperparameters, the resolved
`TopologySpec`, the library versions, the seed, and the metric history.
[`config_hash()`](../spikeforge/tracking/config_hash.py:19) hashes the
reproducibility-relevant config as canonical JSON (sorted keys, tight
separators), so two runs with identical settings compare equal regardless of
when they ran or what their histories show, and
[`set_seed()`](../spikeforge/tracking/seed.py:31) seeds Python, PyTorch, and
every CUDA device.

Reproducibility is stated honestly in the manifest's `reproducible` block; it
is **not** claimed to be bit-exact. Guaranteed to reproduce from the manifest
alone: the topology structure and resolved spec, the dataset/encode config and
hyperparameters, the library versions and seed, and the initial parameter
values for the same library build on CPU. **Not** guaranteed bit-for-bit: CUDA
kernels (cuDNN, parallel reductions), hardware thread scheduling and any float
summation order that follows, and dataset contents if the source files change
between runs. `set_seed` deliberately leaves the global deterministic flags
alone so seeding never slows the default training path.

### Searchable registry, `records` CLI, and metadata diffing

The file-based `MODEL_DIR` registry stays the source of truth, but two
read-only helpers make it searchable and comparable.
[`search_models(...)`](../spikeforge/network/model_search.py:87) filters
checkpoint summaries by dataset, topology, coding, device, minimum accuracy,
and a case-insensitive name substring; each result carries the newest non-null
test accuracy and the stored manifest (or `null` for a legacy checkpoint).
[`checkpoint_diff(...)`](../spikeforge/network/model_diff.py:102) classifies
every metadata key as `added`, `removed`, `changed`, or `same` and compares the
two manifests' config hashes. `list_models` and its payload shape are
untouched, so existing callers are unaffected.

The `spikeforge-records` console script (also `verify records ...`) exposes both:

```bash
spikeforge-records list --dataset mnist --topology conv_net --min-accuracy 90
spikeforge-records diff old_model new_model
spikeforge-records manifest my_model
```

The server mirrors this with read-only `model_search` and `model_diff`
WebSocket actions (`model_search`/`model_diff` replies); a diff without exactly
two names emits the existing `error` message.

### Training scale-ups (opt-in, default-off)

[`ScaleUpMixin`](../spikeforge/training/scaleup_mixin.py:29) adds four
additive options to `TrainConfig`. Every default reproduces the previous
behaviour exactly:

| Flag | Default | What it does |
|---|---|---|
| `amp` | `False` | Autocast float16 on CUDA / bfloat16 on CPU, with a CUDA `GradScaler`; falls back to fp32 when the device rejects the dtype |
| `grad_checkpoint` | `False` | Recomputes each step's activations during the backward pass (smaller activation footprint, more compute) |
| `bptt_steps` | `None` | Detaches the carried neuron state every N steps (truncated BPTT); `None` keeps full backprop-through-time |
| `multi_gpu` | `False` | `DataParallel` fan-out when more than one CUDA device is visible |

AMP numerics are close to, but not bit-identical to, fp32.
[`MultiDeviceManager`](../spikeforge/training/multi_device.py:26) reports an
honest status (`disabled`, `unavailable: ...`, or `active: N cuda devices`)
instead of failing, and both gradient policies live in the one shared temporal
loop via [`GradPolicy`](../spikeforge/simulator/grad_policy.py:44), so the
forward values are untouched when either is off.

### Performance suite: store, suite, and compare

Runs can be recorded and regressions caught over time. A
[`BenchmarkStore`](../spikeforge/benchmark/store.py:40) keeps one JSON record
per run under `SPIKEFORGE_BENCHMARK_DIR` (default `<DATA_DIR>/benchmarks`), and
[`run_suite(...)`](../spikeforge/benchmark/suite.py:55) benchmarks a set of
topologies, attaches the library versions plus a timestamp, and saves the
record:

```bash
# record a CI-sized suite (the saved run id is <timestamp>-<label>)
python -m spikeforge.benchmark --topology fc_small --topology conv_net \
    --steps 8 --repeats 3 --save --label main

# list every stored run, newest first (the list prints each run_id)
python -m spikeforge.benchmark --list

# compare a stored baseline against a fresh run; exit 1 on regression
python -m spikeforge.benchmark --compare <run-id> --threshold 0.1 \
    --fail-on-regression

# or diff two stored runs
python -m spikeforge.benchmark --compare <baseline-id> --against <run-id>
```

[`compare_runs(...)`](../spikeforge/benchmark/compare.py:123) matches records
on `(topology, mode)` and reports the relative change in `ms/step`, `steps/s`,
and peak memory, flagging a regression when a metric moves the wrong way past
the threshold (`--fail-on-regression` turns that into a non-zero exit, so the
command works as a CI gate). `--compare` takes a stored **run id, not a
label**; `--list` prints the ids. The contract is JSON-able end to end.

### Observability: opt-in logs and metrics

[`spikeforge/observability/`](../spikeforge/observability/__init__.py:1)
adds two opt-in surfaces, neither enabled unless asked:

- **Structured logging.** `configure_logging()` attaches one handler to the
  `spikeforge` logger (never the root) and `reset_logging()` restores the
  exact prior state. Set `SPIKEFORGE_LOG_JSON=1` for JSON lines (`timestamp`,
  `level`, `event`, `logger`, plus optional `run_id` / `config_id` /
  `config_hash` / `fields`) or `SPIKEFORGE_LOG_LEVEL=DEBUG` for a level. With neither
  variable set the default human-readable behaviour is untouched, and no entry
  point calls `configure_logging()` for you.
- **Metrics snapshot.** `spikeforge.observability.metrics` is a
  process-wide [`MetricsRegistry`](../spikeforge/observability/registry.py:27)
  of counters, gauges, and timers. The training loop records `train.steps`,
  `train.encode_seconds`, `train.forward_seconds`, and
  `train.backward_seconds`; the validation path records `validation.runs`,
  `validation.seconds`, and `validation.accuracy` (plus `validation.drift` on
  the NIR interpreter). `metrics.snapshot()` returns JSON-able data and is
  surfaced additively as the `metrics` key of the `system_stats` payload.

### Console scripts

Packaging installs a console script per surface, so every headless command has
a stable name:

| Script | Equivalent |
|---|---|
| `spikeforge` | `python main.py` |
| `spikeforge-encodings` | `python main_encodings.py` |
| `spikeforge-verify` | `python -m spikeforge.cli.verify` |
| `spikeforge-records` | `python -m spikeforge.cli.verify records` |
| `spikeforge-targets` | `python -m spikeforge_targets.cli.target_cli` |
| `spikeforge-hub` | `python -m spikeforge_hub.cli` |
| `spikeforge-energy` | `python -m spikeforge_targets.energy.cli` |
| `spikeforge-benchmark` | `python -m spikeforge.benchmark` |

### Docker CPU/GPU profiles

Two opt-in Compose [profiles](../docker-compose.yml) (`cpu`, `gpu`) select
explicit CPU-only / CUDA builds of the same service without changing the
default `docker compose up --build` (CUDA image, dashboard on port 8877, host
GPU reserved). Only one profile can own port 8877 at a time; see the
[Docker profiles](usage.md#docker-profiles-cpu-and-gpu) subsection in Usage.

# snn-interpreter examples

Small, runnable, **offline-safe** scripts, one per core journey. Each has a
module docstring explaining what it demonstrates and how to run it, and each
is importable (all work is behind `if __name__ == "__main__":`).

Run any script from the repository root with the project virtualenv:

```bash
venv/bin/python examples/01_train_image_model.py
```

They complement the copy-pasteable recipes in [`COOKBOOK.md`](../COOKBOOK.md)
by showing **end-to-end runs with real output** rather than command
fragments. Two of them depend on an optional extra and degrade honestly when
it is absent (see the last section).

| # | Script | Journey |
|---|---|---|
| 1 | [`01_train_image_model.py`](01_train_image_model.py) | Train an image model headlessly |
| 2 | [`02_train_event_model.py`](02_train_event_model.py) | Train an event model via the labelled synthetic path |
| 3 | [`03_encode_and_inspect.py`](03_encode_and_inspect.py) | Encode + inspect spikes |
| 4 | [`04_nir_export_validate.py`](04_nir_export_validate.py) | Export to NIR + validate drift |
| 5 | [`05_hub_browse_import.py`](05_hub_browse_import.py) | Browse/inspect/import from the hub |
| 6 | [`06_deploy_and_run_backend.py`](06_deploy_and_run_backend.py) | Deployment report + reference backend |
| 7 | [`07_energy_sparse_dense.py`](07_energy_sparse_dense.py) | Estimate energy sparse-vs-dense |
| 8 | [`08_sequence_experiments.py`](08_sequence_experiments.py) | Sequence experiments (exportable vs simulation-only) |
| 9 | [`09_onnx_roundtrip.py`](09_onnx_roundtrip.py) | ONNX round-trip |
| 10 | [`10_reproducibility_benchmark.py`](10_reproducibility_benchmark.py) | Reproducibility manifest + benchmark save/compare |

---

## 1. Train an image model headlessly

`01_train_image_model.py` runs the fully-connected LIF network on MNIST for
two batches with the Python `TrainingEngine`. The first run downloads MNIST
into `SNN_DATA_DIR` (default `build/`); later runs are offline. Representative
output:

```text
metric keys: ['epoch', 'loss', 'step', 'test_accuracy', 'total', 'train_accuracy']
loss=1.8540 train_accuracy=0.333 test_accuracy=63.75
```

`train_accuracy` is the `-1.0` sentinel until the metric is first computed,
and `test_accuracy` is a percentage (`null` until an evaluation step runs).

## 2. Train an event model via the labelled synthetic path

`02_train_event_model.py` serves explicitly-labelled synthetic N-MNIST
streams through `EventSampleSource(synthetic_only=True)`, bridges them with
`EventSpikeBridge`, and trains `fc_legacy` with `EventTrainingEngine`.
Representative output:

```text
origin: synthetic | label: 0
description: synthetic n_mnist events on a 28x28 sensor (offline; no real recording)
sensor: (28, 28) bridged: (10, 1, 784)
bridge sparsity: 0.9974
loss=2.3026 test_accuracy=12.5
```

Accuracy near chance is expected: this is a wiring demo, not a benchmark.
Without `synthetic_only=True` a missing `tonic` raises the typed
`EventsExtraMissingError` instead of passing a synthetic stream off as a
recording.

## 3. Encode and inspect spikes

`03_encode_and_inspect.py` encodes one registry sample with every coding and
reports firing rate, sparsity, and reconstruction support. Representative
output (the `rate` and `random` codings are stochastic, so those rows vary):

```text
rate    firing_rate=0.1369 sparsity=0.8631 supported=True
latency firing_rate=0.1000 sparsity=0.9000 supported=True
delta   firing_rate=0.1378 sparsity=0.8622 supported=True
random  firing_rate=0.2464 sparsity=0.7536 supported=False
```

Reconstructions are explicitly approximate; `random` carries no image signal,
so `reconstruction_supported` is `False`.

## 4. Export to NIR and validate drift

`04_nir_export_validate.py` renders `conv_net` into a `nir.NIRGraph`, prints
its node inventory, and runs the independent interpreter. Representative
output:

```text
topology: conv_net | input: (4, 1, 1, 28, 28)
nodes: 20
kinds: ['Input', 'Conv2d', 'LI', 'Threshold', 'Delay', 'Scale', 'AvgPool2d', 'Conv2d', 'LI', 'Threshold', 'Delay', 'Scale', 'SumPool2d', 'Flatten', 'Affine', 'LI', 'Threshold', 'Delay', 'Scale', 'Output']
within_tolerance: True
worst: {'layer': 'lif1', 'quantity': 'membrane', 'metric': 'mean_abs', 'value': 1.79e-07}
```

The membrane residual is the documented Euler-versus-zero-order-hold
difference, reported rather than hidden. The `worst` value varies slightly
run to run.

## 5. Browse, inspect, and import from the hub

`05_hub_browse_import.py` uses the hub Python API against the fully-offline
bundled catalog. Representative output:

```text
catalog entries: 10
frameworks: ['lava', 'nir', 'norse', 'snntorch', 'spikingjelly']
live Hugging Face search available: False
  (live search needs the `hub` extra: huggingface_hub)
entry: nir/fc_legacy
inspect kind: nir_graph
verdict: exact
promoted: True
destination: .../build/models/hub_nir_fc_legacy.pt
```

Promoting writes a checkpoint into `MODEL_DIR` (the gitignored
`build/models`). The shell equivalents are `snn-hub list`,
`snn-hub inspect nir/fc_legacy`, and `snn-hub import nir/fc_legacy`.

## 6. Deployment report and the reference backend

`06_deploy_and_run_backend.py` classifies `conv_net` against the always-available
`reference` target and the SDK-gated `xylo` target, then runs the reference
backend. Representative output:

```text
reference deployable=True counts={'supported': 20, 'unsupported': 0, 'substituted': 0, 'total': 20}
xylo deployable=False counts={'supported': 15, 'unsupported': 5, 'substituted': 0, 'total': 20}
reference status: ok
readout max_abs: 0.0
spike agreement: 1.0
```

`xylo deployable=False` is honest: its SDK is not installed. The shell
equivalents are `snn-verify deploy --topology conv_net --target reference`
and `snn-verify run --topology conv_net --target reference`.

## 7. Estimate energy, sparse versus dense

`07_energy_sparse_dense.py` counts SOP/MAC/AC and maps them onto the
`reference` target's declared cost table, sparse and dense. Representative
output:

```text
basis: declared cost table | estimate: True
sparse sop: 60856 mac: 2320640
dense  sop: 2320640 mac: 2320640
sop_over_mac sparse=0.0262 dense=1.0000
energy total sparse=480056.0 pJ dense=11778976.0 pJ
```

Every number is an **estimate** from a `"measured": false` table: sound for
comparing models and sparse-versus-dense trade-offs, never a power budget.
The sparse SOP (and therefore the sparse energy) depends on the randomly
initialised weights, so it varies run to run; the dense SOP/MAC is fixed by
the fixture geometry.

## 8. Sequence experiments

`08_sequence_experiments.py` shows the exportable `sequence_mlp` preset and
the simulation-only `sequence_attn` preset. Representative output:

```text
sequence_mlp frames: (4, 1, 8, 8)
sequence_mlp within_tolerance: True
sequence_attn export refused: kind=embedding
  reason: stage kind 'embedding' cannot be mapped to NIR: the installed nir has no Embedding primitive, so a token lookup table cannot be represented; the stage is simulation-only
sequence_attn simulated logits: (1, 8, 4)
```

`sequence_attn` stays available for simulation and introspection; export
raises the typed `UnsupportedStageError` naming the first unexportable stage.

## 9. ONNX round-trip

`09_onnx_roundtrip.py` exports a topology's single forward step, re-imports
it, and reads a written file back. Representative output:

```text
topology: conv_net
source: metadata | identical: True
ops: ['Add', 'AveragePool', 'Cast', 'Clip', 'Constant', 'Conv', 'Flatten', 'Gemm', 'Greater', 'Identity', 'Mul', 'Sub']
temporal: single-step; the time loop stays in the simulator
re-import source: metadata
```

An exported ONNX file is **not** a complete temporal SNN. Without the `onnx`
extra the script prints a clear message and exits `0`. The export surfaces an
upstream snnTorch `TracerWarning` during tracing; it is benign and comes from
`torch.onnx.export`, not this project.

## 10. Reproducibility manifest and benchmark save/compare

`10_reproducibility_benchmark.py` shows an order-independent config hash, the
manifest's honest `reproducible` block, and a benchmark save/compare cycle.
Representative output:

```text
config_hash stable: True
manifest hash: 24d48de851f4acabcc8b9ade09d305779814616446d8b079a638ca396bce7313
bit_exact: False
saved benchmark run: 1789084378656-example
compared cases: 2
```

The manifest hash covers configuration only; it makes a run reproducible, not
bit-exact. The benchmark writes into a temporary directory.

---

## Optional extras and honest degradation

- **`hub`** (`huggingface_hub`) enables live Hugging Face search/download.
  Without it, `05_hub_browse_import.py` still browses the bundled catalog and
  reports `live Hugging Face search available: False` with the reason.
- **`onnx`** (`onnx`, `onnxruntime`) enables the bridge.
  `09_onnx_roundtrip.py` detects its absence and prints
  `the 'onnx' extra is not installed; skipping.`

The other eight scripts use only the core install plus `nir`/`events` (or the
cached datasets). The console-script equivalents for every journey are listed
in [`COOKBOOK.md`](../COOKBOOK.md); see
[`OPEN_SOURCE_CHECKLIST.md`](../OPEN_SOURCE_CHECKLIST.md) for the project's
pre-release state.

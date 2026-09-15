# Benchmarks — what the reference configurations actually score

Until now this project could not answer, with a link, the first question an
academic or a professional asks: *what does it achieve, on what, in how long,
and can I reproduce it?* This page is that link.

**Read the caveat first.** Every row below is a **reference configuration**:
stock hyperparameters, a modest epoch count, one seed, CPU only. They exist so
a number is checkable, and so the model hub has honestly-labelled trained
weights to carry. They are **not** tuned attempts at state of the art, and
they should not be cited as spikeforge's ceiling — a longer schedule, a tuned
learning rate, and a GPU all move these numbers. What they are is true,
reproducible, and produced by the shipped code path rather than a bespoke
script.

## Results

| Configuration | Dataset | Topology | Test accuracy | Epochs | Steps | Train time |
|---|---|---|---|---|---|---|
| `mnist-fc-legacy` | mnist | `fc_legacy` | **94.29%** | 3 | 25 | 49.8 s |
| `mnist-fc-small` | mnist | `fc_small` | **92.16%** | 3 | 25 | 40.4 s |
| `mnist-conv-net` | mnist | `conv_net` | **97.13%** | 2 | 25 | 249.3 s |
| `mnist-recurrent-net` | mnist | `recurrent_net` | **92.46%** | 3 | 25 | 54.8 s |
| `fashion-fc-legacy` | fashion | `fc_legacy` | **75.55%** | 3 | 25 | 62.2 s |
| `kmnist-fc-legacy` | kmnist | `fc_legacy` | **70.79%** | 3 | 25 | 49.3 s |

Hardware: x86_64, 12 torch threads, CPU only. Python 3.13.5, torch 2.14.0+cpu.

Accuracy is measured on the **complete** held-out test split, not a sample of
it. `TrainingEngine.evaluate()` deliberately scores four cached batches of
1000 every fifth step so the live dashboard stays responsive; that is the
right trade for a dashboard and the wrong one for a full-dataset run, where it
costs several times more than the training it reports on. The harness
therefore shrinks the in-training progress probe to a single small batch and
walks the entire test split once, at the end, for the published number. The
per-step `test_accuracy` recorded in a checkpoint's manifest is that small
probe, not the figure in this table.

## Reproducing a row

Every row is produced by one command:

```bash
pip install "spikeforge[nir]"
git clone https://github.com/Capsize-Games/spikeforge.git
cd spikeforge
python scripts/train_reference_models.py --only mnist-fc-legacy
```

`--list` prints the configuration names; omitting `--only` runs all of them.
The script writes three things under `build/reference/`:

- `results.json` — one record per configuration, with the full hyperparameter
  set, the hardware it ran on, the seed, and a SHA-256 of the checkpoint;
- `results.md` — the table above;
- `<name>.pt` — the trained checkpoint, carrying the reproducibility manifest.

Seeds are fixed, so a rerun on the same torch version and hardware reproduces
the number. Across torch versions or hardware, expect small movement — that is
a property of floating-point reduction order, not of the configuration.

## What is *not* measured here

Being explicit about the gaps matters more than the table does:

- **No event-dataset rows.** N-MNIST, DVS128 Gesture, CIFAR10-DVS and Spiking
  Speech Commands are shipped and wired (see
  [Event datasets](event-datasets.md)), but they are not in this table yet.
  Nothing blocks adding them beyond download size and runtime.
- **No comparison against published SNN numbers.** The literature's MNIST and
  Fashion-MNIST SNN results are generally obtained with longer schedules,
  tuned hyperparameters, and different encoders, so putting them in the same
  table would invite an apples-to-oranges reading. Treat these as a floor that
  the shipped defaults reach, not as a competitive claim.
- **These are task-accuracy numbers, not latency or throughput.** For those,
  `spikeforge-benchmark` has a separate harness with fixtures, warmup,
  repeats, a result store, and `--fail-on-regression` for CI — see
  [Production workflows](production-workflows.md).
- **Energy figures are estimates.** The SOP/MAC/AC accounting in
  [Event runtime and energy](event-runtime-and-energy.md) is labelled
  `estimate` throughout and is not hardware-measured. It is not in this table
  for that reason.

## Where the checkpoints go

The checkpoints these runs produce are the trained entries in the model hub:
`source: "reference"` in
[`spikeforge_hub/models.json`](../spikeforge_hub/models.json), shipped inside
the `spikeforge-hub` wheel and checksum-verified on load. List them with:

```bash
pip install spikeforge-hub
spikeforge-hub list --trained
```

See [Model hub](model-hub.md) for how a reference entry differs from a
`bundled` one (trained weights versus preset structure), and
`spikeforge_hub/CURATION.md` for the policy that keeps the two apart.

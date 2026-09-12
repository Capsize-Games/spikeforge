# Streaming time-series classification / anomaly detection (UC-1 MVP)

The first fully scoped production use case: a continuous numeric stream
(vibration, machine telemetry, grid/sensor channels) delivered as windows,
classified or scored for anomalies per window, on CPU with no neuromorphic
hardware. This page documents the **MVP (P0–P4)** delivered by
[`spikeforge/streaming/`](../spikeforge/streaming/__init__.py:1) against the
design in
[`plans/use_case_streaming_timeseries.md`](../plans/use_case_streaming_timeseries.md).

## What is implemented

| Phase | Deliverable | Where |
|---|---|---|
| P0 | Frozen windowing + z-score contract | [`window_spec.py`](../spikeforge/streaming/window_spec.py:1) |
| P0 | Window → spike encode (delta primary, rate fallback) | [`encoding.py`](../spikeforge/streaming/encoding.py:1) |
| P1 | Deterministic synthetic stream + labels + splits | [`stream_source.py`](../spikeforge/streaming/stream_source.py:1) |
| P2 | `sequence_mlp` CPU training recipe | [`recipe.py`](../spikeforge/streaming/recipe.py:1) |
| P3 | Class metrics + one-class anomaly score + threshold rule | [`recipe.py`](../spikeforge/streaming/recipe.py:1) |
| P4 | `.spkf` bundle, session, and serving parity | [`serving.py`](../spikeforge/streaming/serving.py:1) |

**Deferred (not MVP):** P5 (a `/metrics` latency gate and serving benchmark)
and P6 (container, promotion/rollback, drift monitor). The serving *endpoints*
themselves are PT-W3's `spikeforge-serve`; UC-1 reuses them.

## P0 — windowing and the frozen encode contract

A stream `[N, D]` becomes fixed-shape windows with a frozen
[`WindowSpec`](../spikeforge/streaming/window_spec.py:1) that pins the window
`length`, `stride`, channel order, and the **train-fitted** per-channel
mean/std. Validation and test windows are normalised with the train
statistics, and the same spec is stored in the bundle's `preprocessing.json`,
so a served window can never be normalised differently from a training one.

Windows are encoded by the one
[`encode_windows`](../spikeforge/streaming/encoding.py:1) call site, which
resolves the shared
[`EncodeSpec`](../spikeforge/serving/encode_spec.py:1) and dispatches through
the core `SpikeEncoder`, reshaping its flat code to `[T, B, L, D]`. `delta`
coding is deterministic and is the primary contract; `rate` is the fallback.
The optional `delta_over_window` transform differences the window along its
time axis first, so the temporal change — not the absolute level — is what the
spike train carries.

## P1 — deterministic synthetic dataset

[`generate_stream`](../spikeforge/streaming/stream_source.py:1) stitches
class-labelled segments of smooth multi-channel waveforms, injects transient
and level-shift anomalies, and windows the stream. The class of segment `i` is
`i % classes`, labels are a window's centre class, and the anomaly flag is
raised when any injected anomaly lands inside the window. Train, validation,
and test use independent seeds and the z-score statistics are fitted on train
only, so no test information leaks. Everything is seeded: the same
`StreamSpec` reproduces the same windows and labels.

## P2/P3 — training, evaluation, anomaly head

The recipe trains the NIR-mappable `sequence_mlp` trunk on windowed spikes
with the shared [`run`](../spikeforge/simulator/runner.py:1) temporal loop and
surrogate-gradient cross-entropy — CPU only, deterministic, seconds to run.
Evaluation reports class accuracy, macro-F1, balanced accuracy, per-class
recall, and the additive one-class anomaly score `1 - max softmax` whose
threshold is the `percentile`-th quantile of the **train** score distribution
(a rule, not a per-test fit). AUROC is the tie-corrected Mann-Whitney
statistic; no scikit-learn dependency is used.

## P4 — bundle, serving, and parity

[`save_checkpoint`](../spikeforge/streaming/serving.py:1) records the topology
spec, the encode spec, and the windowing contract; `build_bundle` freezes both
into a `.spkf`. `predict_batch` runs the whole batch through `run`, while
`stream_logits` drives a stateful
[`InferenceSession`](../spikeforge/serving/session.py:1) one timestep at a
time. Because both paths share the same per-step body, the streaming readout
equals the closed-loop batch reference **exactly** (`max_abs_diff == 0.0` in
the example) — parity by construction, not by tolerance. `spikeforge-serve`
drives that same session, so `serve_window` reproduces the batch readout too,
and `reset` clears the carried state.

## Running it

```bash
venv/bin/python examples/11_streaming_timeseries.py
```

Representative output (the synthetic task is small; numbers move with the
config):

```text
windows train/val/test: 95 95 95
window shape: (95, 16, 4)
train anomalies: 34
trained: epochs=25 loss=0.7301 train_accuracy=0.842
test: accuracy=0.884 macro_f1=0.884 balanced_accuracy=0.885
anomaly: auroc=0.680 threshold=0.6240 precision=0.000 recall=0.000
bundle: uc1.spkf coding=delta window_L=16
parity: max_abs_diff=0.00e+00 within_tolerance=True
serve: spikeforge-serve not installed; skipped
```

## Honesty and limits

- This is a runnable MVP slice, not a benchmark. The synthetic dataset is the
  CI fixture; the plan's public-dataset adapter and the `spikeforge-io`
  windowing distribution (W7) are future work, so the helpers live in the core
  `spikeforge.streaming` package for now.
- The plan's MVP floor is dataset-specific; the tests assert the metrics are
  sane (accuracy above chance, AUROC ≥ 0.5) rather than a tuned target.
- The anomaly head is the additive `1 - max softmax` score reusing the
  classifier trunk, exactly as the plan allows; no separate one-class network
  is trained.
- Only `InferenceSession` + `spikeforge-serve` are exercised for P4; P5's
  Prometheus/OTel metrics and CI latency gate and P6's container/rollback/drift
  monitor are out of scope.
- `spikeforge-serve` is a separate distribution; the example degrades honestly
  when it is not installed.

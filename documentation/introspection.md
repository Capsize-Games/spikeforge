# Dual-mode introspection (Phase 2)

Phase 2 adds the educational/professional execution split and full
neuron-state introspection behind one shared code path.

### Execution modes

`ExecutionMode` ([`runtime/execution_mode.py`](../spikeforge/runtime/execution_mode.py))
is a flag on the single temporal loop, not a fork:

- `EDUCATIONAL` records every per-step trace; `PRODUCTION` records none and
  runs lean.
- [`simulator.run()`](../spikeforge/simulator/runner.py:21) takes
  `mode=...` and also exposes `track` / `membrane` / `current` for
  finer-grained capture.
- [`simulator.run_production()`](../spikeforge/simulator/production.py:14)
  returns a `ProductionResult(trajectory, compiled, status)`. `compiled` is
  opt-in (`torch.compile`) and falls back transparently to eager, with
  `status` in `{"eager", "unavailable", "compiled", "fallback"}`.

Both paths share one loop, so the difference is recording overhead, not
behaviour.

### Trajectory capture

`Trajectory` ([`simulator/trajectory.py`](../spikeforge/simulator/trajectory.py:9))
carries the averaged readout `logits` plus per-neuron-stage traces of spikes
`S[t]`, membrane `U[t]`, and input current `I[t]` (`currents` is the merged
inbound activation each stage received before its update). Educational mode
fills all three; production mode leaves them empty.

### Trajectory metrics

[`introspection.metrics.trajectory_metrics()`](../spikeforge/introspection/metrics.py:28)
gathers, per stage:

- **firing rate** — mean spikes per neuron per step.
- **sparsity** — fraction of silent entries.
- **ISI** — count/mean/median/std/cv of inter-spike intervals (`null` when
  fewer than two spikes).
- **histogram** — per-neuron firing-rate bin edges and counts.

The result holds only plain JSON types.

### Encoding and decoding introspection

[`introspection.encoding.encoding_report()`](../spikeforge/introspection/encoding.py:142)
encodes one image, reconstructs it where the coding is invertible, and
reports firing rate, sparsity, and coding-specific stats. The reconstruction
is explicitly approximate, documented in the report's `approximation` field:

- **rate** — mean spike count; a Bernoulli estimate of the clamped intensity
  that converges as `num_steps` grows.
- **latency** — inverts the time-to-first-spike map; quantised to integer
  steps and saturating at the threshold ceiling for sub-threshold pixels.
- **delta** — integrates the on/off stream crediting one threshold per
  spike; a lower bound, exact only when each step rises by exactly the
  threshold.
- **random** — carries no image signal, so `reconstruction` is `null` and
  `reconstruction_supported` is `false`.

### Surrogate gradients

[`introspection.surrogate`](../spikeforge/introspection/surrogate.py:1)
discovers the selectable surrogate factories from the installed
`snntorch.surrogate` (so the list always matches what snnTorch provides).
`list_surrogates()` names them and
[`surrogate_curve()`](../spikeforge/introspection/surrogate.py:94) samples
the backward-pass derivative `dS/dU` into parallel `x`/`y` lists. Neurons
accept an optional `surrogate` build parameter; leaving it unset (the
default) keeps the build byte-identical to before.

### Neuron comparison lab

[`introspection.comparison.compare_neurons()`](../spikeforge/introspection/comparison.py:59)
runs the same seeded input through every registered neuron kind (Leaky,
Lapicque, Synaptic, recurrent LIF, and Alpha) and returns
`{kind: Trajectory}` for side-by-side diffing.

### Benchmark harness

[`spikeforge/benchmark/`](../spikeforge/benchmark/__init__.py:1)
measures wall time and memory of forward and backward passes for each mode
(and, with `--compiled`, the compiled production path):

```bash
python -m spikeforge.benchmark                     # tiny default fixture
python -m spikeforge.benchmark --topology conv_net --steps 16 --compiled
python -m spikeforge.benchmark --out bench.json
```

The same report is available from Python via
`benchmark.run_benchmark(BenchmarkConfig(...))`; every measurement is seeded
and warmed up, and unavailable metrics are reported as `null`.

### WebSocket actions

Six data-only actions were added, with client types in
[`client/src/introspectionTypes.ts`](../client/src/introspectionTypes.ts:1):

| Action | Server reply | Payload |
|---|---|---|
| `trajectory` | `trajectory` | Bounded `U[t]`/`I[t]`/`S[t]` rows (≤8 stages, ≤64 neurons) |
| `metrics` | `metrics` | Firing rate, sparsity, ISI, histogram per stage |
| `encoding_report` | `encoding_report` | Reconstruction + approximation note for the sample |
| `surrogates` | `surrogate_list` | Selectable surrogate names |
| `surrogate_curve` | `surrogate_curve` | Derivative `x`/`y` samples for one surrogate |
| `benchmark` | `benchmark` | Config, environment, and per-mode results |

All are read-only; precondition failures emit the existing `error` message.
The `stats` action's `system_stats` reply now also carries an additive
`metrics` snapshot (Phase 6); its existing `cpu`/`gpu`/`device` keys are
unchanged.

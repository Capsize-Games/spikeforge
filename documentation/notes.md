# Notes

- Dataset downloads run in an isolated worker process, so the dashboard stays
  responsive: it shows a progress overlay with a live byte counter and a
  **Cancel** button instead of appearing frozen.
- The Bernoulli encoder in `spikegen.rate` is stochastic, so the reported
  spiking percentage and spike patterns vary run to run — expected.
- Larger `num_steps` values (e.g., 100) produce longer, richer animations;
  `subset`/`batch_size` trade dataset coverage for encode speed.
- **Interpreter limitations (Phase 1d).** snnTorch's `Lapicque` uses a
  first-order Euler update while the reference interpreter uses the exact
  zero-order-hold form, and `Synaptic`'s subtract reset carries a small
  residual; both residuals are reported in the `ValidationReport` rather
  than hidden. Extracting NIR graphs from arbitrary external modules via
  `nirtorch` landed in WS-F (`spikeforge-targets extract`; see below).
- **Introspection limitations (Phase 2).** Four behaviours are deliberate.
  (1) `snn.Alpha` is simulation/introspection-only: the installed `nir` has
  no alpha-function primitive that matches its three-state dynamics, so
  exporting an `alpha` stage raises the typed `UnsupportedStageError`
  instead of inventing a lossy mapping. (2) `torch.compile` is opt-in
  because dynamo retraces on the shape-changing neuron state, so eager stays
  the default and `ProductionResult.status` makes a genuine compiled run
  distinguishable from a transparent eager fallback. (3) The decoding
  inverses are approximate: latency is quantised to integer steps and
  saturates at the threshold ceiling for sub-threshold pixels, delta
  integrates to a lower bound that is exact only when each step rises by
  exactly the threshold, and rate is a noisy Bernoulli estimate; `random`
  carries no image signal at all. (4) Upstream's `LSO` surrogate is listed
  because snnTorch exposes it, but applying it raises `TypeError` from
  upstream's wrapper.
- **Dashboard limitations (Phase 3).** (1) The hardware-target picker is
  delivered by the `TargetsPanel`/`BackendRunPanel` (Phase 5/WS-B). (2) Event
  datasets landed in Phase 4 (see the event datasets section above), so that
  walkthrough step now describes the live path. (3) The trajectory viewer,
  metrics, and histogram are gated to educational mode and show an
  explanatory empty state in production.
- **Event limitations (Phase 4).** (1) Training on event datasets is now
  wired in WS-F (`EventTrainingEngine`), but it still needs the `events`
  extra; without `tonic` the engine raises the typed
  `EventsExtraMissingError`. (2) Spatial topologies such as `conv_net` expect
  28x28-like single-channel geometry; a polar or non-square sensor should use
  a feature-input topology or declare an explicit `input_size` (see the
  non-square geometry fold-in). (3) The offline synthetic path is labelled as
  such in `origin`/`description` and is never passed off as a recording.
  (4) Event datasets need the optional `events` extra; without it they show
  as unavailable and the loader raises `EventsExtraMissingError`.
- **Target limitations (Phase 5, updated in WS-B/WS-F and PT-W8).** (1) The
  in-process `reference` target is always available. `norse` and `lava_loihi2`
  have real backends and compile and run when their extras (`norse`, `lava`)
  are installed; without them they report `available: false` and `run` returns
  `status: "unavailable"`. `speck` (`sinabs`), `xylo` (`rockpool`), and
  `spinnaker2` now have isolated probes and backends too, so the test-deploy
  matrix runs them when their SDK is present and otherwise reports
  `available: false` with a named reason; every run is an `estimate`, never a
  device measurement. (2) Substitutions are now *executed* by the
  rewrite executor with a report and a post-rewrite drift check (WS-B), so a
  declared substitute is applied, not merely stated. (3) No on-device runtime
  is wired: `lava_loihi2` requires the Lava SDK and no physical device is
  present, so nothing measures real hardware timing. (4) Graph exchange
  round-trips through this project's version-stamped JSON envelope over NIR's
  own node vocabulary, and `nirtorch` extraction of third-party PyTorch
  modules now ships (WS-F) with a typed error naming any unmappable node.
- **Production limitations (Phase 6).** (1) A manifest makes a run
  *reproducible*, not *bit-exact*: CUDA kernels, thread scheduling/summation
  order, and changed dataset files can still move results, and the manifest
  says so in its `reproducible` block. (2) The registry and its search/diff
  are local and file-based (`MODEL_DIR`); optional TensorBoard/W&B sinks
  (WS-E) forward a run *after* the local manifest is written, and an absent
  backend is a recorded reason, not an error.
  (3) Benchmark records are compared per `(topology, mode)` on `ms/step`,
  `steps/s`, and peak memory; wall-time noise on a shared CPU runner can move
  a metric a few percent, so CI regressions use a threshold (default 10
  percent) rather than bit-exact equality, and `--compare` takes a run id, not
  a label. (4) The metrics registry is per-process and in-memory by default;
  WS-E adds opt-in JSON snapshot persistence (`SPIKEFORGE_METRICS_PERSIST`) but does
  not aggregate across workers. (5) Structured logging is opt-in via `SPIKEFORGE_LOG_JSON` /
  `SPIKEFORGE_LOG_LEVEL`; nothing calls `configure_logging()` automatically, so the
  default log output is unchanged. (6) AMP, gradient checkpointing, truncated
  BPTT, and multi-GPU are all opt-in and default-off, so a default run is
  numerically identical; multi-GPU needs more than one visible CUDA device and
  otherwise reports an honest "unavailable" status. (7) The Docker `cpu`/`gpu`
  profiles are alternate services; only one can own port 8877 at a time.
- **Professionalization limitations (WS-A…WS-F).** These are the honest
  boundaries of the shipped program. The six that shape real use — no measured
  energy, simulation-only `sequence_attn`, simulated-only quantization, the
  single-step ONNX bridge, fetch-on-demand hub weights, and SDK-gated hardware
  backends — are each reasoned about in
  [Implications and boundaries](implications-and-boundaries.md), with *why* they
  exist and *what* they imply. One further boundary is not about any of those
  six: **determinism narrows, not closes, the bit-exactness gap** — hardware
  scheduling and some CUDA kernels remain outside the process's control.

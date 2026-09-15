# Project layout

```
main.py                      Thin entry point: rate pipeline -> exporters
main_encodings.py            Extra tutorial-1 encodings (latency/delta/random)
packages/spikeforge/    Core distribution (pyproject.toml authority)
packages/spikeforge-server/  Server distribution (pulls core)
packages/spikeforge-targets/        Deploy targets distribution (pulls core)
packages/spikeforge-hub/            Model hub distribution (pulls core)
examples/                    Small runnable scripts (see examples/README.md)
spikeforge/
  config.py                  Paths/settings resolved from the environment
  data/                      Dataset registry, loaders, sample access
    datasets.py              Dataset registry (MNIST/Fashion/KMNIST/...)
    dataset_spec.py          DatasetSpec: registry metadata + modality
    data_loader.py           Loader construction with subset reduction
    sample_source.py         Single transformed images/labels for the viewer
    event_loader.py          Tonic (x,y,t,p) stream -> EventSample
    event_errors.py          EventsExtraMissingError for a missing extra
    event_geometry.py        Event frames -> a topology's input geometry
    image_size.py            int / (H, W) sensor geometry normalisation
    sequence_source.py       Synthetic (tokens, label) toy sequence task
  events/                    Event-modality model, dense forms, bridge
    event_sample.py          EventSample: sparse (x, y, t, p) stream
    dense.py                 to_frames / to_voxel -> [T, 2, H, W]
    synthetic.py             Deterministic offline event generators
    event_bridge.py          EventSample -> simulator/NIR spike tensor
    event_source.py          EventSampleSource: tonic or explicit synthetic
    tonic_api.py             The only module importing tonic
  encoding/                  Spike-encoding transforms
    spike_encoder.py         SpikeEncoder: rate/latency/delta/random
    latency_trainer.py       LatencyTrainer (tutorial 2.3)
    delta_trainer.py         DeltaTrainer (tutorial 2.4)
    random_spikegen.py       RandomSpikeGenerator (tutorial 3)
  network/                   Model, inference, and persistence
    spiking_net.py           SpikingNet: fully-connected LIF model
    inference.py             Per-sample prediction + layer activity
    model_store.py           Save/load/list/delete model checkpoints
    model_search.py          Filter checkpoints by stored metadata
    model_diff.py            Classify metadata changes between checkpoints
    hidden_frames.py         Per-step hidden-layer frames for animation
  training/                  Training loop and its collaborators
    trainer.py               SNNTrainer: MNIST loading + rate coding
    logger.py                SNNTrainerLogger: diagnostic prints
    training_engine.py       TrainingEngine: train loop yielding metrics
    checkpoint_mixin.py      Checkpoint save/restore behaviour
    encoding_mixin.py        Raw-pixel / spike input encoding
    eval_mixin.py            Periodic held-out evaluation
    scaleup_mixin.py         Opt-in AMP/grad-ckpt/BPTT/multi-GPU wiring
    amp_controller.py        Resolve and apply autocast/GradScaler AMP
    multi_device.py          DataParallel decision + honest status
    topology_mixin.py        Resolve the spec/module for a training run
    event_engine.py          EventTrainingEngine: steps over an event source
    event_batches.py         Batch an event stream into [T, B, ...] frames
  topology/                  Topology specs, presets, module builder
    spec.py                  TopologySpec: stages + edges (+ chain helpers)
    presets.py               fc_legacy / fc_small / conv_net / recurrent_net
    sequence_presets.py      sequence_mlp / sequence_attn demo presets
    sequence_stages.py       embedding / norm / attention factories
    attention.py             Single-head self-attention module
    multihead_attention.py   Multi-head self-attention module
    positional_encoding.py   Deterministic sinusoidal positional encoding
    sum_pool.py              SumPool2d used by the Loihi substitution
    stage_module.py          StageModule: one (input, state) step per stage
    stage_modules.py         Module-kind factory table
    registry.py              name -> builder; build_topology/resolved_params
    builder.py               build_module(spec) -> snnTorch StageModule
  neurons/                   Neuron registry + canonical NIR param contract
    registry.py              NEURONS: name -> factory (incl. alpha); build()
    alpha.py                 snn.Alpha handler (simulation/introspection only)
    spike_grad.py            Optional `surrogate` param -> spike_grad callable
  simulator/                 The single temporal loop and trajectory capture
    execution.py             execute(...): one loop, both execution modes
    runner.py                run(module, spikes, mode=...) -> Trajectory
    compiled_step.py         Opt-in torch.compile wrapper + eager fallback
    production.py            run_production(...) -> ProductionResult
    trajectory.py            Per-stage S[t] / U[t] / I[t] traces
    grad_policy.py           Opt-in grad checkpoint + truncated BPTT
    parallel_runner.py       DataParallel wrapper around the temporal loop
  introspection/             Educational mode: metrics, codings, surrogates
    metrics.py               trajectory_metrics(...) -> JSON-able metrics
    firing_rate.py           Per-stage firing rate
    sparsity.py              Per-stage sparsity
    isi.py                   Inter-spike-interval statistics
    histogram.py             Per-neuron firing-rate histograms
    encoding.py              encoding_report(...) + reconstruction
    decoding.py              Approximate per-coding image inverses
    surrogate.py             Surrogate registry + derivative curve
    comparison.py            compare_neurons(...) across every kind
  benchmark/                 Production-mode timing and memory harness
    config.py                BenchmarkConfig fixture
    harness.py               run_benchmark(...) -> JSON-able report
    timing.py                Warmup/repeat call timing
    memory.py                CUDA/RSS/tracemalloc snapshots
    store.py                 BenchmarkStore: file-based JSON run records
    suite.py                 run_suite(...): multi-topology run + metadata
    compare.py               compare_runs(...) + regression exit code
    cli.py                   save / list / compare CLI (spikeforge-benchmark)
    __main__.py              python -m spikeforge.benchmark
  observability/             Opt-in structured logging + metrics
    logging_setup.py         configure_logging / reset_logging (reversible)
    json_formatter.py        JsonFormatter: one JSON object per log line
    registry.py              MetricsRegistry: counters/gauges/timers
    timer.py                 Context-manager timer recording into a registry
    metrics.py               Shared registry + snapshot/JSON helpers
    snapshot.py              MetricSnapshot: metrics + run id + timestamp
    store.py                 SnapshotStore: write/read metric snapshots
    persistence.py           Opt-in flush/load hook (SPIKEFORGE_METRICS_PERSIST)
  nir_bridge/                NIR export, interpreter, validation, interop
    api.py                   The only module importing nir/nirtorch
    exporter.py              to_nir(spec, module); graph_summary(...)
    interpreter.py           NirInterpreter: runs a graph without snnTorch
    post_node.py             PostNode: the optional per-node transform hook
    validator.py             validate(...) -> ValidationReport
    drift.py                 Error metrics between two trajectories
    serialization.py         save_graph / load_graph (version-stamped JSON)
    array_codec.py           Tagged numpy encoding for an exact reload
    ingest.py                load_external / interpret_graph / interpret_file
    roundtrip.py             Persist + reload + compare graph fidelity
    extract.py               nirtorch: third-party torch module -> NIR graph
    torch_map.py             nn.Linear/nn.Flatten -> NIR for extraction
    mapper.py                Stage -> NIR node(s); consults the tables below
    node_builders.py         Builder table for mapped module kinds
    neuron_nodes.py          Neuron-kind -> NIR node(s); alpha precedent
    stage_builders.py        Builder table for the new stage kinds
    stages_unmappable.py     kind -> honest reason export cannot map it
  tracking/                  Reproducibility: manifest, hash, seed, versions
    manifest.py              ReproducibilityManifest: config/seed/history
    config_hash.py           Canonical-JSON SHA-256 of the run config
    seed.py                  set_seed: Python/torch/CUDA deterministic seed
    versions.py              Library versions recorded in a manifest
    determinism.py           enable_deterministic + bit_exactness_check
    sink.py / sinks.py       Sink protocol + active-sink resolution
    tensorboard_sink.py      TensorBoard SummaryWriter sink (tracking extra)
    wandb_sink.py            Weights & Biases sink (tracking-wandb extra)
    sink_probe.py            Isolated tensorboard / wandb probes
  onnx_bridge/               Optional ONNX import/export (onnx extra)
    api.py                   The only module importing onnx/onnxruntime
    export.py                One-step topology export with spec metadata
    import_onnx.py           Map ONNX ops to stage kinds, or name the op
    roundtrip.py             Export + re-import fidelity check
  cli/                       Headless commands
    verify.py                export / validate + the shared subcommands
    records_cli.py           records list / diff / manifest (spikeforge-records)
    backend_cli.py           rewrite / run backend commands
    extract_cli.py           extract a torch module via nirtorch
    onnx_cli.py              onnx-export / onnx-import / onnx-roundtrip
    fixture.py               Offline spike fixtures shaped per topology
  exporters/                 matplotlib/GIF/MP4 output -> build/
    exporter.py              Exporter base + build/ output resolution
    plot_utils.py            shared fig/GIF helpers
    *_exporter.py            per-visual exporters
  runtime/                   Compute environment
    device.py                CPU/GPU selection + auto benchmark
    execution_mode.py        Educational/Production execution flag
    system_stats.py          CPU RAM / GPU VRAM snapshots
spikeforge_targets/                 Deployment targets, energy, event runtime
                               (ARCH-0001 Phase 3; distribution spikeforge-targets;
                               extracted to capsize-games/spikeforge-targets)
  target_spec.py             TargetSpec: support, substitutions, constraints
  catalog.py                 Built-in targets (reference + placeholders)
  registry.py                name -> spec; live availability lookup
  primitives.py              EMITTED_PRIMITIVES: the mapper's NIR vocabulary
  probe.py                   The only module importing a backend SDK
  capability_matrix.py       classify(...) -> per-node CapabilityMatrix
  matrix_result.py           Buckets, counts, and deployable() semantics
  node_view.py               Node name/kind view of a spec or graph
  substitution.py            Substitution record
  rewrite.py                 rewrite(graph, target) -> RewriteResult
  rewrite_report.py          JSON-able applied/skipped/unfixable deltas
  substitute_ops.py          One rewrite function per declared substitution
  rewrite_drift.py           Post-rewrite drift check vs. the original
  quantize.py                Apply a target's declared weight quantization
                             and run the drift check (optionally under a
                             simulated activation/membrane scheme)
  quantize_schemes.py        none / weight_int8 / weight_uint8 schemes
  activation_quant.py        Serving-side activation/membrane quantizer
  activation_quant_graph.py  The same grid as an interpreter post_node hook
  activation_quant_keys.py   Which node tensors quantize, under what key
  activation_quant_records.py Per-tensor range/error records
  fixed_point.py             The shared symmetric fixed-point snap
  report.py                  deployment_report(...) -> JSON
  summary.py                 Availability-annotated registry summaries
  backends/                  Executable backends behind one isolated probe
    __init__.py              compile_run(...) -> BackendResult (never raises)
    reference_backend.py     In-process NIR interpreter (always available)
    norse_backend.py         Norse PyTorch simulator (norse extra)
    lava_backend.py          Lava / Loihi 2 path (lava extra)
    compare.py               Backend result vs. reference comparison
  energy/                    Declared cost tables + energy accounting
    cost_table.py            Load/validate a target's declared costs
    target_costs.py          Bundled per-target cost-table lookup
    accounting.py            account(...) -> EnergyReport (estimate: true)
    report.py                JSON-able report assembly
    costs/*.json             reference / norse / lava_loihi2 / ... tables
    cli.py                   spikeforge-energy entry point
  event_runtime/             Sparse / event-driven execution path
    spike_view.py            SparseSpikes: indices/values per frame
    ops.py / sparse_step.py  Event-driven ops for one temporal step
    counters.py              SynapticCounter: SOP / MAC / AC / timesteps
    sparse_runner.py         sparse_run(...) -> SparseResult
    dense_compare.py         Sparse-vs-dense readout parity check
  cli/
    target_cli.py            targets / roundtrip / test-deploy / ingest
    deploy_cli.py            deploy (with --activation-quantization)
spikeforge_hub/                     Curated model hub (ARCH-0001 Phase 4; distribution
                               spikeforge-hub; extracted to capsize-games/spikeforge-hub)
  models.json                Bundled catalog (10 entries, five frameworks)
  entry.py / catalog.py      Validate and list/search the catalog
  probe.py / hf_api.py       Isolated huggingface_hub probe + live access
  cache.py                   HUB_CACHE_DIR resolution + entry paths
  download_cli.py            Isolated child-process downloader + verify
  downloads.py               Async download manager: progress + cancel
  verify.py                  sha256 / size verification
  inspect.py / compat.py     Structure report + exact/mappable/incompatible
  weight_map.py              Load compatible weights into a preset module
  import_model.py            inspect -> compat -> promote into MODEL_DIR
  cli.py                     spikeforge-hub entry point (spikeforge-hub distribution)
server/
  app.py                     FastAPI app + WebSocket endpoint
  handlers.py                Inbound message routing (encode + train)
  protocol_handlers.py       Router for the NIR + introspection actions
  nir_handlers.py            nir_export / nir_validate handlers
  target_handlers.py         targets / deployment_report handlers
  target_payloads.py         JSON payloads for the target actions
  backend_handlers.py        deploy_run handler (compile + run)
  backend_payloads.py        JSON payloads for the backend actions
  energy_handlers.py         energy_report handler
  energy_payloads.py         JSON payloads for the energy actions
  hub_handlers.py            hub_list/search/download/inspect/import
  hub_payloads.py            JSON payloads for the hub actions
  hub_downloads.py           Hub download manager singleton + emitter
  model_handlers.py          model_search / model_diff handlers
  model_payloads.py          JSON payloads for the registry actions
  introspection_handlers.py  trajectory / metrics / encoding / surrogate
  introspection_payloads.py  JSON payload builders for introspection
  payloads.py                Model-list / model-load / NIR payload builders
  messages.py                Outbound WS message helpers
  encoder.py                 Config -> encoder engine (JSON payloads)
  event_engine.py            Config -> event engine (ON/OFF frames, raster)
  training.py                Threaded training bridge -> asyncio queue
  session.py                 Per-connection state (engine/train/stream)
  schemas/                   Pydantic WS message/config schemas
    encode_config.py         EncodeConfig + CodingType
    train_config.py          TrainConfig
    client_message.py        ClientMessage
    server_message.py        ServerMessage
client/                      Vite + React + TypeScript dashboard
  src/                       app shell, theme, types, WS/training hooks
  src/hooks/                 viewer state, encode config, model/tour actions
  src/components/            controls, charts, canvas panels, plus the hub /
                             backend / energy / stage-editor panels
  src/hubTypes.ts            Typed hub payloads
  src/energyTypes.ts         Typed energy payloads
  src/tour/                  guided-walkthrough lessons + target highlighting
  src/styles/                split stylesheet (base/sections/controls/mode/
                             panels/analysis/targets/tour/...)
```

Code is kept tidy by construction: modules are grouped into focused
subpackages, almost every Python file is under 250 lines (the one exception,
[`server/messages.py`](../server/messages.py:1), is tracked as style debt), every
Python function stays under 20 lines, and each class lives in its own file.

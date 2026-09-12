# Sequence primitives and per-stage neurons (WS-C)

### Per-stage heterogeneous neurons

Every neuron stage can now choose its own kind, params, and surrogate. The
presets accept additive `neurons` (per-stage kind) and `stage_params`
(per-stage `beta`/`threshold`/`reset`/`surrogate`) maps, and `TrainConfig` gains
additive `stage_neurons`/`stage_params`. A default build is byte-identical to
before, and `fc_legacy` keeps its `_fc1/_lif1/_fc2/_lif2` contract.

### New stage kinds

[`topology/kinds.py`](../spikeforge/topology/kinds.py:8) grows `conv1d`,
`maxpool1d`, `maxpool2d`, `embedding`, `layer_norm`, `batch_norm`, `dropout`,
`positional_encoding`, `attention`, and `multihead_attention`, each with a
module factory and an explicit NIR contract — `mapped`, `passthrough`
(`dropout` is identity at inference), or `unexportable`.

### `sequence_mlp` and `sequence_attn`

- `sequence_mlp` is built only from NIR-mappable kinds over a `[T, B, L, D]`
  sequence and **validates end to end**. Its neurons default to `reset="zero"`,
  rendering a single `nir.LIF` with no `Delay`, so it is runnable by the Norse
  target too.
- `sequence_attn` is the spiking-transformer-shaped demo (`embedding` →
  `positional_encoding` → `multihead_attention` → `layer_norm` → `linear` →
  neuron). The installed `nir` has no embedding, attention, or normalisation
  primitive, so export raises the typed `UnsupportedStageError` naming the
  first unexportable stage. It stays available for simulation and
  introspection — the honest `alpha` precedent, applied to stages.

The toy token task
([`data/sequence_source.py`](../spikeforge/data/sequence_source.py:1))
supplies `[T, B, L, D]` frames, and the client
[`StageNeuronEditor`](../client/src/components/StageNeuronEditor.tsx:1) edits the
per-stage configuration. This enables sequence/attention *experimentation*, not
production LLM training.

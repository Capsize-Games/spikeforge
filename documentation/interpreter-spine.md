# Interpreter spine (Phase 1)

Phase 1 lifts the models onto the Neuromorphic Intermediate Representation
([NIR](https://github.com/neuromorphs/NIR)) and proves the translation. A
model is declared once as a `TopologySpec` and rendered twice: into the
snnTorch module that trains, and into the `nir.NIRGraph` that exports.

### Topology presets

Pick a topology by name — the training server takes a `topology` field and
the CLI takes `--topology`:

| Preset | Shape |
|---|---|
| `fc_legacy` | The original two-layer FC LIF `SpikingNet` (default) |
| `fc_small` | Small FC LIF with an explicit flatten entry stage |
| `conv_net` | Conv/pool feature extractor with a linear LIF readout |
| `recurrent_net` | FC LIF with a one-step delayed feedback edge |

`fc_legacy` stays the default and keeps `SpikingNet`'s
`_fc1`/`_lif1`/`_fc2`/`_lif2` state-dict keys, so existing checkpoints load
and infer unchanged.

### Neuron registry

`spikeforge/neurons/` maps a neuron name to its snnTorch factory and
its canonical NIR parameter contract. The registry ships `leaky`,
`lapicque`, `synaptic`, and `recurrent` (RLeaky) neurons.

### NIR export, interpretation, and validation

- `to_nir(spec, module)` exports a JSON-able `nir.NIRGraph`; `graph_summary`
  describes its nodes and edges.
- `NirInterpreter` executes the exported graph independently of snnTorch, so
  `validate(spec, module, spikes)` reports a genuine `ValidationReport`
  (per-layer max/mean/relative error plus spike agreement, and an overall
  `within_tolerance` flag). Every shipped preset validates with bit-exact
  spikes.
- All `nir`/`nirtorch` imports are confined to `nir_bridge/api.py`.

### Verify CLI

Headless `export` and `validate` commands. `validate` exits non-zero when a
report falls outside tolerance, so it doubles as a CI gate:

```bash
python -m spikeforge.cli.verify export --topology conv_net
python -m spikeforge.cli.verify export --topology fc_legacy --out g.json
python -m spikeforge.cli.verify validate --topology conv_net
python -m spikeforge.cli.verify validate --topology recurrent_net
```

### WebSocket actions

The server answers two new client actions: `nir_export` returns the graph
summary for the active or configured topology, and `nir_validate` returns a
drift report for the active model.

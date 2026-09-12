# Event-driven runtime and energy accounting (WS-D)

### Sparse runner

[`event_runtime.sparse_run()`](../spikeforge_targets/event_runtime/sparse_runner.py:1)
is a parallel, training-free inference path that propagates spike events
instead of dense MACs. It returns a `SparseResult` with the same readout
contract as the dense `Trajectory`, and
[`dense_compare`](../spikeforge_targets/event_runtime/dense_compare.py:1) proves
parity within tolerance. The dense path stays the untouched default.

[`SynapticCounter`](../spikeforge_targets/event_runtime/counters.py:1) tallies **SOP**
(synaptic ops), **MAC** (dense baseline), **AC**, and timesteps; for a sparse
input `SOP < MAC` by the active-spike ratio.

### Declared per-target cost tables

Each target carries a declared cost table under
[`energy/costs/`](../spikeforge_targets/energy/costs/reference.json:1) (`reference`,
`norse`, `lava_loihi2`, `spinnaker2`, `speck`, `xylo`), giving energy per
SOP/MAC/AC and latency per timestep. Every table is `"measured": false` and
carries its `source`.

### `spikeforge-energy`

```bash
spikeforge-energy account --topology conv_net --target reference --sparse
spikeforge-energy report  --topology conv_net --target reference
```

`account` prints the report; `report` adds the sparse-vs-dense parity block.
The `energy_report` WebSocket action and the benchmark `--energy` path carry
the same payload; [`EnergyPanel`](../client/src/components/EnergyPanel.tsx:1)
renders it with a prominent estimate badge.

### The estimate-vs-measured rule

Every report carries `estimate: true` and a `basis` (`"declared cost table"`,
or `"unavailable"` when a target has no table — never a fabricated number).
Numbers are **estimates from declared tables**, never presented as measured,
until a device reports its own timing.

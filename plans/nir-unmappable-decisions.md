# NIR-unmappable stage decisions

Nine stage kinds have no NIR rendering in the installed `nir` package, so
export raises the typed `UnsupportedStageError` naming the kind
([`stages_unmappable.py`](spikeforge/nir_bridge/stages_unmappable.py),
[`neuron_nodes.py`](spikeforge/nir_bridge/neuron_nodes.py)). This record
gives each kind a verdict — `await-upstream-primitive`,
`decompose-into-existing-NIR`, or `permanently-simulation-only` — with
the reason and, for decompositions, the exact construction. The error
message points here so a caller hitting it learns the status.

This is broader than the C runtime's own kind gap (`Linear`/`Conv1d`/
`LIF`/`IF`/`CubaLIF`, which export fine but the C loader rejects): these
stages fail before a graph even exists. The C side is profiled in
[`plans/nir-c-runtime-profile.md`](plans/nir-c-runtime-profile.md).

| Stage kind | Verdict | Reason | Follow-up |
|---|---|---|---|
| `embedding` | decompose-into-existing-NIR | A row gather is exactly a one-hot product: expand token ids to one-hot at ingest, then `Affine(weight=table, bias=zeros)`. | #169 |
| `maxpool1d` | await-upstream-primitive | No max or comparison primitive exists. `SumPool`+`Threshold` is exact only for binary inputs, which the stage does not guarantee. | — |
| `maxpool2d` | await-upstream-primitive | No max or comparison primitive exists. `SumPool`+`Threshold` is exact only for binary inputs, which the stage does not guarantee. | — |
| `layer_norm` | await-upstream-primitive | Per-sample mean and variance are runtime reductions with no primitive, and input-dependent statistics cannot fold into static weights. | — |
| `batch_norm` | decompose-into-existing-NIR | Eval-mode statistics are frozen: fold per-channel `gamma/sqrt(var+eps)` and `beta-gamma*mean/sqrt(var+eps)` into one `Affine`. Exact for `[B, C]` inference inputs only. | #170 |
| `positional_encoding` | await-upstream-primitive | The fixed table varies along the sequence dim and slices to the input length, so it fits no static weight; there is no Constant node. | — |
| `attention` | await-upstream-primitive | The Q/K/V projections already map to `Linear`/`Affine`; the score matmul, softmax, and weighted sum need dynamic-matmul, exp, and reduction primitives with no exact decomposition. | — |
| `multihead_attention` | await-upstream-primitive | Per-head Q/K/V projections already map to `Linear`/`Affine`; the score matmul, softmax, weighted sum, and head mixing have no exact decomposition. | — |
| `alpha` | await-upstream-primitive | Three coupled states (`syn_exc`, `syn_inh`, `mem`) with spike-gated reset across all of them: zero-reset needs a dynamic multiplier, subtract-reset terms contain dynamic products, `LI` has no reset, and `LIF` hides its analog state. | — |

Until the follow-ups land, every kind above stays simulation-only:
available for simulation and introspection, rejected at export.

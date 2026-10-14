# NIR C-runtime export profile

The Python reference interpreter implements 14 node kinds, but the
`nir-c-runtime` C loader implements only 11: a graph using `Linear`,
`Conv1d`, `LIF`, `IF`, or `CubaLIF` fails loading with an `unsupported
node kind` error. This profile declares the exact supported set as data
so exports can be gated spikeforge-side instead of failing at the C
loader. Stage kinds with no NIR rendering at all are decided separately
in [`plans/nir-unmappable-decisions.md`](plans/nir-unmappable-decisions.md).

## Supported kinds

Single-sourced from `SUPPORTED` in
[`c_runtime.py`](spikeforge_targets/c_runtime.py), which names the
tracked `nir-c-runtime` commit (currently `7348c82`) next to the
constant. The validator and this table both reference it; re-audit
`graph.h`, `graph.c`, `ops.c`, and `runtime.c` when the commit moves.

| NIR kind | C loader support |
|---|---|
| `Input` | Entry node, params ignored. |
| `Output` | Pass-through readout, params ignored. |
| `Conv2d` | Dense-only weights `[out, in, kh, kw]`; bias optional; stride, padding, and dilation as `[h, w]` pairs. |
| `Affine` | Weights `[out, in]` with a required bias `[out]`. |
| `Flatten` | Ignores its dims and flattens every dim, so exports must set `start_dim=1, end_dim=-1`. |
| `AvgPool2d` | Kernel, stride, and padding as `[h, w]` pairs. |
| `SumPool2d` | Kernel, stride, and padding as `[h, w]` pairs. |
| `LI` | `tau` (positive), `r`, `v_leak`. |
| `Threshold` | Scalar threshold. |
| `Scale` | Scalar scale. |
| `Delay` | Length ignored; the runtime hardcodes one step, so exports must set `delay=1`. |

## Validator

[`c_runtime_validate.py`](spikeforge_targets/c_runtime_validate.py)
exposes `validate_for_c_runtime`, which accepts a `TopologySpec`, an
exported NIR graph, or a summary mapping and returns a report naming
every incompatible node, edge, and structural assumption. Each check
mirrors one C-loader behavior: the kind table, Affine-bias and
weight-layout parsing (raw or per-tensor int8/uint8 envelope),
conv/pool pair shapes, the dense-only convolution, Flatten coverage,
the one-step delay, positive `tau`, edge references, per-node inputs,
single input/output, and Delay-broken cycles.
`check_for_c_runtime` raises the typed `CRuntimeUnsupportedError` on
the same violations for refuse-early export paths.

## Profile, not a registry entry

The profile ships as a `TargetSpec` that `classify` accepts, so a
capability row is one call away, but it stays out of
`builtin_targets()`: every registered target is required to have an
executable simulator backend, and none is wired for the C runtime.

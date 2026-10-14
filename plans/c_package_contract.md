# C package contract

How a spikeforge bundle reaches a torch-free target. A `.spkf` bundle needs
Python and torch (a pickled `weights.pt` and a Python-envelope graph), so
deployment to C goes through a derived plain-JSON artifact instead. This
document owns the exact JSON the C loader consumes, so the two sides cannot
drift silently. The consumer is `nir_graph_load` in `nir-c-runtime`'s
`src/graph.c`.

## Producers

- `DeploymentBundle.export_c_package(dir)` (`spikeforge/serving/c_package.py`)
  writes a float package: the NIR graph in `graph_summary` shape with the
  bundle's real weights as plain nested float arrays. Compressed bundles
  export their dequantized floats. The `.spkf` format itself is unchanged.
- `export_quantized_summary(graph_or_spec, scheme)`
  (`spikeforge_targets/quantized_summary.py`) returns the same shape with
  every floating `weight` replaced by a quantized envelope. The `none`
  scheme returns the float summary itself, so it is byte-comparable to the
  existing float path.
- Grids come from `spikeforge/compression/grid.py`, shared by the bundle
  codec and the drift simulation, so all three agree on every
  `(scale, zero_point)` pair by construction.

## Graph shape

Top level `{"nodes": [...], "edges": [...]}`. A node is
`{name, kind, params}`; an edge is `{source, target, delayed}` (the C loader
reads `source`/`target` and ignores `delayed`). Eleven node kinds load;
anything else fails in C with a named error, and `export_c_package` refuses
it first with `CExportError` naming the kinds:

`Input`, `Output`, `Conv2d`, `Affine`, `Flatten`, `AvgPool2d`, `SumPool2d`,
`LI`, `Threshold`, `Scale`, `Delay`.

Per-kind parameters the C loader reads (extra parameters are ignored):

- `Conv2d`: `weight` (4-D), optional `bias` (1-D), `stride`/`padding`/
  `dilation` pairs.
- `Affine`: `weight` (2-D) and `bias` (1-D); the bias is required.
- `AvgPool2d`/`SumPool2d`: `kernel_size`/`stride`/`padding` pairs.
- `LI`: `tau`, `r`, `v_leak`. `Threshold`: `threshold`. `Scale`: `scale`.
- `Input`, `Output`, `Flatten`, `Delay`: no parameters read.

## Quantized envelope

A quantized weight is `{data, dtype, scale, zero_point}`, decoded as
`(code - zero_point) * scale`:

- `data`: nested integer lists with the same nesting as the plain weight.
- `dtype`: `"int8"` (symmetric, zero point 0, codes in [-127, 127]) or
  `"uint8"` (asymmetric, codes in [0, 255]).
- `scale`: a number greater than 0 and finite; `zero_point`: a number.
  The C loader rejects a missing, non-positive, or non-finite scale.

Only `weight` tensors are enveloped, exactly the tensors the drift
simulation quantizes; biases and every other parameter stay the plain
`graph_summary` values. Envelopes decode to the simulation's quantized
weights, so the C readout matches the simulated-quantization drift run.

Degenerate tensors keep a loadable, exact grid:

| Tensor | Grid `(scale, zero_point)` | Codes | Decodes to |
|---|---|---|---|
| int8 all-zero or empty (peak <= 1e-12) | `(1.0, 0)` | all 0 | zeros |
| uint8 constant `c > 0` | `(c, 0)` | all 1 | `c` |
| uint8 constant `c < 0` | `(-c, 1)` | all 0 | `c` |
| uint8 constant `0` or empty | `(1.0, 0)` | all 0 | `0` |
| uint8 near-constant (span <= 1e-12) | single level at the midpoint | per sign above | within 1e-12 of every value |

Non-finite weights are refused with `QuantizationError`: no finite grid
can represent them. Empty dimensions do not survive nested-list JSON
(`[]` carries no shape); the float summary has the same property.

## Package layout

`export_c_package` writes two files into the target directory:

- `graph.json`: the graph summary the C loader parses.
- `manifest.json`: `format` (`"spikeforge-c-package"`), `version` (1),
  `created_at`, `graph` (`"graph.json"`), node/edge counts, and `source`:
  the source bundle's path, its file SHA-256 (null for an in-memory
  bundle), and the SHA-256 of its canonical manifest JSON.

## Tolerances

Float weights round-trip through JSON bit-exactly (float32 values are
exact in doubles), and the quantized envelopes decode to the same floats
the drift simulation runs. Measured on a `conv_net` bundle over 6 steps
of 28x28 input, silent and firing, float/int8/uint8/compressed: C readout
matches the reference readout with max abs diff 0.0 in every case. The
acceptance bar stays `1e-3` readout max-abs, the same bar `nir-c-runtime`
uses in its own end-to-end test, leaving margin for larger graphs.

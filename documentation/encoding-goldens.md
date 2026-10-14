# Encoding golden vectors

The committed fixtures under `tests/fixtures/encoding_goldens/` pin the
exact input/output behavior of the four spike codings and the two
streaming pre-steps, so a future C encoder has an oracle to verify
against. This page specifies the math and the random-number procedure
each fixture pins, precisely enough to reimplement without reading the
torch or snntorch source. The fixtures record current behavior, bugs
included: a behavior change is a new issue, not a silent regeneration.

## Fixture layout

One JSON file per case: `rate.json`, `latency.json`, `delta.json`,
`random.json`, `delta_over_window.json`, `window_normalize.json`. Each
file carries its case name, a `generator` block (the generating script,
the spikeforge commit, and the torch, snntorch, and Python versions),
the frozen spec plus its digest, the procedure in words, and the input
and output tensors as shapes with row-major `values`. Spike tensors
store `0`/`1` integers; float tensors store decimals that are exactly
representable in float32. `tests/test_encoding_goldens.py` re-encodes
every fixture input and compares bit-for-bit.

## Common setup

All four coding fixtures encode the same image through
[`preprocess.encode`](../spikeforge/serving/preprocess.py:1) with an
explicit [`EncodeSpec`](../spikeforge/serving/encode_spec.py:1): batch 1,
one channel, 2 rows by 4 columns, pixels
`0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0` in row-major order,
`num_steps` 8. Every value is exactly representable in float32 and lies
in `[0, 1]`, which the latency coding requires. All tensors are
contiguous float32. The encoder flattens the trailing image axes in
row-major order, so each output is `[8, 1, 8]`: time first, then batch,
then the 8 pixels in input order. Outputs are float32 `0.0`/`1.0`
spikes.

## Rate

Bernoulli rate code at unit gain. Each pixel `x` is clamped to `[0, 1]`,
repeated across the 8 time steps, and then each of the 64 outputs is an
independent Bernoulli draw with `p` equal to the clamped pixel: spike
(`1.0`) with probability `p`, else `0.0`. The draws consume the RNG
stream in row-major order over the `[8, 1, 8]` tensor, one uniform per
element (see "Random numbers" below). The fixture pins seed 1234 and
holds 23 spikes. Out-of-range pixels do not raise; they clamp.

## Latency

Linear, normalized RC latency code with `tau` 5.0, `threshold` 0.01,
`clip` false, `normalize` true, `linear` true. Any pixel outside
`[0, 1]` raises instead of encoding. Spike times first: with
normalization on, the time constant is replaced by `num_steps - 1`
(the configured `tau` is ignored), and each pixel `x` fires at
`min(7 * (1 - x), 6.93)` — the cap is `7 * (1 - 0.01)`, so
below-threshold pixels saturate at the last step rather than clipping.
Each time is rounded to the nearest step with round-half-to-even
(`torch.round` banker's rounding: the fixture's `x = 0.5` lands on
exactly 3.5 and fires at step 4), and each pixel emits exactly one
spike at its rounded step. The fixture fires at steps
`7, 6, 5, 4, 4, 3, 2, 0` for the eight pixels in order. No RNG is
consumed. A rounded time past the last step raises; with these settings
the maximum is 6.93, so that cannot trigger.

## Delta

Per-pixel intensity ramp through a threshold detector, with
`delta_threshold` 4.0 (percent). The ramp is the float32 linspace from
0 to 1 inclusive over the 8 steps; each pixel's ramp is its pixel value
times the ramp. The detector compares each step against the previous
one, padding the first step with zeros, and spikes wherever the rise
reaches `delta_threshold / 100 = 0.04`. Because the ramp starts at 0,
the first step is always silent; rises are constant along the ramp, so
each pixel either never fires (pixels 0.0, 0.125, 0.25) or fires at
every later step (pixels 0.375 and up, 7 spikes each, 35 total). No RNG
is consumed. Negative changes never spike: the off-spike detector is
hard off.

## Random

Noise baseline with `random_scale` 0.5 and seed 1234. The input pixels
are ignored — only the input shape matters — and the output is
`[8, 1, 8]` from a two-draw RNG procedure: first one uniform `[0, 1)`
per output element in row-major order, then one Bernoulli draw per
element with `p` equal to the uniform times `random_scale`, clamped to
`[0, 1]`. The fixture holds 18 spikes. A different input with the same
shape encodes to the identical tensor; the golden test asserts that.

## `delta_over_window`

First difference along the window time axis, from
[`streaming/encoding.py`](../spikeforge/streaming/encoding.py:1). The
first sample of every window is zeros (it has no predecessor) and
sample `t` becomes `x[t] - x[t-1]`, in float32, over the trailing
`[L, D]` axes. The fixture differences one `[1, 4, 3]` window of
exactly representable values. No RNG is consumed.

## `WindowSpec.normalize`

Z-scoring with a frozen train-fitted spec, from
[`streaming/window_spec.py`](../spikeforge/streaming/window_spec.py:1).
Each value becomes `(x - mean[c]) / std[c]` in float32, broadcasting
the per-channel statistics over the leading axes. The fixture spec has
length 4, stride 2, 2 channels, means `(0.5, -1.0)`, standard
deviations `(0.25, 2.0)`, and digest
`d85c2557…4d27b8f`; it normalizes one `[1, 4, 2]` batch to
`0, 0, 1, 1, 2, 2, -2, -2`. No RNG is consumed.

## Random numbers

Seeded paths call `torch.manual_seed(seed)` on the CPU generator
immediately before dispatch, with no other RNG use in between — both
[`preprocess.encode`](../spikeforge/serving/preprocess.py:1) and
`encode_windows` seed this way when the spec pins `random_seed`. The
`random` coding then re-seeds with the same seed inside the encoder
before its own draws, so its stream starts from the seed regardless.

The CPU generator is a stock Mersenne Twister (MT19937) initialized
with the seed modulo 2^32. Each float32 uniform is the low 24 bits of
the next MT19937 output divided by 2^24, consumed in row-major
(tensor-storage) order; each Bernoulli draw takes the next uniform `u`
and spikes if and only if `u < p`, with strict comparison (`p = 1`
always spikes, `p = 0` never does). This chain was verified bit-for-bit
against an independent MT19937 implementation over 1300 draws,
including across a twist boundary.

PyTorch does not promise this stream across versions, so the fixtures
also pin the toolchain: torch 2.14.1+cpu (the `requirements.txt` pin),
snntorch 1.0.0, CPU, contiguous float32 tensors. If a torch bump moves
the stream, the golden tests fail and the fixtures must be deliberately
regenerated (below), never silently updated.

## Frozen but inert fields

`EncodeSpec` freezes `gain` (default 1.0) and `off_spike` (default
false), but the current encoder ignores both under every coding: rate
hardcodes gain 1.0, delta hardcodes the on-spike-only detector. The
spec reports them through `unsupported()` rather than dropping them.
The fixtures record this behavior as specified — a port reproduces the
inertness, and any decision to honor these fields is a separate issue.

## Port conformance

A port conforms when it reproduces each fixture output from the
fixture input and spec. Spike tensors (all four codings) must match
bit-for-bit. The two float pre-steps are correctly rounded IEEE-754
elementwise arithmetic, so they must also match bit-for-bit on any
IEEE-754 platform; a mismatch there is a bug, not tolerance. For the
stochastic codings, a port that implements the MT19937 stream above
must reproduce the pinned seed-1234 outputs exactly; a port with its
own RNG cannot match them and should instead verify the Bernoulli
procedure statistically plus the deterministic codings exactly.

## Regeneration

```bash
PYTHONPATH=. python scripts/generate_encoding_goldens.py
pytest tests/test_encoding_goldens.py
```

Regenerate only when an accepted encoder or torch change moves the
outputs: inspect the fixture diff, confirm it shows exactly the
intended change, and commit the fixtures with the behavior change that
caused them.

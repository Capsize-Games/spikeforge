# SHD and SSC: every timestamp becomes INT64_MIN under NumPy 2 (all timing lost, silently)

**Affects:** `tonic.datasets.SHD`, `tonic.datasets.SSC` (both via `HSD`)
**Confirmed on:** tonic 1.4.3 with NumPy 2.5.2 / Python 3.13 / Linux, against
the real SSC and SHD files. `hsd.py` carries the same conversion verbatim in
1.6.0, and the promotion behaviour below is a property of NumPy rather than of
tonic, so 1.6.0 is expected to be affected identically — though the end-to-end
run above was on 1.4.3
**Severity:** silent data corruption — training succeeds and reports plausible accuracy

## Summary

`HSD.__getitem__` converts the files' timestamps from seconds to microseconds:

```python
# tonic/datasets/hsd.py
events = make_structured_array(
    file["spikes/times"][index] * 1e6,
    file["spikes/units"][index],
    1,
    dtype=self.dtype,          # [("t", int), ("x", int), ("p", int)]
)
```

The Heidelberg HDF5 files store `spikes/times` as **`float16`**. Under NumPy 2's
NEP 50 promotion, `float16_array * python_float` stays `float16` — and `1e6`
does not fit in `float16` at all (max 65504), so the *scale factor itself*
becomes `inf`.

Every product is then `inf`, or `NaN` where the timestamp is exactly `0.0`, and
casting either to `int64` yields `INT64_MIN`. The result is that **every
timestamp in every sample of both SHD and SSC is `-9223372036854775808`**.

Nothing raises. The only visible sign is three `RuntimeWarning`s, which are
easy to miss and are not errors.

## Reproduction

```python
import h5py, numpy as np, tonic

ds = tonic.datasets.SSC(save_to="./data", split="test")
events, label = ds[0]
print(events["t"].min(), events["t"].max())
# -9223372036854775808 -9223372036854775808

# The mechanism, directly:
f = h5py.File("./data/SSC/ssc_test.h5", "r")
t = f["spikes/times"][0]
print(t.dtype, t.min(), t.max())        # float16 0.0 0.9937
print((t * 1e6).dtype)                  # float16
print((t * 1e6)[:3])                    # [nan inf inf]
print((t * 1e6).astype(int)[:3])        # [-9223372036854775808 ...] x3
```

Same for SHD (`shd_test.h5`); `times` is `float16` in both datasets and both
splits.

## Why this is worse than a crash

Downstream code typically bins timestamps into a fixed number of time steps by
scaling against the sample's own `[t_min, t_max]` span. With every timestamp
equal, the span is zero, so **every event lands in the first time step**. A
25-step spiking network then trains on a single static frame — and trains
perfectly happily, converging to a plausible-looking accuracy.

Measured on real SSC train samples, binned to 25 steps:

```
sample     0 (label 0): 12618 events, distinct time bins = 1
   spikes per time step: [634, 0, 0, 0, 0, 0, ...]   total 634
sample 40000 (label 15): 5207 events, distinct time bins = 1
   spikes per time step: [589, 0, 0, 0, 0, 0, ...]   total 589
```

With the timestamps read correctly, the same sample spreads across the window
with the temporal envelope you would expect of an utterance:

```
sample 0: distinct time bins = 25 of 25
   spikes per time step: [122, 150, 141, 145, 136, 137, 144, 132, 150, 230,
                          247, 193, 133, 420, 507, 389, 349, 347, 348, 320,
                          299, 283, 250, 213, 137]
```

SHD and SSC are the standard spiking-audio benchmarks, and temporal structure is
the entire point of using them. Anyone who has trained on them through tonic on
NumPy 2 may have been training on single-bin data without any indication.

## Suggested fix

Widen before scaling, so the arithmetic is exact regardless of the file's
storage dtype:

```python
times = np.asarray(file["spikes/times"][index], dtype=np.float64) * 1e6
```

A one-line change. `np.float64` also makes the conversion independent of NumPy's
promotion rules, so it will not silently change behaviour again.

Two things worth considering alongside it:

1. **A guard.** A negative microsecond timestamp cannot be real data. Raising on
   one would have turned this into a loud failure everywhere rather than a
   silent one.
2. **`np.errstate`.** The three `RuntimeWarning`s are the only present signal;
   promoting overflow/invalid to an error in this conversion would surface the
   problem immediately.

## Note on `dtype`

`HSD.dtype` is `[("t", int), ("x", int), ("p", int)]` — no `y` field, since a
cochlea has channels rather than pixel rows, and `sensor_size` is `(700, 1, 1)`.
That is consistent and fine; noting it only because code written against the
2-D DVS `(x, y, t, p)` layout will need to handle its absence.

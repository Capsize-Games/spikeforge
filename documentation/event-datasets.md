# Event datasets (Phase 4)

Phase 4 adds an **event** (neuromorphic) data modality beside the static
image path. Datasets such as N-MNIST, DVS128 Gesture, CIFAR10-DVS, and
Spiking Speech Commands load through [Tonic](https://tonic.readthedocs.io/),
and their recordings flow through the same simulator and NIR validation as
encoded images.

### Installing the `events` extra

Event loading is opt-in so the default install stays lean:

```bash
pip install -e "./packages/spikeforge[events]"
```

`tonic` is deliberately kept out of `requirements.txt`; without it the
event datasets are reported unavailable rather than silently broken. The
picker marks them "unavailable (install the events extra)", the loader
raises the typed `EventsExtraMissingError`, and the server falls back to
the explicit synthetic path described below.

### Datasets, modality, and availability

Every registry entry is now a `DatasetSpec` carrying a `modality` (`image`
or `event`). `catalog()` reports `modality` and `available` per dataset,
and the same fields ride on the `config_ack` and `status` payloads, so the
picker only offers valid options:

| Dataset | Modality | Classes | Available when |
|---|---|---|---|
| `n_mnist` | event | 10 | the `events` extra is installed |
| `dvs128_gesture` | event | 11 | the `events` extra is installed |
| `cifar10_dvs` | event | 10 | the `events` extra is installed |
| `ssc` | event | 35 | the `events` extra is installed |
| image datasets | image | 10-26 | always |

Event datasets download through the existing isolated worker, so tonic's
cache writes under `DATA_DIR/events/` (the gitignored `build/`) and the
progress/cancel UI keeps working.

### How event data flows

```
source -> EventSample -> frames -> bridge -> simulator / NIR
```

1. **Source** — `EventSampleSource` (`events/event_source.py`) serves a
   `(EventSample, label)` pair per index. It uses tonic whenever the
   dataset is loadable and otherwise the deterministic generators in
   `events/synthetic.py`. The chosen backend is recorded in `origin` and
   `description`, so a synthetic stream is never presented as a recording.
2. **`EventSample`** — a validated sparse stream in `(x, y, t, p)` form
   (`x` column, `y` row, `t` 0-based time bin, `p` `+1` ON / `-1` OFF)
   with an `(H, W)` sensor layout.
3. **Frames** — `events/dense.py` accumulates the stream into a time-major
   `[T, 2, H, W]` tensor: `to_frames` binarises and `to_voxel` keeps raw
   counts. Channel 0 is ON and channel 1 is OFF.
4. **Bridge** — `EventSpikeBridge.encode(sample, spec)` lays the frames
   out for the target topology: `[T, B, F]` for feature inputs and
   `[T, B, C, H, W]` for spatial ones, exactly what `simulator.run()`
   already consumes. It reuses the shared `input_shape` and sparsity
   helpers rather than re-implementing coding math.
5. **Simulator / NIR** — because the bridge emits the same spike-tensor
   contract, a topology runs and NIR-validates an event sample unchanged.

The server exposes this through `EventEngine` (`server/event_engine.py`),
which presents the same method surface as `EncoderEngine` (frame, raster,
spike input, label), so `engine_factory` routes a dataset to the right
engine by modality and every downstream handler stays shared.

### Polarity-aware raster and playback

The event raster puts ON events at neuron indices `[0, H*W)` and OFF
events at `[H*W, 2*H*W)`, so the shared `RasterPayload` shape is reused
while the two polarities stay distinguishable. The viewer shows the
recording's own whole-sample ON/OFF frame (`kind: "event_frame"` on the
existing `image` message) beside the playback frame, and `send_initial`
skips reconstruction for events because there is no image coding to
invert.

### Coding controls are gated for events

Event recordings are already spike trains in their own time bins, so
rate/latency/delta/random coding does not apply. The Encoding section
disables the coding and per-coding controls with an explicit note ("events
are already spikes") and leaves only the playback interval adjustable. The
`encoding_report` action rejects an event sample with a typed error instead
of decoding a missing image.

### Limitations

> The consequences of the extras-gated boundaries below are reasoned about in
> [Implications and boundaries](implications-and-boundaries.md).

- **Training on event datasets needs the `events` extra.** Event-mode
  training shipped in WS-F through `EventTrainingEngine`, which batches a
  stream of event samples into the shared training loop; without `tonic` it
  raises the typed `EventsExtraMissingError` rather than silently running.
- **Spatial topologies need 28x28-like geometry.** `conv_net` consumes
  `[T, B, 1, H, W]`; the bridge passes a polar frame through unchanged and
  never reshapes a non-square frame. Use a feature-input topology
  (`fc_legacy`, `fc_small`, `recurrent_net`) for other sensor geometries.
- **The synthetic path is explicitly labelled.** With no `tonic`, or with
  `synthetic_only=True`, the source serves deterministic moving-dot
  streams and says so in `origin`/`description` — they are offline
  fixtures, not real recordings.

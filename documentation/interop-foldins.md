# Interop fold-ins (WS-F)

### Event-dataset training

[`training/event_engine.py`](../spikeforge/training/event_engine.py:1) and
[`training/event_batches.py`](../spikeforge/training/event_batches.py:1)
batch an event stream through the existing bridge into the `[T, B, …]`
contract the training loop already consumes, so the loss/optimizer/metrics/
checkpointing path is shared. A checkpoint records `modality: event`; without
`tonic` the engine raises the typed `EventsExtraMissingError`, and a synthetic
stream is never presented as a recording.

### ONNX bridge

[`onnx_bridge/`](../spikeforge/onnx_bridge/__init__.py:1) exports a topology's
**single forward step** to ONNX (the time loop stays in the simulator) with the
spec in metadata, and imports a third-party graph by mapping ops to stage kinds
— or failing with a typed error naming the op. The `onnx` extra provides
`onnx`/`onnxruntime`; imports are confined to `onnx_bridge/api.py`.

```bash
spikeforge-verify onnx-export    --topology conv_net --out build/model.onnx
spikeforge-verify onnx-import    --file build/model.onnx
spikeforge-verify onnx-roundtrip --topology conv_net
```

### `nirtorch` extraction

[`nir_bridge/extract.py`](../spikeforge/nir_bridge/extract.py:1) lifts an
arbitrary `torch.nn.Module` into NIR through the isolated `nirtorch` wrapper,
then runs it on the independent interpreter:

```bash
spikeforge-targets extract --module model.pt
```

[`torch_map.NODE_MAP`](../spikeforge/nir_bridge/torch_map.py:39) maps only
`nn.Linear` and `nn.Flatten`; any other module raises the typed
`UnsupportedNodeError` naming the class — no silent truncation.

### Quantization

[`targets/quantize.py`](../spikeforge_targets/quantize.py:103) applies a
target's **declared** scheme (`none`, `weight_int8`, `weight_uint8`) to a
graph's weights, reporting per-layer before/after ranges and the induced drift.
It is **weight-level only** (no activations, no device), a `none` target is a
reported no-op, and an unknown scheme is reported unapplied.

### Non-square geometry and `input_size`

[`data/image_size.py`](../spikeforge/data/image_size.py:16) normalises a
geometry declared as an `int` side or an explicit `(H, W)` pair, and
`EncodeConfig.input_size` propagates it. Presets keep 28×28 by default, so
every shipped preset is byte-identical until a shape is requested.

### Hidden-layer animation

`EncodeConfig.animate_hidden` (default off) streams a per-step hidden-layer
frame over the existing `spike_frame`/`animation_state` channel;
[`network/hidden_frames.py`](../spikeforge/network/hidden_frames.py:1) caps
the width so an oversized layer cannot flood the socket. With the flag unset,
the payload stream is identical to before.

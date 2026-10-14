"""Input encoding for the training engine (raw pixels or spike codes)."""

from typing import Any, Dict, Optional, Tuple

import torch

from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.serving import preprocess
from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.simulator import input_shape


class EncodingMixin:
    """Turn a batch of images into spikes shaped for the active topology."""

    _encoder: Optional[SpikeEncoder]
    _encode_spec: EncodeSpec
    _explicit_mode: Optional[str]
    _input_mode: str
    _num_classes: int
    _hidden: int
    _num_steps: int
    _batch_size: int

    def _setup_input(
        self, encode: Any, input_mode: Optional[str], device: Optional[str]
    ) -> None:
        """Resolve the encoder, effective input mode, and compute device."""
        self._encoder = (
            SpikeEncoder.from_encode_config(encode)
            if encode is not None
            else None
        )
        self._encode_spec = EncodeSpec.from_mapping(encode)
        self._explicit_mode = input_mode
        coding = encode.coding if encode is not None else "raw"
        self._input_mode = input_mode or coding
        self._set_device(device or "auto", encoder=self._encoder,
                         hidden=self._hidden, num_classes=self._num_classes,
                         num_steps=self._num_steps,
                         batch_size=self._batch_size)

    def _encode_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return [T,B,F] or [T,B,1,H,W] spikes per input stage."""
        return self._to_input_shape(self._flat_spikes(inputs))

    def _flat_spikes(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return flattened [T,B,784] spikes for the configured mode.

        The spike path goes through the shared pure
        :func:`spikeforge.serving.preprocess.encode` so training and serving
        produce byte-identical codes. The raw short-circuit and the
        no-label-signal refusal for the ``random`` coding are unchanged.
        """
        if self._encoder is None or self._input_mode == "raw":
            return self._repeat_pixels(inputs).to(self._device)
        if self._input_mode == "random":
            raise ValueError("random coding carries no label signal")
        return preprocess.encode(inputs, self._encode_spec).to(self._device)

    def _to_input_shape(self, spikes: torch.Tensor) -> torch.Tensor:
        """Reshape flat spikes to the topology's input-stage layout.

        The sensor geometry comes from the topology's declared ``input_size``
        when its input stage is a convolution; a ``flatten`` input declares a
        flat feature count, and a feature/sequence topology keeps its flat
        ``[T, B, F]`` layout, so both fall back to the default geometry.
        """
        value = self._architecture.get("input_size")
        shape = input_shape.spatial_shape(self._spec, value)
        if shape is None:
            return input_shape.to_input_shape(spikes, self._spec)
        return input_shape.to_input_shape(spikes, self._spec, shape)

    def _input_geometry(self) -> Optional[Tuple[int, int]]:
        """Return the sensor ``(H, W)`` the input stage consumes, or None.

        A convolution-like input declares a geometry; a flat feature or
        sequence input has none, so ``None`` is recorded and the serve path
        keeps the flat layout.
        """
        value = self._architecture.get("input_size")
        return input_shape.spatial_shape(self._spec, value)

    def _encode_spec_meta(self) -> Dict[str, Any]:
        """Return the normalised encode spec recorded in a checkpoint.

        The frozen :class:`~spikeforge.serving.encode_spec.EncodeSpec` is
        dumped canonically (so it carries ``spec_version``), and its
        ``input_size`` is filled in from the topology geometry when the spec
        itself does not pin one.
        """
        data = self._encode_spec.to_dict()
        geometry = self._input_geometry()
        if geometry is not None:
            data["input_size"] = [int(geometry[0]), int(geometry[1])]
        return data

    def _repeat_pixels(self, inputs: torch.Tensor) -> torch.Tensor:
        """Legacy raw path: repeat normalised pixels across steps."""
        flat = inputs.view(inputs.size(0), -1)
        return flat.unsqueeze(0).repeat(self._num_steps, 1, 1)

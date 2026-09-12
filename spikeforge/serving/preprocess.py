"""One pure encode call site shared by the training and serving paths.

Training and serving must not encode a sample differently. Both now call
:func:`encode`, which resolves an :class:`~spikeforge.serving.encode_spec.\
EncodeSpec`, coerces the raw sample to a tensor, seeds the RNG when the spec
pins a seed, and dispatches through the single
:class:`~spikeforge.encoding.spike_encoder.SpikeEncoder` the core already
owns. No encoding math lives here; this module only normalises the shapes and
the RNG around it.

When a ``topology_spec`` is supplied the flat ``[T, B, F]`` code is reshaped
to the input stage's layout (``[T, B, C, H, W]`` for a spatial input) with
:func:`spikeforge.simulator.input_shape.to_input_shape`, so a served sample
lands in exactly the shape training produced.
"""

from typing import Any, Optional, Sequence

import torch

from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.simulator import input_shape
from spikeforge.topology.spec import TopologySpec


def encoder_for(spec: Any = None) -> SpikeEncoder:
    """Return the validated :class:`SpikeEncoder` an encode spec pins."""
    resolved = EncodeSpec.from_mapping(spec)
    resolved.validate()
    return resolved.to_encoder()


def encode(
    sample: Any,
    spec: Any = None,
    topology_spec: Optional[TopologySpec] = None,
    geometry: Any = None,
) -> torch.Tensor:
    """Encode one sample (or a batch) into a frozen spike tensor.

    ``sample`` is coerced to a float tensor: a 3-D image is promoted to a
    single-item batch and a 4-D tensor is treated as a batch. ``spec`` accepts
    an :class:`EncodeSpec`, a mapping, a ``model_dump()``-able config, or
    ``None`` for the defaults, and is validated before any work. When the spec
    pins ``random_seed`` the torch RNG is seeded immediately before dispatch,
    so a seeded coding repeats byte-for-byte. Supplying ``topology_spec`` (and
    optionally ``geometry``) reshapes the result for the input stage.
    """
    resolved = EncodeSpec.from_mapping(spec)
    resolved.validate()
    tensor = _as_sample(sample)
    if resolved.random_seed is not None:
        torch.manual_seed(int(resolved.random_seed))
    spikes = resolved.to_encoder().encode(tensor)
    if topology_spec is not None:
        spikes = _shape_for(spikes, topology_spec, geometry)
    return spikes


def encode_batch(
    samples: Any,
    spec: Any,
    topology_spec: Optional[TopologySpec] = None,
    geometry: Any = None,
) -> torch.Tensor:
    """Encode a batch of samples with one shared spec.

    A batch is a single tensor whose leading dimension indexes the samples;
    a sequence of same-shaped samples is stacked into one. The result is the
    same ``[T, B, ...]`` tensor :func:`encode` returns.
    """
    return encode(samples, spec, topology_spec, geometry)


def _as_sample(sample: Any) -> torch.Tensor:
    """Return ``sample`` as a ``[B, ...]`` float tensor.

    A 3-D image ``[C, H, W]`` is promoted to a single-item batch; any other
    tensor with at least two dimensions is treated as a batch already. A
    non-numeric payload or a payload with fewer than two dimensions is
    refused rather than silently reshaped.
    """
    try:
        tensor = torch.as_tensor(sample, dtype=torch.float32)
    except (TypeError, ValueError) as error:
        raise TypeError(f"sample is not numeric: {error}") from None
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    if tensor.dim() < 2:
        raise ValueError(
            "sample must be at least 2-D (a batch or a single image)"
        )
    return tensor


def _shape_for(
    spikes: torch.Tensor,
    topology_spec: TopologySpec,
    geometry: Any,
) -> torch.Tensor:
    """Return ``spikes`` laid out for ``topology_spec``'s input stage."""
    if geometry is None:
        return input_shape.to_input_shape(spikes, topology_spec)
    return input_shape.to_input_shape(spikes, topology_spec, geometry)


#: A batch of samples: a tensor or a sequence the encoder can stack.
Samples = Sequence[Any]

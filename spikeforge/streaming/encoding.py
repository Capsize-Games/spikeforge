"""P0: the single window-to-spike encode call site for train and serve.

Training and serving must never encode a window differently. Both call
:func:`encode_windows`, which resolves a frozen
:class:`~spikeforge.serving.encode_spec.EncodeSpec`, coerces the window to a
``[B, L, D]`` tensor, and dispatches through the core
:class:`~spikeforge.encoding.spike_encoder.SpikeEncoder`. The encoder's flat
``[T, B, L*D]`` output is reshaped to the ``[T, B, L, D]`` the sequence preset
consumes, so the window's token axis is preserved without re-implementing any
encoding math.

``delta`` coding (deterministic) is the primary contract; ``rate`` is kept as
a fallback baseline. :func:`delta_over_window` optionally differences the
window along its time axis first, so the temporal change -- not the absolute
level -- is what the spike train carries.

The module is pure torch and stdlib.
"""

from typing import Any

import torch

from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.streaming.window_spec import WindowSpec


def _as_windows(windows: Any) -> torch.Tensor:
    """Return ``windows`` as a ``[B, L, D]`` float32 tensor.

    A single ``[L, D]`` window is promoted to a one-item batch; a tensor with
    at least three dimensions is treated as a batch already. A 1-D payload is
    refused rather than silently reinterpreted.
    """
    tensor = torch.as_tensor(windows, dtype=torch.float32)
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() < 3:
        raise ValueError("windows must be at least 2-D ([L, D] or [B, L, D])")
    return tensor


def delta_over_window(windows: Any) -> torch.Tensor:
    """Return the per-channel change along the window time axis.

    The first sample of every window has no predecessor, so it is zero, and
    sample ``t`` becomes ``x[t] - x[t-1]``. This turns "absolute level" into
    "temporal change", which is the signal the SNN integrates.
    """
    tensor = torch.as_tensor(windows, dtype=torch.float32)
    if tensor.dim() < 2:
        raise ValueError("windows must be at least 2-D")
    first = tensor[..., :1, :]
    step = tensor[..., 1:, :] - tensor[..., :-1, :]
    return torch.cat([torch.zeros_like(first), step], dim=-2)


def encode_windows(
    windows: Any,
    spec: Any = None,
    *,
    delta_along_window: bool = False,
) -> torch.Tensor:
    """Encode ``[B, L, D]`` windows into a ``[T, B, L, D]`` spike tensor.

    ``spec`` accepts an :class:`EncodeSpec`, a mapping, or ``None`` for the
    defaults, and is validated before any work. When the spec pins
    ``random_seed`` the torch RNG is seeded immediately before dispatch, so a
    seeded (stochastic) coding repeats. ``delta_along_window`` prefixes the
    deterministic first-difference transform.
    """
    resolved = EncodeSpec.from_mapping(spec)
    resolved.validate()
    tensor = _as_windows(windows)
    if delta_along_window:
        tensor = delta_over_window(tensor)
    if resolved.random_seed is not None:
        torch.manual_seed(int(resolved.random_seed))
    flat = resolved.to_encoder().encode(tensor)
    shape = tensor.shape[1:]
    return flat.reshape(flat.size(0), flat.size(1), *shape)


def encode_from_bundle(bundle: Any, windows: Any) -> torch.Tensor:
    """Encode raw windows with a bundle's frozen window + encode contract.

    The windowing spec (train-fitted z-score stats, ``L``, stride, channel
    order) and the encode spec both come from the loaded
    :class:`~spikeforge.serving.bundle.DeploymentBundle`, so inference cannot
    normalise or encode a window differently from training.
    """
    spec = bundle.encode_spec()
    preprocessing = dict(bundle.preprocessing or {})
    tensor = torch.as_tensor(windows, dtype=torch.float32)
    window_data = preprocessing.get("window")
    if window_data:
        tensor = WindowSpec.from_dict(window_data).normalize(tensor)
    return encode_windows(
        tensor,
        spec,
        delta_along_window=bool(preprocessing.get("delta_over_window", False)),
    )

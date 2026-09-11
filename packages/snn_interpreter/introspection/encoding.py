"""JSON-serialisable encoding introspection and image reconstruction.

:func:`encoding_report` encodes one image with the caller's
:class:`SpikeEncoder`, reconstructs the image where the coding is invertible,
and reports firing rate, sparsity, and coding-specific stats. Every value is
a plain ``dict``/``list``/``int``/``float``/``str``/``None``, so the report
passes straight through ``json.dumps``.
"""

from typing import Any, Callable, Dict, Tuple

import torch

from snn_interpreter.encoding.spike_encoder import SpikeEncoder
from snn_interpreter.introspection.decoding import (
    decode_delta,
    decode_latency,
    decode_rate,
)
from snn_interpreter.introspection.firing_rate import firing_rate
from snn_interpreter.introspection.sparsity import sparsity

_DELTA_SCALE = 100.0

_RATE_NOTE = (
    "Mean spike count over time; a Bernoulli estimate of the clamped "
    "intensity that converges as num_steps grows."
)
_LATENCY_NOTE = (
    "Inverts the time-to-first-spike mapping; quantised to integer steps "
    "and saturating at the threshold ceiling for sub-threshold pixels."
)
_DELTA_NOTE = (
    "Integration of the on/off stream crediting one delta threshold per "
    "spike; a lower bound that is exact only when each step rises by "
    "exactly the threshold."
)
_RANDOM_NOTE = (
    "The random coding ignores the image and emits uniform noise, so it "
    "carries no image signal to reconstruct."
)

#: A coding block: flat reconstruction, documented note, and stats.
Block = Callable[
    [torch.Tensor, SpikeEncoder], Tuple[torch.Tensor, str, Dict[str, Any]]
]


def _as_image(image: torch.Tensor) -> torch.Tensor:
    """Return ``image`` as a float ``[C, H, W]`` tensor."""
    data = image.detach().float()
    if data.dim() == 2:
        return data.unsqueeze(0)
    if data.dim() == 3:
        return data
    raise ValueError(f"expected a 2D or 3D image, got {data.dim()} dims")


def _nested(flat: torch.Tensor, shape: Tuple[int, ...]) -> Any:
    """Reshape a flat reconstruction into nested Python lists."""
    return flat.reshape(shape).tolist()


def _rate_block(
    spikes: torch.Tensor, encoder: SpikeEncoder
) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
    """Return the flat rate reconstruction, note, and stats."""
    stats: Dict[str, Any] = {"estimator": "mean spike count", "gain": 1.0}
    return decode_rate(spikes), _RATE_NOTE, stats


def _latency_block(
    spikes: torch.Tensor, encoder: SpikeEncoder
) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
    """Return the flat latency reconstruction, note, and stats."""
    flat = decode_latency(
        spikes,
        num_steps=encoder.num_steps,
        tau=encoder.tau,
        threshold=encoder.threshold,
        normalize=encoder.normalize,
        linear=encoder.linear,
    )
    stats: Dict[str, Any] = {
        "linear": encoder.linear,
        "normalize": encoder.normalize,
        "tau": encoder.tau,
        "threshold": encoder.threshold,
        "clip": encoder.clip,
    }
    return flat, _LATENCY_NOTE, stats


def _delta_block(
    spikes: torch.Tensor, encoder: SpikeEncoder
) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
    """Return the flat delta reconstruction, note, and stats."""
    scale = encoder.delta_threshold / _DELTA_SCALE
    stats: Dict[str, Any] = {
        "delta_threshold": encoder.delta_threshold,
        "integration_scale": scale,
    }
    return decode_delta(spikes, scale), _DELTA_NOTE, stats


_BLOCKS: Dict[str, Block] = {
    "rate": _rate_block,
    "latency": _latency_block,
    "delta": _delta_block,
}


def _random_block(encoder: SpikeEncoder) -> Dict[str, Any]:
    """Return the random coding's explicit no-image-signal block."""
    return {
        "reconstruction": None,
        "reconstruction_supported": False,
        "approximation": _RANDOM_NOTE,
        "stats": {
            "image_signal": False,
            "seed": encoder.seed,
            "random_scale": encoder.random_scale,
        },
    }


def _decode(
    data: torch.Tensor, spikes: torch.Tensor, encoder: SpikeEncoder
) -> Dict[str, Any]:
    """Return the reconstruction block for the encoder's coding."""
    if encoder.coding == "random":
        return _random_block(encoder)
    flat, note, stats = _BLOCKS[encoder.coding](spikes, encoder)
    return {
        "reconstruction": _nested(flat, tuple(data.shape)),
        "reconstruction_supported": True,
        "approximation": note,
        "stats": stats,
    }


def encoding_report(
    image: torch.Tensor, encoder: SpikeEncoder
) -> Dict[str, Any]:
    """Return a JSON-able reconstruction report for one ``image``."""
    data = _as_image(image)
    spikes = encoder.encode_image(data)
    report: Dict[str, Any] = {
        "coding": encoder.coding,
        "num_steps": encoder.num_steps,
        "shape": list(data.shape),
        "firing_rate": firing_rate(spikes),
        "sparsity": sparsity(spikes),
    }
    report.update(_decode(data, spikes, encoder))
    return report

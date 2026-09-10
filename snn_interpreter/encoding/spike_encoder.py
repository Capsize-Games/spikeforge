"""Single source of truth for image-to-spike encoding math."""

from typing import Any, Dict, Optional, Tuple

import torch
from snntorch import spikegen


class SpikeEncoder:
    """Encode image batches into [T, B, 784] spike tensors."""

    _CODINGS: Tuple[str, ...] = ("rate", "latency", "delta", "random")
    _FIELDS: Tuple[str, ...] = (
        "coding",
        "num_steps",
        "tau",
        "threshold",
        "clip",
        "normalize",
        "linear",
        "delta_threshold",
        "random_scale",
    )

    def __init__(
        self, coding: str = "rate", num_steps: int = 100, tau: float = 5.0,
        threshold: float = 0.01, clip: bool = False, normalize: bool = True,
        linear: bool = True, delta_threshold: float = 4.0,
        random_scale: float = 0.5, seed: Optional[int] = None,
    ) -> None:
        """Store the coding parameters used by every encode path."""
        self._coding = coding if coding in self._CODINGS else "rate"
        self._num_steps = max(1, int(num_steps))
        self._tau = float(tau)
        self._threshold = float(threshold)
        self._clip = bool(clip)
        self._normalize = bool(normalize)
        self._linear = bool(linear)
        self._delta_threshold = float(delta_threshold)
        self._random_scale = float(random_scale)
        self._seed = seed

    @classmethod
    def from_encode_config(cls, cfg: Any) -> "SpikeEncoder":
        """Build an encoder from an EncodeConfig-like object."""
        kwargs: Dict[str, Any] = {
            field: getattr(cfg, field, None) for field in cls._FIELDS
        }
        kwargs = {key: val for key, val in kwargs.items() if val is not None}
        return cls(seed=getattr(cfg, "random_seed", None), **kwargs)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch [B,C,H,W] into [T,B,784] spikes."""
        return self._dispatch(images)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a single image [C,H,W] into [T,1,784] spikes."""
        batch = image.unsqueeze(0) if image.dim() == 3 else image
        return self._dispatch(batch)

    def _dispatch(self, images: torch.Tensor) -> torch.Tensor:
        """Route to the coding implementation and flatten trailing dims."""
        raw = {
            "rate": self._rate,
            "latency": self._latency,
            "delta": self._delta,
            "random": self._random,
        }[self._coding](images)
        return raw.reshape(raw.size(0), raw.size(1), -1)

    def _rate(self, images: torch.Tensor) -> torch.Tensor:
        """Bernoulli rate code of pixel intensity at unit gain."""
        return spikegen.rate(images, num_steps=self._num_steps, gain=1.0)

    def _latency(self, images: torch.Tensor) -> torch.Tensor:
        """RC latency code with the configured tau/threshold/flags."""
        return spikegen.latency(
            images,
            num_steps=self._num_steps,
            tau=self._tau,
            threshold=self._threshold,
            clip=self._clip,
            normalize=self._normalize,
            linear=self._linear,
        )

    def _delta(self, images: torch.Tensor) -> torch.Tensor:
        """Per-pixel intensity ramp encoded with spikegen.delta."""
        pixels = images.reshape(images.size(0), -1)
        ramp = torch.linspace(0, 1, self._num_steps).view(-1, 1, 1)
        return spikegen.delta(
            ramp * pixels.unsqueeze(0),
            threshold=self._delta_threshold / 100.0,
        )

    def _random(self, images: torch.Tensor) -> torch.Tensor:
        """Noise baseline: rate-coded uniform noise, sample ignored."""
        shape = (self._num_steps, images.size(0), images[0].numel())
        if self._seed is not None:
            torch.manual_seed(self._seed)
        return spikegen.rate_conv(torch.rand(shape) * self._random_scale)

    @property
    def num_steps(self) -> int:
        """Return the number of encoded time steps."""
        return self._num_steps

    @property
    def coding(self) -> str:
        """Return the active coding name."""
        return self._coding

    @property
    def tau(self) -> float:
        """Return the latency RC time constant."""
        return self._tau

    @property
    def threshold(self) -> float:
        """Return the latency firing threshold."""
        return self._threshold

    @property
    def clip(self) -> bool:
        """Return whether below-threshold latencies are clipped."""
        return self._clip

    @property
    def normalize(self) -> bool:
        """Return whether latency codes are normalised to ``num_steps``."""
        return self._normalize

    @property
    def linear(self) -> bool:
        """Return whether the latency code is linear (not logarithmic)."""
        return self._linear

    @property
    def delta_threshold(self) -> float:
        """Return the delta change threshold (in percent)."""
        return self._delta_threshold

    @property
    def random_scale(self) -> float:
        """Return the noise scale used by the random coding."""
        return self._random_scale

    @property
    def seed(self) -> Optional[int]:
        """Return the optional seed used by the random coding."""
        return self._seed

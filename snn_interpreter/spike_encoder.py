"""Single source of truth for image-to-spike encoding math."""

import torch
from snntorch import spikegen


class SpikeEncoder:
    """Encode image batches into [T, B, 784] spike tensors."""

    _CODINGS = ("rate", "latency", "delta", "random")
    _FIELDS = ("coding", "num_steps", "gain", "vector_value", "tau",
               "threshold", "clip", "normalize", "linear",
               "delta_threshold", "random_scale")

    def __init__(self, coding="rate", num_steps=100, gain=0.25,
                 vector_value=0.5, tau=5.0, threshold=0.01, clip=False,
                 normalize=True, linear=True, delta_threshold=4.0,
                 random_scale=0.5, seed=None):
        self._coding = coding if coding in self._CODINGS else "rate"
        self._num_steps = max(1, int(num_steps))
        self._gain = float(gain)
        self._vector_value = float(vector_value)
        self._tau = float(tau)
        self._threshold = float(threshold)
        self._clip = bool(clip)
        self._normalize = bool(normalize)
        self._linear = bool(linear)
        self._delta_threshold = float(delta_threshold)
        self._random_scale = float(random_scale)
        self._seed = seed

    @classmethod
    def from_encode_config(cls, cfg):
        """Build an encoder from an EncodeConfig-like object."""
        kwargs = {f: getattr(cfg, f, None) for f in cls._FIELDS}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return cls(seed=getattr(cfg, "random_seed", None), **kwargs)

    def encode(self, images):
        """Encode a batch [B,C,H,W] into [T,B,784] spikes."""
        return self._dispatch(images)

    def encode_image(self, image):
        """Encode a single image [C,H,W] into [T,1,784] spikes."""
        batch = image.unsqueeze(0) if image.dim() == 3 else image
        return self._dispatch(batch)

    def _dispatch(self, images):
        """Route to the coding implementation and flatten trailing dims."""
        raw = {
            "rate": self._rate,
            "latency": self._latency,
            "delta": self._delta,
            "random": self._random,
        }[self._coding](images)
        return raw.reshape(raw.size(0), raw.size(1), -1)

    def _rate(self, images):
        """Bernoulli rate code of pixel intensity."""
        return spikegen.rate(images, num_steps=self._num_steps, gain=1.0)

    def _latency(self, images):
        """RC latency code with the configured tau/threshold/flags."""
        return spikegen.latency(
            images, num_steps=self._num_steps, tau=self._tau,
            threshold=self._threshold, clip=self._clip,
            normalize=self._normalize, linear=self._linear,
        )

    def _delta(self, images):
        """Per-pixel intensity ramp encoded with spikegen.delta."""
        pixels = images.reshape(images.size(0), -1)
        ramp = torch.linspace(0, 1, self._num_steps).view(-1, 1, 1)
        return spikegen.delta(
            ramp * pixels.unsqueeze(0),
            threshold=self._delta_threshold / 100.0,
        )

    def _random(self, images):
        """Noise baseline: rate-coded uniform noise, sample ignored."""
        shape = (self._num_steps, images.size(0), images[0].numel())
        if self._seed is not None:
            torch.manual_seed(self._seed)
        return spikegen.rate_conv(torch.rand(shape) * self._random_scale)

    @property
    def num_steps(self):
        """Return the number of encoded time steps."""
        return self._num_steps

    @property
    def coding(self):
        """Return the active coding name."""
        return self._coding

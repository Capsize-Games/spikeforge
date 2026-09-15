"""Event-dataset training on the shared simulator and loss loop.

This engine composes the existing event pieces — :class:`EventSampleSource`,
:class:`EventSpikeBridge`, and the topology registry — so the loop, metrics,
and checkpointing in :mod:`training_engine` are reused unchanged. Only the
dataset step differs: image modality reads a torchvision loader, event
modality bridges a stream of samples into ``[T, B, ...]`` spikes.

The engine holds *two* sources, one per split, mirroring the way the image
path holds a train and a test loader. ``_epoch_batches(train=...)`` picks
between them, so held-out scoring reads the dataset's test split rather than
the head of the training stream, and ``event_batches`` needs no split
argument. A dataset that declares no test split — CIFAR10-DVS ships as one
undivided pool — fails at construction with
:class:`~spikeforge.data.event_errors.EventSplitMissingError` instead of
training happily and reporting its own training data as a held-out score.

Sensor geometry is checked up front. A conv topology needs a square,
single/dual-channel sensor of its declared side; a feature topology needs
``input_size`` to equal the sensor area. A mismatch raises
:class:`EventGeometryError` naming both sides rather than reshaping a
stream the network was never built for.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from spikeforge.data.event_errors import EventsExtraMissingError
from spikeforge.data.event_geometry import EventGeometryError
from spikeforge.data.image_size import as_size
from spikeforge.events import event_source
from spikeforge.events.event_bridge import EventSpikeBridge
from spikeforge.events.event_source import EventSampleSource
from spikeforge.training import event_batches
from spikeforge.training.eval_mixin import EVAL_BATCHES
from spikeforge.training.training_engine import TrainingEngine

#: Input-stage kinds that consume a fixed square spatial frame.
_SPATIAL_INPUT_KINDS = ("conv2d", "avgpool2d", "maxpool2d", "sumpool2d")
#: Channel counts the event bridge can produce (single or ON/OFF polar).
_SUPPORTED_CHANNELS = (1, 2)


class EventTrainingEngine(TrainingEngine):
    """Train the shared simulator on a stream of event samples.

    ``synthetic_only`` forces the explicitly-labelled offline generators so
    tests and demos stay network-free; otherwise a missing ``tonic`` loader
    raises :class:`EventsExtraMissingError` instead of presenting a
    synthetic stream as a recording.

    :attr:`_event_source` serves the training split and
    :attr:`_test_source` the held-out one. Both are opened up front so a
    dataset that cannot supply held-out data says so before a single batch
    is trained.
    """

    _synthetic_only: bool
    _event_source: EventSampleSource
    _test_source: EventSampleSource
    _bridge: EventSpikeBridge
    #: Training samples per epoch, or ``None`` for the dashboard's batch cap.
    _epoch_samples: Optional[int]

    def __init__(
        self, dataset: str = "n_mnist", synthetic_only: bool = False,
        epoch_samples: Optional[int] = None, **kwargs: Any,
    ) -> None:
        """Resolve both sources, build the engine, then check geometry.

        ``epoch_samples`` (default: the dashboard's batch cap) sets the
        training epoch's sample count; pass the split's own size for a
        full pass. Held-out scoring uses its own source/extent regardless.
        """
        self._synthetic_only = bool(synthetic_only)
        self._epoch_samples = (
            None if epoch_samples is None else int(epoch_samples)
        )
        self._bridge = EventSpikeBridge()
        self._event_source = self._source(dataset, kwargs)
        self._test_source = self._source(dataset, kwargs, split="test")
        super().__init__(dataset=dataset, **kwargs)
        self._validate_geometry()

    def _source(
        self, dataset: str, kwargs: Dict[str, Any],
        split: str = event_source.DEFAULT_SPLIT,
    ) -> EventSampleSource:
        """Build one split's source and enforce the absent-extra honesty rule.

        A dataset that declares no such split raises
        :class:`~spikeforge.data.event_errors.EventSplitMissingError` from
        the source's own registry lookup; it propagates untouched so the
        reason names the dataset and the split, not the engine.
        """
        steps = int(kwargs.get("num_steps") or event_source.SYNTHETIC_STEPS)
        source = EventSampleSource(
            dataset, synthetic_only=self._synthetic_only, num_steps=steps,
            split=split,
        )
        if not self._synthetic_only and source.origin != event_source.TONIC:
            raise EventsExtraMissingError(dataset)
        return source

    def _setup_input(
        self, encode: Any, input_mode: Optional[str], device: Optional[str]
    ) -> None:
        """Skip the image encoder: event samples are already spike trains."""
        super()._setup_input(None, input_mode, device)

    def _dummy_spikes(self) -> torch.Tensor:
        """Return a correctly-shaped event spike train for warmup."""
        sample, _label = self._event_source.load(self._event_source.clamp(0))
        spikes, _meta = self._bridge.encode(sample, self._spec)
        return spikes.to(self._device)

    def _encode_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return event spikes unchanged: the bridge already laid them out."""
        return inputs.to(self._device)

    def _epoch_batches(
        self, train: bool = True
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return one split's events already bridged for the simulator.

        The training split is visited in a seeded random order and the
        held-out split sequentially -- real event datasets ship grouped by
        class, so reading a training epoch in order makes every batch a
        single class.
        """
        source = self._event_source if train else self._test_source
        return event_batches.event_batches(
            source, self._spec, self._subset, self._batch_size,
            self._epoch_samples if train else None,
            shuffle=train, seed=self._seed,
        )

    def _load_test_batches(
        self,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Cache the first ``EVAL_BATCHES`` batches of the held-out split.

        This is a sample of the test split, matching what the image path
        scores during training; the reference-model scripts score the
        complete split separately.
        """
        if self._test_batches is None:
            batches = self._epoch_batches(train=False)
            self._test_batches = batches[:EVAL_BATCHES]
        return self._test_batches

    def predict_sample(self) -> Dict[str, List[int]]:
        """Return digits/labels for one bridged event batch."""
        inputs, targets = self._epoch_batches(train=False)[0]
        preds = self.predict(inputs)
        return {
            "digits": [int(p) for p in preds[:10]],
            "labels": [int(t) for t in targets[:10]],
        }

    def _meta(self) -> Dict[str, Any]:
        """Add the event modality and both splits' provenance to the card.

        The held-out source is recorded too: an accuracy on the card is only
        as trustworthy as the stream it was measured on, so a reader can see
        whether that stream was a recording or a generated fixture.
        """
        meta = super()._meta()
        meta.update({
            "modality": "event",
            "event_origin": self._event_source.origin,
            "event_description": self._event_source.description,
            "event_test_origin": self._test_source.origin,
            "event_test_description": self._test_source.description,
        })
        return meta

    def _manifest_config(self) -> Dict[str, Any]:
        """Record the event modality and epoch extent in the manifest.

        ``epoch_samples`` changes what an epoch means, so a manifest that
        omitted it could not reproduce the run it describes.
        """
        config = super()._manifest_config()
        config["modality"] = "event"
        config["epoch_samples"] = self._epoch_samples
        return config

    # --- geometry ---------------------------------------------------------

    def _sensor_shape(self) -> Tuple[int, int]:
        """Return the sensor ``(H, W)`` of the source's first sample."""
        sample, _label = self._event_source.load(self._event_source.clamp(0))
        return sample.shape

    def _validate_geometry(self) -> None:
        """Reject a sensor whose geometry the topology cannot consume."""
        expected = self._architecture.get("input_size")
        if expected is None:
            return
        height, width = self._sensor_shape()
        kind = self._spec.stage(self._spec.input).kind
        if kind in _SPATIAL_INPUT_KINDS:
            self._check_spatial(height, width, expected)
        else:
            self._check_features(height, width, int(expected))

    def _check_spatial(
        self, height: int, width: int, expected: Any
    ) -> None:
        """Require a supported-channel sensor matching the declared shape.

        The declared ``input_size`` may be a square side or an explicit
        ``(H, W)`` pair, so a non-square conv sensor is accepted exactly when
        the topology was built for that shape.
        """
        stage = self._spec.stage(self._spec.input)
        channels = int(stage.params.get("in_channels", 1))
        want_h, want_w = as_size(expected)
        wrong = (
            channels not in _SUPPORTED_CHANNELS
            or (height, width) != (want_h, want_w)
        )
        if wrong:
            raise EventGeometryError(
                self._spatial_reason(height, width, channels, want_h, want_w)
            )

    def _spatial_reason(
        self, height: int, width: int, channels: int,
        want_h: int, want_w: int,
    ) -> str:
        """Return the named reason a conv input rejected this sensor."""
        if channels not in _SUPPORTED_CHANNELS:
            detail = (
                "needs 1 or 2 input channels, but the topology declares "
                f"{channels}"
            )
        else:
            detail = (
                f"expects a {want_h}x{want_w} sensor, but "
                f"{self._dataset} supplies {height}x{width}"
            )
        return (
            f"{self._topology} {detail}; use a feature-input topology whose "
            f"input_size equals the sensor area ({height * width})"
        )

    def _check_features(
        self, height: int, width: int, expected: int
    ) -> None:
        """Require the flat sensor area to match the topology's input size."""
        if height * width != expected:
            raise EventGeometryError(
                self._feature_reason(height, width, expected)
            )

    def _feature_reason(self, height: int, width: int, expected: int) -> str:
        """Return the named reason a feature input rejected this sensor."""
        features = height * width
        return (
            f"{self._topology} expects {expected} input features, but "
            f"{self._dataset}'s {height}x{width} sensor flattens to "
            f"{features}; set topology_params input_size={features} or pick "
            "a matching topology"
        )

"""Sensor-geometry validation for the event training engine."""

from typing import Any, Dict, Tuple

from spikeforge.data.event_geometry import EventGeometryError
from spikeforge.data.image_size import as_size
from spikeforge.events.event_source import EventSampleSource
from spikeforge.topology.spec import TopologySpec

#: Input-stage kinds that consume a fixed square spatial frame.
_SPATIAL_INPUT_KINDS = ("conv2d", "avgpool2d", "maxpool2d", "sumpool2d")
#: Channel counts the event bridge can produce (single or ON/OFF polar).
_SUPPORTED_CHANNELS = (1, 2)


class EventGeometryMixin:
    """Reject a sensor whose geometry the topology cannot consume."""

    _architecture: Dict[str, Any]
    _spec: TopologySpec
    _topology: str
    _dataset: str
    _event_source: EventSampleSource

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

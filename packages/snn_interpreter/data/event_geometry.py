"""Error raised when an event sensor cannot feed a topology honestly.

An event dataset carries its own sensor geometry (for example N-MNIST is
34x34 and SSC is a linear sensor). A topology built for a different shape
cannot consume those spikes, and silently reshaping them would corrupt the
stream. This typed error names the dataset, the sensor, and the topology's
expected input so the mismatch is reported instead of smoothed over.
"""


class EventGeometryError(RuntimeError):
    """Raised when an event sensor's geometry mismatches the topology.

    The message is fully formed by the caller and names the dataset, the
    sensor shape, the expected input geometry, and the feature-input
    alternative, so a reader never has to infer why training was refused.
    """

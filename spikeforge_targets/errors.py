"""Typed failures raised by the deployment-targets distribution.

Quantized exports are always explicit: an unknown scheme or a weight tensor
no finite grid can represent raises a named :class:`QuantizationError`
rather than silently producing an unloadable summary.
"""


class QuantizationError(Exception):
    """A quantized export request could not be honoured as specified.

    ``detail`` carries the human-readable reason so a caller can name the
    refusal instead of guessing at a scheme that is not implemented.
    """

    def __init__(self, detail: str) -> None:
        """Record the refusal reason and render it in the message."""
        super().__init__(detail)
        self.detail = detail

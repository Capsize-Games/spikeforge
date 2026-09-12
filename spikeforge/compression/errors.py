"""Typed failures raised by the compression package.

Compression is always explicit: an unsupported scheme, an out-of-range
sparsity, or a malformed encoding raises a named :class:`CompressionError`
rather than silently producing an approximation the caller did not ask for.
"""


class CompressionError(Exception):
    """A compression request could not be honoured as specified.

    ``detail`` carries the human-readable reason so a report can name the
    refusal instead of guessing at a scheme that is not implemented.
    """

    def __init__(self, detail: str) -> None:
        """Record the refusal reason and render it in the message."""
        super().__init__(detail)
        self.detail = detail

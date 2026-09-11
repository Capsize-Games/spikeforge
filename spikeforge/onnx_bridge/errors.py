"""Typed errors raised by the ONNX interop bridge.

Mirroring :mod:`spikeforge.nir_bridge.errors`, the small exception
classes live together so a caller can import one module. Every failure carries
a machine-readable attribute (``op_type``, ``extra`` or ``path``) in addition
to a human message, so an unmappable op or a missing extra is never a bare
traceback.
"""


class OnnxBridgeError(Exception):
    """Base class for every ONNX bridge failure."""


class OnnxExtraMissingError(OnnxBridgeError):
    """Raised when the optional ``onnx`` extra is not installed."""

    def __init__(self, extra: str = "onnx") -> None:
        """Record ``extra`` and build an install hint."""
        message = (
            f"the {extra!r} extra is required for ONNX interop; "
            f"install it with: pip install spikeforge[{extra}]"
        )
        super().__init__(message)
        self.extra: str = extra


class UnsupportedOnnxOpError(OnnxBridgeError):
    """Raised when an ONNX op has no faithful SNN stage mapping.

    The offending ``op_type`` is stored as an attribute so callers can react
    without parsing the message.
    """

    def __init__(self, op_type: str, detail: str = "") -> None:
        """Record ``op_type`` and fold ``detail`` into the message."""
        message = f"ONNX op {op_type!r} has no faithful SNN stage mapping"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.op_type: str = op_type


class OnnxExportError(OnnxBridgeError):
    """Raised when a torch module cannot be exported to ONNX."""

    def __init__(self, detail: str) -> None:
        """Record ``detail`` and build a clear message."""
        super().__init__(f"ONNX export failed: {detail}")
        self.detail: str = detail


class OnnxGraphFileError(OnnxBridgeError):
    """Raised when an ONNX graph file cannot be read."""

    def __init__(self, path: str, detail: str) -> None:
        """Record ``path`` and ``detail`` and build a clear message."""
        super().__init__(f"{detail} (onnx file: {path})")
        self.path: str = path
        self.detail: str = detail


class OnnxGraphNotFoundError(OnnxGraphFileError):
    """Raised when no ONNX file exists at the requested path."""

    def __init__(self, path: str) -> None:
        """Record the missing ``path``."""
        super().__init__(path, "onnx file not found")

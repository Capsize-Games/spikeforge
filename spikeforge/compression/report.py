"""The JSON-able report of a weight-compression pass.

The report restates the encoding a bundle carries - the scheme, its bit-width,
the tensor and byte counts, the achieved ratio, and the quantize round-trip
error - in the same shape convention the ``spikeforge_targets`` quantization
reports use, so a caller can render either without special-casing. The
optional ``pruning`` section carries a :class:`~spikeforge.compression.pruning.
PruningReport` mapping when the weights were pruned before being compressed.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from spikeforge.compression.codec import WEIGHTS_ENCODING_VERSION


@dataclass(frozen=True)
class CompressionReport:
    """The recorded outcome of compressing a weight state dict."""

    scheme: str
    bits: int
    tensor_count: int
    element_count: int
    original_bytes: int
    compressed_bytes: int
    ratio: float
    error: Mapping[str, float] = field(default_factory=dict)
    pruning: Optional[Mapping[str, Any]] = field(default=None)
    version: int = WEIGHTS_ENCODING_VERSION

    def counts(self) -> Dict[str, int]:
        """Return the compressed tensor and element counts."""
        return {"tensors": self.tensor_count, "elements": self.element_count}

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-serialisable form of the report."""
        return {
            "version": self.version,
            "scheme": self.scheme,
            "bits": self.bits,
            "counts": self.counts(),
            "bytes": {
                "original": self.original_bytes,
                "compressed": self.compressed_bytes,
            },
            "ratio": self.ratio,
            "error": dict(self.error),
            "pruning": None if self.pruning is None else dict(self.pruning),
        }

    @classmethod
    def from_encoding(
        cls, encoding: Mapping[str, Any]
    ) -> "CompressionReport":
        """Return the report for a manifest ``weights_encoding`` block."""
        counts = encoding.get("counts") or {}
        sizes = encoding.get("bytes") or {}
        return cls(
            scheme=str(encoding.get("scheme", "none")),
            bits=int(encoding.get("bits", 0)),
            tensor_count=int(counts.get("tensors", 0)),
            element_count=int(counts.get("elements", 0)),
            original_bytes=int(sizes.get("original", 0)),
            compressed_bytes=int(sizes.get("compressed", 0)),
            ratio=float(encoding.get("ratio", 0.0)),
            error=dict(encoding.get("error") or {}),
            pruning=encoding.get("pruning"),
            version=int(encoding.get("version", WEIGHTS_ENCODING_VERSION)),
        )

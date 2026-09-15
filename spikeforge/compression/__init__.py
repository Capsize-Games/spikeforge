"""CPU-side weight compression for the production toolkit.

The package is pure torch plus stdlib: it never imports pydantic, fastapi, or
any optional SDK, so a headless core install can shrink a deployment bundle's
weights. It owns three things:

* :mod:`~spikeforge.compression.codec` - deterministic 8-bit weight
  quantize/dequantize with a documented manifest encoding;
* :mod:`~spikeforge.compression.pruning` - magnitude and structured pruning
  with an honest sparsity and drift report;
* :mod:`~spikeforge.compression.report` - the JSON-able compression report.
"""

from spikeforge.compression.codec import (
    SCHEME_INT8,
    SCHEME_NONE,
    SCHEME_UINT8,
    SCHEMES,
    WEIGHTS_ENCODING_VERSION,
    CompressedWeights,
    dequantize_tensor,
    quantize_tensor,
)
from spikeforge.compression.encoding import (
    compress_state_dict,
    dequantize_state_dict,
)
from spikeforge.compression.errors import CompressionError
from spikeforge.compression.pruned_weights import PrunedWeights
from spikeforge.compression.pruning import prune, sparsity
from spikeforge.compression.pruning_report import PruningReport
from spikeforge.compression.report import CompressionReport

__all__ = [
    "SCHEMES",
    "SCHEME_INT8",
    "SCHEME_NONE",
    "SCHEME_UINT8",
    "WEIGHTS_ENCODING_VERSION",
    "CompressedWeights",
    "CompressionError",
    "CompressionReport",
    "PrunedWeights",
    "PruningReport",
    "compress_state_dict",
    "dequantize_state_dict",
    "dequantize_tensor",
    "prune",
    "quantize_tensor",
    "sparsity",
]

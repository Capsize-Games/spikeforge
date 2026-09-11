"""Optional ONNX import/export bridge (Phase F2).

Only :mod:`spikeforge.onnx_bridge.api` imports ``onnx``/``onnxruntime``,
and every import is defensive, so importing this package without the ``onnx``
extra is always safe. :func:`export_topology` renders a topology's single
forward step to ONNX with the spec in metadata, :func:`roundtrip` proves the
artifact re-imports exactly, and :func:`import_report` maps a third-party
model's ops to stage kinds or fails with a typed error naming the op.
"""

from spikeforge.onnx_bridge.api import available, capability
from spikeforge.onnx_bridge.errors import (
    OnnxBridgeError,
    OnnxExportError,
    OnnxExtraMissingError,
    OnnxGraphFileError,
    OnnxGraphNotFoundError,
    UnsupportedOnnxOpError,
)
from spikeforge.onnx_bridge.export import (
    export_built,
    export_summary,
    export_topology,
    sample_input,
)
from spikeforge.onnx_bridge.import_onnx import (
    import_report,
    load_spec,
    spec_from_file,
    spec_from_ops,
)
from spikeforge.onnx_bridge.roundtrip import roundtrip

__all__ = [
    "OnnxBridgeError",
    "OnnxExportError",
    "OnnxExtraMissingError",
    "OnnxGraphFileError",
    "OnnxGraphNotFoundError",
    "UnsupportedOnnxOpError",
    "available",
    "capability",
    "export_built",
    "export_summary",
    "export_topology",
    "import_report",
    "load_spec",
    "roundtrip",
    "sample_input",
    "spec_from_file",
    "spec_from_ops",
]

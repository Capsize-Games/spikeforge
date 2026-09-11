"""Export a topology to ONNX and re-import it, reporting fidelity.

Because an export stamps the declarative spec into the model's metadata, the
round-trip is exact rather than approximate: ``identical`` is True only when
every stage and edge survives export and re-import unchanged.
"""

import os
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from spikeforge.onnx_bridge import export, import_onnx
from spikeforge.topology.registry import build_topology


@contextmanager
def _target_path(path: Optional[str]) -> Iterator[str]:
    """Yield ``path``, or a temporary file removed when the block exits."""
    if path is not None:
        yield path
        return
    handle, temporary = tempfile.mkstemp(suffix=".onnx")
    os.close(handle)
    try:
        yield temporary
    finally:
        os.remove(temporary)


def roundtrip(
    topology: str,
    path: Optional[str] = None,
    batch: int = export.DEFAULT_BATCH,
) -> Dict[str, Any]:
    """Export ``topology`` and re-import it, reporting whether it matches."""
    with _target_path(path) as target:
        report = export.export_topology(topology, target, batch=batch)
        spec, source = import_onnx.load_spec(target)
    original, _ = build_topology(topology)
    return {
        "path": report["path"],
        "topology": topology,
        "source": source,
        "identical": spec.to_dict() == original.to_dict(),
        "ops": report["ops"],
        "temporal": report["temporal"],
    }

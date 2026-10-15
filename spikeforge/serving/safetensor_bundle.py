"""One verified SafeTensors artifact a serving runtime can load."""

import io
import zipfile
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch

from spikeforge.serving.errors import BundleFormatError
from spikeforge.serving.st_format import (
    ENCODE_NAME,
    EXPECTED_NAME,
    GRAPH_NAME,
    MANIFEST_NAME,
    MODEL_NAME,
    PREPROCESSING_NAME,
    SUMS_NAME,
    WEIGHTS_NAME,
    dump_json,
    graph_entry,
    load_weights,
    read_archive,
    slot_json,
    sums_text,
)
from spikeforge.serving.st_verify import (
    check_manifest,
    verify_hashes,
    verify_meta,
    verify_sums,
    verify_tensors,
)
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec


def _verified_fields(
    path: str, payloads: Mapping[str, bytes], manifest: Mapping[str, Any],
    weights: Mapping[str, torch.Tensor], strict: bool,
) -> Mapping[str, Any]:
    """Return the verified constructor fields for a bundle load."""
    model = slot_json(path, payloads, 1)
    encode = slot_json(path, payloads, 2)
    verify_meta(path, manifest, model, encode, strict)
    graph = graph_entry(path, payloads) if GRAPH_NAME in payloads else None
    return {
        "manifest": manifest,
        "weights": weights,
        "encode_config": encode,
        "preprocessing": slot_json(path, payloads, 3),
        "model": model,
        "expected": slot_json(path, payloads, 4),
        "graph": graph,
    }


@dataclass(frozen=True)
class STBundle:
    """One verified SafeTensors artifact a runtime can load."""

    manifest: Mapping[str, Any]
    weights: Mapping[str, torch.Tensor]
    encode_config: Mapping[str, Any] = field(default_factory=dict)
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    model: Mapping[str, Any] = field(default_factory=dict)
    expected: Mapping[str, Any] = field(default_factory=dict)
    graph: Optional[Mapping[str, Any]] = None
    path: Optional[str] = None

    @property
    def spec(self) -> TopologySpec:
        """Return the topology spec the bundle rebuilds."""
        return TopologySpec.from_dict(self.model["spec"])

    @property
    def model_hash(self) -> str:
        """Return the manifest's recorded weights digest."""
        return str(self.manifest["model_hash"])

    def build_module(self, device: Any = "cpu") -> Any:
        """Return the manifest module with verified weights loaded."""
        module = build_module(self.spec)
        try:
            module.load_state_dict(dict(self.weights), strict=True)
        except RuntimeError as err:
            where = self.path or "<memory>"
            raise BundleFormatError(where, f"{err}") from None
        module.eval()
        return module.to(torch.device(device))

    def to_bytes(self) -> bytes:
        """Return the complete version 2 archive as bytes."""
        from safetensors.torch import save as pack

        entries = {
            MANIFEST_NAME: dump_json(self.manifest),
            WEIGHTS_NAME: pack(dict(self.weights)),
            ENCODE_NAME: dump_json(self.encode_config),
            PREPROCESSING_NAME: dump_json(self.preprocessing),
            MODEL_NAME: dump_json(self.model),
            EXPECTED_NAME: dump_json(self.expected),
        }
        if self.graph is not None:
            entries[GRAPH_NAME] = dump_json(self.graph)
        entries[SUMS_NAME] = sums_text(entries).encode()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for name, payload in entries.items():
                zipf.writestr(name, payload)
        return buffer.getvalue()

    def save(self, path: str) -> str:
        """Write the bundle to ``path`` and return it."""
        with open(path, "wb") as handle:
            handle.write(self.to_bytes())
        return path

    @classmethod
    def load(cls, path: str, strict: bool = True) -> "STBundle":
        """Read, verify, and return the bundle at ``path``."""
        payloads = read_archive(path)
        verify_sums(path, payloads)
        manifest = check_manifest(path, slot_json(path, payloads, 0))
        weights = load_weights(path, payloads[WEIGHTS_NAME])
        verify_tensors(path, manifest, weights)
        verify_hashes(path, manifest, payloads)
        return cls(
            **_verified_fields(path, payloads, manifest, weights, strict),
            path=path,
        )

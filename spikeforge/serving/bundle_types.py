"""One portable artifact a serving runtime can load."""

import io
import zipfile
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Union

import torch

from spikeforge.compression import (
    CompressionError,
    CompressionReport,
    dequantize_state_dict,
)
from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle_read import (
    _load_preprocessing,
    _load_weights,
    _read_bundle_parts,
)
from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.serving.errors import BundleFormatError
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec

_MEMORY = "<memory>"


def _weights_bytes(state_dict: Mapping[str, torch.Tensor]) -> bytes:
    """Return ``state_dict`` as a serialized ``weights.pt`` payload."""
    buffer = io.BytesIO()
    torch.save(dict(state_dict), buffer)
    return buffer.getvalue()


@dataclass(frozen=True)
class DeploymentBundle:
    """One portable artifact a serving runtime can load.

    It carries the resolved :class:`TopologySpec`, the trained weights, the
    frozen encode and preprocessing configs, and an optional NIR graph, with a
    manifest recording the library versions and the checkpoint's provenance.
    When ``weights_encoding`` is set the ``weights`` mapping holds integer
    codes rather than floats, and :meth:`resolved_weights` dequantizes them;
    a bundle without it is byte-for-byte a raw float bundle.
    """

    manifest: Mapping[str, Any]
    weights: Mapping[str, torch.Tensor]
    encode_config: Mapping[str, Any] = field(default_factory=dict)
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    graph: Optional[Mapping[str, Any]] = None
    path: Optional[str] = None
    weights_encoding: Optional[Mapping[str, Any]] = None

    @property
    def spec(self) -> TopologySpec:
        """Return the topology spec the bundle rebuilds."""
        return TopologySpec.from_dict(self.manifest["spec"])

    def has_encode(self) -> bool:
        """Return True when the bundle carries a frozen encode config."""
        return bool(self.encode_config)

    def encode_spec(self) -> EncodeSpec:
        """Return the frozen, validated encode spec.

        An empty config resolves to the defaults, so a bundle that predates
        the contract still answers with a usable spec.
        """
        spec = EncodeSpec.from_mapping(self.encode_config or {})
        spec.validate()
        return spec

    def preprocess_spec(self) -> EncodeSpec:
        """Return the encode spec after a version check (MVP: check only).

        The full preprocessing/windowing contract is deferred; for now this
        only asserts the frozen encode spec validates against this runtime's
        :data:`ENCODE_SPEC_VERSION`.
        """
        return self.encode_spec()

    def resolved_weights(self) -> Dict[str, torch.Tensor]:
        """Return the float state dict, dequantizing a compressed payload.

        A bundle with no ``weights_encoding`` returns its weights verbatim; a
        compressed bundle expands every recorded tensor through the codec's
        documented ``(code - zero_point) * scale`` rule. A malformed encoding
        is refused with :class:`CompressionError` rather than loaded as-is.
        """
        if not self.weights_encoding:
            return dict(self.weights)
        try:
            return dequantize_state_dict(self.weights, self.weights_encoding)
        except (KeyError, TypeError, ValueError) as error:
            raise CompressionError(
                f"weights_encoding is unreadable: {error}"
            ) from None

    def compression_report(self) -> Optional[CompressionReport]:
        """Return the compression report, or ``None`` for a raw bundle.

        A pruning block recorded alongside the encoding is folded in, so one
        report carries both the sparsity gained and the compressed ratio.
        """
        if not self.weights_encoding:
            return None
        report = CompressionReport.from_encoding(self.weights_encoding)
        pruning = self.manifest.get("pruning")
        if pruning is None:
            return report
        return replace(report, pruning=dict(pruning))

    def build_module(
        self, device: Union[str, torch.device] = "cpu"
    ) -> Any:
        """Return the module described by the manifest, weights loaded."""
        module = build_module(self.spec)
        try:
            module.load_state_dict(self.resolved_weights(), strict=True)
        except RuntimeError as error:
            raise BundleFormatError(
                self.path or _MEMORY,
                f"weights do not match the manifest spec: {error}",
            ) from None
        module.eval()
        return module.to(torch.device(device))

    def to_bytes(self) -> bytes:
        """Return the complete ``.spkf`` archive as bytes."""
        entries: Dict[str, bytes] = {
            bm.MANIFEST_NAME: bm.dump_json(self.manifest),
            bm.WEIGHTS_NAME: _weights_bytes(self.weights),
            bm.ENCODE_NAME: bm.dump_json(self.encode_config),
            bm.PREPROCESSING_NAME: bm.dump_json(self.preprocessing),
        }
        if self.graph is not None:
            entries[bm.GRAPH_NAME] = bm.dump_json(self.graph)
        entries[bm.SUMS_NAME] = bm.sums_text(entries).encode("utf-8")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, payload in entries.items():
                archive.writestr(name, payload)
        return buffer.getvalue()

    def save(self, path: str) -> str:
        """Write the bundle to ``path`` and return it."""
        with open(path, "wb") as handle:
            handle.write(self.to_bytes())
        return path

    def export_c_package(self, path: str) -> str:
        """Write this bundle's C-loadable package into ``path``.

        The package is a derived artifact (the ``.spkf`` format is
        unchanged): the NIR graph in ``graph_summary`` shape with this
        bundle's real weights as plain float arrays, plus a manifest
        recording the source bundle's identity. Needs the ``nir`` extra;
        a graph the C runtime cannot load is refused with a typed error
        naming the offending nodes and kinds instead of writing an
        unloadable package.
        """
        from spikeforge.serving.c_package import write_c_package

        return write_c_package(self, path)

    @classmethod
    def load(cls, path: str, strict: bool = True) -> "DeploymentBundle":
        """Read, verify, and return the bundle at ``path``.

        Rejected with a typed error when missing, malformed, tampered, or
        built against an incompatible runtime.
        """
        manifest, encode_config, graph, payloads = _read_bundle_parts(
            path, strict
        )
        return cls(
            manifest=manifest,
            weights=_load_weights(path, payloads[bm.WEIGHTS_NAME]),
            encode_config=encode_config,
            preprocessing=_load_preprocessing(path, payloads),
            graph=graph,
            path=path,
            weights_encoding=manifest.get("weights_encoding"),
        )

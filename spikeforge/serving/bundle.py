"""Build, persist, verify, and load the portable deployment bundle."""

import io
import json
import os
import platform
import tempfile
import zipfile
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from spikeforge.compression import (
    SCHEME_NONE,
    CompressionError,
    CompressionReport,
    compress_state_dict,
    dequantize_state_dict,
    prune,
)
from spikeforge.network import model_store
from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.encode_spec import (
    ENCODE_SPEC_VERSION,
    EncodeSpec,
)
from spikeforge.serving.errors import (
    BundleCompatibilityError,
    BundleFormatError,
    BundleIntegrityError,
    BundleNotFoundError,
    EncodeSpecError,
)
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec
from spikeforge.tracking.versions import library_versions

_MEMORY = "<memory>"


def _major_minor(version: Any) -> Optional[str]:
    """Return the ``major.minor`` prefix of ``version``, or None."""
    if not version:
        return None
    parts = str(version).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(version)


def _versions() -> Dict[str, Optional[str]]:
    """Return the runtime versions recorded in a bundle manifest."""
    versions: Dict[str, Optional[str]] = dict(library_versions())
    versions["python"] = platform.python_version()
    return versions


def _load_json(path: str, payload: bytes, name: str) -> Any:
    """Return the JSON value in ``payload``, or raise a typed error."""
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleFormatError(
            path, f"{name} is not valid JSON: {error}"
        ) from None


def _load_weights(path: str, payload: bytes) -> Mapping[str, torch.Tensor]:
    """Return the state dict in ``payload``, or raise a typed error."""
    try:
        weights = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=True
        )
    except Exception as error:  # noqa: BLE001 - any load failure is a format error
        raise BundleFormatError(
            path, f"weights are not loadable: {error}"
        ) from None
    if not isinstance(weights, dict):
        raise BundleFormatError(path, "weights payload is not a state dict")
    return weights


def _weights_bytes(state_dict: Mapping[str, torch.Tensor]) -> bytes:
    """Return ``state_dict`` as a serialized ``weights.pt`` payload."""
    buffer = io.BytesIO()
    torch.save(dict(state_dict), buffer)
    return buffer.getvalue()


def _verify_integrity(path: str, payloads: Mapping[str, bytes]) -> None:
    """Raise when any entry is missing, tampered, or unlisted."""
    try:
        expected = bm.parse_sums(payloads[bm.SUMS_NAME].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise BundleFormatError(
            path, f"SHA256SUMS is unreadable: {error}"
        ) from None
    for name, digest in expected.items():
        if name not in payloads:
            raise BundleIntegrityError(path, name, "listed but absent")
        if bm.checksum(payloads[name]) != digest:
            raise BundleIntegrityError(path, name)
    unlisted = set(payloads) - set(expected) - {bm.SUMS_NAME}
    if unlisted:
        raise BundleIntegrityError(
            path, sorted(unlisted)[0], "present but not checksummed"
        )


def _check_compatibility(
    path: str, recorded: Mapping[str, Any], strict: bool
) -> None:
    """Raise when the runtime disagrees with a bundle under ``strict``."""
    if not strict:
        return
    current = _versions()
    for key in ("torch", "python"):
        want = _major_minor(recorded.get(key))
        have = _major_minor(current.get(key))
        if want and have and want != have:
            raise BundleCompatibilityError(
                path, f"{key} {have!r} does not match the bundle's {want!r}"
            )


def _parse_encode_spec(path: str, encode_config: Any) -> EncodeSpec:
    """Parse and validate an encode spec, wrapping a parse error typed."""
    spec = EncodeSpec.from_mapping(encode_config or {})
    try:
        spec.validate()
    except EncodeSpecError as error:
        raise BundleFormatError(
            path, f"invalid encode spec: {error}"
        ) from None
    return spec


def _check_encode_version(
    path: str, recorded: Optional[Any], strict: bool
) -> None:
    """Raise when ``recorded`` mismatches the runtime's spec version."""
    if recorded is None:
        return
    if int(recorded) != ENCODE_SPEC_VERSION and strict:
        raise BundleCompatibilityError(
            path,
            f"encode spec version {recorded!r} does not match the runtime's "
            f"{ENCODE_SPEC_VERSION!r}",
        )


def _check_encode(
    path: str, manifest: Mapping[str, Any], encode_config: Any, strict: bool
) -> EncodeSpec:
    """Validate a bundle's frozen encode spec and pin its contract version.

    An empty ``encode_config`` (a legacy in-memory bundle) resolves to the
    defaults, preserving the current behaviour. A non-empty spec must parse
    and validate, and its manifest ``encode_spec_version`` must match this
    runtime; a mismatch is refused under ``strict`` rather than re-encoded.
    """
    spec = _parse_encode_spec(path, encode_config)
    _check_encode_version(path, manifest.get("encode_spec_version"), strict)
    return spec


def _meta_input_size(meta: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    """Return the frozen sensor ``(H, W)`` geometry a checkpoint records.

    The normalised ``encode_spec`` a trained checkpoint stores is preferred;
    a legacy checkpoint may instead carry an ``input_size`` on the meta or the
    topology params. Only an explicit pair is a geometry, so a flat feature
    count (which is not an ``(H, W)`` sensor) is ignored.
    """
    frozen = meta.get("encode_spec") or {}
    candidates = (
        frozen.get("input_size"),
        meta.get("input_size"),
        (meta.get("topology_params") or {}).get("input_size"),
    )
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
    return None


def _check_decompressed_size(
    path: str, archive: zipfile.ZipFile, names: Any
) -> None:
    """Raise when ``names``' total decompressed size exceeds the cap.

    Checked against each entry's recorded ``file_size`` before any entry is
    decompressed, so a small compressed payload declaring an enormous
    decompressed size (a zip bomb) is refused rather than read into memory.
    """
    total = sum(archive.getinfo(name).file_size for name in names)
    if total > bm.MAX_DECOMPRESSED_BYTES:
        raise BundleFormatError(
            path,
            f"decompressed size {total} exceeds the "
            f"{bm.MAX_DECOMPRESSED_BYTES} byte cap",
        )


def _read_archive(path: str) -> Dict[str, bytes]:
    """Return every entry of the zip at ``path``, or raise a typed error."""
    if not os.path.exists(path):
        raise BundleNotFoundError(path)
    try:
        with zipfile.ZipFile(path) as archive:
            names = {n for n in archive.namelist() if not n.endswith("/")}
            missing = [n for n in bm.REQUIRED_ENTRIES if n not in names]
            if missing:
                raise BundleFormatError(
                    path, f"missing required entries: {missing}"
                )
            _check_decompressed_size(path, archive, names)
            return {name: archive.read(name) for name in names}
    except zipfile.BadZipFile as error:
        raise BundleFormatError(
            path, f"not a zip archive: {error}"
        ) from None


def _load_module(
    spec_dict: Mapping[str, Any], state_dict: Mapping[str, torch.Tensor]
) -> Tuple[TopologySpec, Any]:
    """Build the topology module and load its trained weights."""
    spec = TopologySpec.from_dict(spec_dict)
    module = build_module(spec)
    module.load_state_dict(dict(state_dict), strict=True)
    return spec, module


def _nir_envelope(
    spec_dict: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    source: str,
) -> Mapping[str, Any]:
    """Return the in-memory NIR graph envelope for the bundle."""
    from spikeforge.nir_bridge import api as nir_api
    from spikeforge.nir_bridge import serialization
    from spikeforge.nir_bridge.exporter import to_nir

    if not nir_api.available():
        raise BundleFormatError(
            source, "the 'nir' extra is required to include a NIR graph"
        )
    spec, module = _load_module(spec_dict, state_dict)
    with tempfile.TemporaryDirectory() as folder:
        graph_path = os.path.join(folder, bm.GRAPH_NAME)
        serialization.save_graph(to_nir(spec, module), graph_path)
        with open(graph_path, encoding="utf-8") as handle:
            return json.load(handle)


def _read_bundle_parts(
    path: str, strict: bool
) -> Tuple[Dict[str, Any], Any, Optional[Any], Dict[str, bytes]]:
    """Read and validate every manifest-adjacent part of a bundle archive."""
    payloads = _read_archive(path)
    _verify_integrity(path, payloads)
    manifest = bm.validate_manifest(
        _load_json(path, payloads[bm.MANIFEST_NAME], "manifest"), path
    )
    _check_compatibility(path, manifest.get("library_versions") or {}, strict)
    encode_config = _load_json(
        path, payloads[bm.ENCODE_NAME], "encode_config"
    )
    _check_encode(path, manifest, encode_config, strict)
    graph = None
    if bm.GRAPH_NAME in payloads:
        graph = _load_json(path, payloads[bm.GRAPH_NAME], "graph")
    return manifest, encode_config, graph, payloads


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
            preprocessing=_load_json(
                path, payloads[bm.PREPROCESSING_NAME], "preprocessing"
            ),
            graph=graph,
            path=path,
            weights_encoding=manifest.get("weights_encoding"),
        )


def _checkpoint_parts(
    checkpoint: str,
) -> Any:
    """Return ``(meta, spec, state_dict, provenance)`` from a checkpoint."""
    stored = model_store.load(checkpoint)
    meta = dict(stored.get("meta") or {})
    spec = meta.get("spec")
    if not isinstance(spec, dict):
        raise BundleFormatError(checkpoint, "checkpoint has no topology spec")
    state_dict = stored.get("state_dict")
    if state_dict is None:
        raise BundleFormatError(checkpoint, "checkpoint has no state_dict")
    return meta, spec, state_dict, dict(stored.get("manifest") or {})


def _frozen_encode(
    encode_config: Optional[Mapping[str, Any]], meta: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the canonical encode config frozen into a bundle.

    An explicit ``encode_config`` wins; otherwise the checkpoint's own
    ``meta['encode']`` is used. Either way the config is normalised through
    :class:`EncodeSpec` so the stored JSON always carries ``spec_version``,
    and the sensor geometry is injected when the checkpoint knows it but the
    config does not.
    """
    base = encode_config if encode_config is not None else meta.get("encode")
    resolved = EncodeSpec.from_mapping(base)
    if resolved.input_size is None:
        size = _meta_input_size(meta)
        if size is not None:
            resolved = replace(resolved, input_size=size)
    resolved.validate()
    return resolved.to_dict()


#: (stored, encoding, pruning, float weights) returned by ``_prepare_weights``.
WeightsPrep = Tuple[
    Dict[str, Any],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Dict[str, torch.Tensor],
]


def _apply_pruning(
    resolved: Dict[str, Any], prune_sparsity: Optional[float], strategy: str
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Prune ``resolved`` when requested; return it and the report block."""
    if prune_sparsity is None:
        return resolved, None
    pruned = prune(resolved, prune_sparsity, strategy=strategy)
    return dict(pruned.tensors), pruned.report.to_dict()


def _uncompressed_weights(
    resolved: Dict[str, Any],
) -> Tuple[Dict[str, Any], None, Dict[str, torch.Tensor]]:
    """Return ``resolved`` unchanged, with its float tensors picked out."""
    floats = {
        name: value
        for name, value in resolved.items()
        if torch.is_tensor(value)
    }
    return resolved, None, floats


def _compressed_weights(
    resolved: Dict[str, Any], compress: str, compress_bits: int
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, torch.Tensor]]:
    """Compress ``resolved`` and return the stored/encoding/float triple."""
    compressed = compress_state_dict(resolved, compress, compress_bits)
    encoding = dict(compressed.encoding)
    stored = dict(compressed.tensors)
    return stored, encoding, dequantize_state_dict(stored, encoding)


def _prepare_weights(
    state_dict: Mapping[str, Any],
    compress: Optional[str],
    compress_bits: int,
    prune_sparsity: Optional[float],
    prune_strategy: str,
) -> WeightsPrep:
    """Prune then compress ``state_dict``; return stored/encoding/floats."""
    resolved, pruning = _apply_pruning(
        dict(state_dict), prune_sparsity, prune_strategy
    )
    if not compress or compress == SCHEME_NONE:
        stored, encoding, floats = _uncompressed_weights(resolved)
    else:
        stored, encoding, floats = _compressed_weights(
            resolved, compress, compress_bits
        )
    return stored, encoding, pruning, floats


def _manifest_source_fields(
    spec: Any, meta: Dict[str, Any], provenance: Any, encode: Dict[str, Any]
) -> Dict[str, Any]:
    """Return the checkpoint/encode-derived manifest fields."""
    return {
        "spec": spec,
        "meta": meta,
        "provenance": provenance,
        "versions": _versions(),
        "encode_config": encode,
        "encode_spec_version": encode["spec_version"],
    }


def _manifest_option_fields(
    preprocess: Dict[str, Any],
    label_map: Optional[Mapping[int, str]],
    expected_metrics: Optional[Mapping[str, Any]],
    protocol_version: Optional[str],
    encoding: Optional[Dict[str, Any]],
    pruning: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the caller-supplied option fields of the manifest."""
    return {
        "preprocessing": preprocess,
        "label_map": dict(label_map or {}),
        "expected_metrics": dict(expected_metrics or {}),
        "protocol_version": protocol_version,
        "weights_encoding": encoding,
        "pruning": pruning,
    }


def _build_manifest(
    spec: Any, meta: Dict[str, Any], provenance: Any, encode: Dict[str, Any],
    preprocess: Dict[str, Any], label_map: Optional[Mapping[int, str]],
    expected_metrics: Optional[Mapping[str, Any]],
    protocol_version: Optional[str], encoding: Optional[Dict[str, Any]],
    pruning: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the manifest block for a new bundle."""
    fields = _manifest_source_fields(spec, meta, provenance, encode)
    fields.update(
        _manifest_option_fields(
            preprocess, label_map, expected_metrics, protocol_version,
            encoding, pruning,
        )
    )
    return bm.new_manifest(**fields)


def _finalize_bundle(
    manifest: Dict[str, Any], stored: Dict[str, Any], encode: Dict[str, Any],
    preprocess: Dict[str, Any], graph: Optional[Any], out: Optional[str],
    encoding: Optional[Dict[str, Any]],
) -> DeploymentBundle:
    """Construct the bundle and, when ``out`` is given, save it."""
    bundle = DeploymentBundle(
        manifest=manifest,
        weights=stored,
        encode_config=encode,
        preprocessing=preprocess,
        graph=graph,
        path=out,
        weights_encoding=encoding,
    )
    if out:
        bundle.save(out)
    return bundle


def build(
    checkpoint: str,
    out: Optional[str] = None,
    encode_config: Optional[Mapping[str, Any]] = None,
    preprocessing: Optional[Mapping[str, Any]] = None,
    label_map: Optional[Mapping[int, str]] = None,
    expected_metrics: Optional[Mapping[str, Any]] = None,
    include_nir: bool = False,
    protocol_version: Optional[str] = None,
    compress: Optional[str] = None,
    compress_bits: int = 8,
    prune_sparsity: Optional[float] = None,
    prune_strategy: str = "unstructured",
) -> DeploymentBundle:
    """Build a :class:`DeploymentBundle` from a saved checkpoint.

    The topology spec, topology name, and resolved parameters are read from
    the checkpoint's own metadata, so the artifact is self-describing.
    ``out`` also writes the ``.spkf`` archive. ``include_nir`` adds the NIR
    graph envelope and needs the ``nir`` extra. ``compress`` names an 8-bit
    weight scheme (``int8``/``uint8``) and ``prune_sparsity`` an optional
    magnitude/structured pruning level; either records its report in the
    manifest so the reduction and its cost are visible.
    """
    meta, spec, state_dict, provenance = _checkpoint_parts(checkpoint)
    encode = _frozen_encode(encode_config, meta)
    preprocess = dict(preprocessing or {})
    stored, encoding, pruning, resolved = _prepare_weights(
        state_dict, compress, compress_bits, prune_sparsity, prune_strategy
    )
    manifest = _build_manifest(
        spec, meta, provenance, encode, preprocess, label_map,
        expected_metrics, protocol_version, encoding, pruning,
    )
    graph = None
    if include_nir:
        graph = _nir_envelope(spec, resolved, checkpoint)
    return _finalize_bundle(
        manifest, stored, encode, preprocess, graph, out, encoding
    )

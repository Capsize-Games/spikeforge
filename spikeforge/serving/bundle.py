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


def _check_encode(
    path: str, manifest: Mapping[str, Any], encode_config: Any, strict: bool
) -> EncodeSpec:
    """Validate a bundle's frozen encode spec and pin its contract version.

    An empty ``encode_config`` (a legacy in-memory bundle) resolves to the
    defaults, preserving the current behaviour. A non-empty spec must parse
    and validate, and its manifest ``encode_spec_version`` must match this
    runtime; a mismatch is refused under ``strict`` rather than re-encoded.
    """
    spec = EncodeSpec.from_mapping(encode_config or {})
    try:
        spec.validate()
    except EncodeSpecError as error:
        raise BundleFormatError(
            path, f"invalid encode spec: {error}"
        ) from None
    recorded = manifest.get("encode_spec_version")
    if recorded is None:
        return spec
    if int(recorded) != ENCODE_SPEC_VERSION and strict:
        raise BundleCompatibilityError(
            path,
            f"encode spec version {recorded!r} does not match the runtime's "
            f"{ENCODE_SPEC_VERSION!r}",
        )
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
            return {name: archive.read(name) for name in names}
    except zipfile.BadZipFile as error:
        raise BundleFormatError(
            path, f"not a zip archive: {error}"
        ) from None


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
    spec = TopologySpec.from_dict(spec_dict)
    module = build_module(spec)
    module.load_state_dict(dict(state_dict), strict=True)
    with tempfile.TemporaryDirectory() as folder:
        graph_path = os.path.join(folder, bm.GRAPH_NAME)
        serialization.save_graph(to_nir(spec, module), graph_path)
        with open(graph_path, encoding="utf-8") as handle:
            return json.load(handle)


@dataclass(frozen=True)
class DeploymentBundle:
    """One portable artifact a serving runtime can load.

    It carries the resolved :class:`TopologySpec`, the trained weights, the
    frozen encode and preprocessing configs, and an optional NIR graph, with a
    manifest recording the library versions and the checkpoint's provenance.
    """

    manifest: Mapping[str, Any]
    weights: Mapping[str, torch.Tensor]
    encode_config: Mapping[str, Any] = field(default_factory=dict)
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    graph: Optional[Mapping[str, Any]] = None
    path: Optional[str] = None

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

    def build_module(
        self, device: Union[str, torch.device] = "cpu"
    ) -> Any:
        """Return the module described by the manifest, weights loaded."""
        module = build_module(self.spec)
        try:
            module.load_state_dict(dict(self.weights), strict=True)
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

        The archive is rejected with a typed error when it is missing
        (:class:`BundleNotFoundError`), malformed
        (:class:`BundleFormatError`), tampered
        (:class:`BundleIntegrityError`), or built against an incompatible
        runtime (:class:`BundleCompatibilityError`).
        """
        payloads = _read_archive(path)
        _verify_integrity(path, payloads)
        manifest = bm.validate_manifest(
            _load_json(path, payloads[bm.MANIFEST_NAME], "manifest"), path
        )
        _check_compatibility(
            path, manifest.get("library_versions") or {}, strict
        )
        encode_config = _load_json(
            path, payloads[bm.ENCODE_NAME], "encode_config"
        )
        _check_encode(path, manifest, encode_config, strict)
        graph = None
        if bm.GRAPH_NAME in payloads:
            graph = _load_json(path, payloads[bm.GRAPH_NAME], "graph")
        return cls(
            manifest=manifest,
            weights=_load_weights(path, payloads[bm.WEIGHTS_NAME]),
            encode_config=encode_config,
            preprocessing=_load_json(
                path, payloads[bm.PREPROCESSING_NAME], "preprocessing"
            ),
            graph=graph,
            path=path,
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


def build(
    checkpoint: str,
    out: Optional[str] = None,
    encode_config: Optional[Mapping[str, Any]] = None,
    preprocessing: Optional[Mapping[str, Any]] = None,
    label_map: Optional[Mapping[int, str]] = None,
    expected_metrics: Optional[Mapping[str, Any]] = None,
    include_nir: bool = False,
    protocol_version: Optional[str] = None,
) -> DeploymentBundle:
    """Build a :class:`DeploymentBundle` from a saved checkpoint.

    The topology spec, topology name, and resolved parameters are read from
    the checkpoint's own metadata, so the artifact is self-describing.
    ``out`` also writes the ``.spkf`` archive. ``include_nir`` adds the NIR
    graph envelope and needs the ``nir`` extra.
    """
    meta, spec, state_dict, provenance = _checkpoint_parts(checkpoint)
    encode = _frozen_encode(encode_config, meta)
    preprocess = dict(preprocessing or {})
    manifest = bm.new_manifest(
        spec=spec,
        meta=meta,
        provenance=provenance,
        versions=_versions(),
        encode_config=encode,
        preprocessing=preprocess,
        label_map=dict(label_map or {}),
        expected_metrics=dict(expected_metrics or {}),
        protocol_version=protocol_version,
        encode_spec_version=encode["spec_version"],
    )
    graph = None
    if include_nir:
        graph = _nir_envelope(spec, state_dict, checkpoint)
    bundle = DeploymentBundle(
        manifest=manifest,
        weights=dict(state_dict),
        encode_config=encode,
        preprocessing=preprocess,
        graph=graph,
        path=out,
    )
    if out:
        bundle.save(out)
    return bundle

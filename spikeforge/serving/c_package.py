"""Export a deployment bundle as a C-loadable package.

A ``.spkf`` bundle needs Python and torch (a pickled ``weights.pt`` and a
Python-envelope graph), so it cannot be consumed on a torch-free target.
This module derives a plain-JSON package from a bundle instead: the NIR
graph in :func:`graph_summary` shape with the bundle's real weights, plus
a manifest recording the source bundle's identity. The ``.spkf`` format
itself is unchanged; the package is a derived artifact.

The gate is the single-sourced ``nir-c-runtime`` profile owned by
:mod:`spikeforge_targets.c_runtime`: a graph the C runtime cannot load is
refused with :class:`CExportError` naming each offending node and kind,
so an unloadable package is never written silently.
"""

import os
import time
from typing import TYPE_CHECKING, Any, Dict, Mapping

from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.errors import BundleFormatError, CExportError
from spikeforge.targets_extra import require

if TYPE_CHECKING:
    from spikeforge.serving.bundle_types import DeploymentBundle

#: Marker identifying a derived C package.
PACKAGE_FORMAT = "spikeforge-c-package"
#: Package version; bump when the file set changes incompatibly.
PACKAGE_VERSION = 1
#: The graph summary file the C loader parses.
GRAPH_FILENAME = "graph.json"
#: The package manifest recording the source bundle's identity.
MANIFEST_FILENAME = "manifest.json"


def __getattr__(name: str) -> Any:
    """Resolve ``SUPPORTED_NODE_KINDS`` from the C runtime profile.

    Backward-compatible alias for the single-sourced kind set owned by
    :mod:`spikeforge_targets.c_runtime`, imported lazily (PEP 562) so
    this module stays importable on a headless core install.
    """
    if name != "SUPPORTED_NODE_KINDS":
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    kinds = require("spikeforge_targets.c_runtime").SUPPORTED
    globals()[name] = kinds
    return kinds


def _summary_with_real_weights(
    bundle: "DeploymentBundle",
) -> Dict[str, Any]:
    """Return ``bundle``'s graph summary with its trained weights."""
    from spikeforge.nir_bridge import api as nir_api
    from spikeforge.nir_bridge.exporter import graph_summary, to_nir

    source = bundle.path or "<memory>"
    if not nir_api.available():
        raise BundleFormatError(
            source, "the 'nir' extra is required to export a C package"
        )
    module = bundle.build_module()
    return dict(graph_summary(to_nir(bundle.spec, module)))


def _refuse_unloadable(path: str, summary: Mapping[str, Any]) -> None:
    """Raise :class:`CExportError` unless the C runtime loads ``summary``.

    The check runs by node kind and by every structural C-loader
    assumption, so a package the C loader rejects is never written.
    """
    validator = require("spikeforge_targets.c_runtime_validate")
    refusal = require("spikeforge_targets.c_runtime_errors")
    try:
        validator.check_for_c_runtime(summary)
    except refusal.CRuntimeUnsupportedError as exc:
        found = [
            (item.name, item.kind, item.reason)
            for item in exc.violations
        ]
        raise CExportError(path, exc.kinds, found) from exc


def _bundle_checksum(bundle: "DeploymentBundle") -> Any:
    """Return the source file's SHA-256, or ``None`` when unavailable.

    An in-memory bundle has no file, and a moved source is no reason to
    refuse an export, so either answers honestly with ``None`` while the
    manifest checksum below still identifies the content.
    """
    if bundle.path is None:
        return None
    try:
        with open(bundle.path, "rb") as handle:
            return bm.checksum(handle.read())
    except OSError:
        return None


def _source_identity(bundle: "DeploymentBundle") -> Dict[str, Any]:
    """Return the source bundle's identity block for the manifest."""
    return {
        "bundle_path": bundle.path,
        "bundle_sha256": _bundle_checksum(bundle),
        "manifest_sha256": bm.checksum(bm.dump_json(bundle.manifest)),
    }


def _package_manifest(
    bundle: "DeploymentBundle", summary: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the manifest describing a package of ``summary``."""
    return {
        "format": PACKAGE_FORMAT,
        "version": PACKAGE_VERSION,
        "created_at": time.time(),
        "graph": GRAPH_FILENAME,
        "nodes": len(summary["nodes"]),
        "edges": len(summary["edges"]),
        "source": _source_identity(bundle),
    }


def write_c_package(bundle: "DeploymentBundle", path: str) -> str:
    """Write ``bundle``'s C package into ``path`` and return ``path``.

    ``path`` is created when missing. The graph carries the bundle's
    real (dequantized, when compressed) weights as plain float arrays;
    a graph the C runtime cannot load raises :class:`CExportError`
    naming the offending nodes and kinds instead of writing a package
    the C loader rejects.
    """
    summary = _summary_with_real_weights(bundle)
    _refuse_unloadable(bundle.path or "<memory>", summary)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, GRAPH_FILENAME), "wb") as handle:
        handle.write(bm.dump_json(summary))
    manifest = _package_manifest(bundle, summary)
    with open(os.path.join(path, MANIFEST_FILENAME), "wb") as handle:
        handle.write(bm.dump_json(manifest))
    return path

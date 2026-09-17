"""Run the inspect -> compat -> NIR-export-and-drift funnel on a bundle.

Reuses :mod:`spikeforge_hub.inspect`, :mod:`spikeforge_hub.compat`, and
:func:`spikeforge.nir_bridge.validate` exactly as
:mod:`spikeforge_hub.import_model` runs them for the curated catalog
(``rules.md`` invariant 4: validation stays independent of the module it
checks, so a drift check is a genuine cross-check). A bundle already
carries its own explicit :class:`~spikeforge.topology.spec.TopologySpec`,
so unlike a bare downloaded artifact this funnel never has to guess a
topology before it can build a module: ``compat`` here is informational
(does this map onto a *known, registered* preset?), not a gate -- the
drift check always runs against the bundle's own declared architecture.
"""

import os
from typing import Any, Dict

import torch

from hub_verify.errors import VerificationStepError
from hub_verify.fixtures import probe_spikes
from spikeforge.nir_bridge import validate
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge_hub.compat import classify
from spikeforge_hub.inspect import inspect_artifact


def _write_weights(bundle: DeploymentBundle, tmp_dir: str) -> str:
    """Write the bundle's already-verified weights to a bare ``.pt`` file.

    The bytes have already passed a ``weights_only=True`` load in
    :func:`hub_verify.pipeline.load_and_build`; re-loading the identical
    bytes through :mod:`spikeforge_hub.inspect`'s bare-torch reader is
    safe, because that reload can only reconstruct the same objects
    already proven benign by the stricter load.
    """
    path = os.path.join(tmp_dir, "weights.pt")
    torch.save(dict(bundle.weights), path)
    return path


def _drift_check(bundle: DeploymentBundle, module: Any) -> Dict[str, Any]:
    """Run the reference-interpreter drift check, raising on drift."""
    spikes = probe_spikes(bundle.spec)
    result = validate(bundle.spec, module, spikes)
    if not result["within_tolerance"]:
        worst = result.get("worst") or {}
        where = worst.get("layer", "unknown layer")
        what = worst.get("quantity", "unknown quantity")
        raise VerificationStepError(
            "drift",
            "the exported NIR graph drifts from the trained module "
            f"at {where} ({what})",
        )
    return dict(result)


def run_funnel(
    bundle: DeploymentBundle, module: Any, tmp_dir: str
) -> Dict[str, Any]:
    """Return the inspect/compat/drift block of the verification report."""
    weights_path = _write_weights(bundle, tmp_dir)
    report = inspect_artifact(weights_path)
    verdict = classify(report, bundle.manifest.get("topology"))
    return {
        "inspect": {"kind": report.kind},
        "compat": verdict.to_dict(),
        "drift": _drift_check(bundle, module),
    }

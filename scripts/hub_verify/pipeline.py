"""Load and build the ``.spkf`` bundle under verification.

Wraps :class:`spikeforge.serving.bundle.DeploymentBundle` so a load or
build failure becomes a named
:class:`~hub_verify.errors.VerificationStepError` instead of an unhandled
traceback. The checksum/manifest verification and the
``weights_only=True`` weight load are exactly the ones ``bundle.py``
already performs for every bundle load (non-negotiable for untrusted
content, per ``plans/hub_accounts_plan.md`` §7.2) -- nothing here
reimplements them.
"""

from typing import Any, Tuple

from hub_verify.errors import VerificationStepError
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.errors import BundleError


def load_and_build(path: str) -> Tuple[DeploymentBundle, Any]:
    """Return ``(bundle, module)`` for the ``.spkf`` archive at ``path``.

    :meth:`DeploymentBundle.load` runs the checksum/manifest verification
    and the ``weights_only=True`` weight load; :meth:`build_module` then
    strictly loads those weights into the module the manifest's own spec
    describes -- a second, independent proof the weights actually fit the
    declared architecture, not just that they deserialize.
    """
    try:
        bundle = DeploymentBundle.load(path, strict=True)
        module = bundle.build_module()
    except BundleError as error:
        raise VerificationStepError("bundle", error.detail) from error
    return bundle, module

"""Isolated probes and imports for the optional backend SDKs.

This is the only package in the backend tree that imports a backend SDK.
Imports happen inside functions, never at module import time, so a missing
package never breaks importing the backends; callers read
:func:`module_available` first and degrade honestly. Norse neuron factories
and the Lava execution touch-point are wrapped here so upstream API churn is
contained to one place.

This package is the import root: the probes live in
:mod:`spikeforge_targets.backends.api.probes`, the Norse factories in
:mod:`spikeforge_targets.backends.api.norse`, the Lava runner in
:mod:`spikeforge_targets.backends.api.lava`, and the vendor simulators in
:mod:`spikeforge_targets.backends.api.vendors`. Everything is re-exported
here so importers keep using this path.
"""

from spikeforge_targets.backends.api.lava import (
    LAVA_DEVICE_ENV,
    LAVA_DEVICE_LEGACY_ENV,
    lava_device_present,
    lava_run,
)
from spikeforge_targets.backends.api.norse import norse_li, norse_lif
from spikeforge_targets.backends.api.probes import (
    BACKEND_CAPABILITIES,
    BACKEND_MODULES,
    backend_available,
    module_available,
    module_with_capability,
    require,
    version,
)
from spikeforge_targets.backends.api.vendors import (
    NEURON_KINDS,
    rockpool_run,
    sinabs_run,
    spinnaker2_run,
)

__all__ = [
    "BACKEND_CAPABILITIES",
    "BACKEND_MODULES",
    "LAVA_DEVICE_ENV",
    "LAVA_DEVICE_LEGACY_ENV",
    "NEURON_KINDS",
    "backend_available",
    "lava_device_present",
    "lava_run",
    "module_available",
    "module_with_capability",
    "norse_li",
    "norse_lif",
    "require",
    "rockpool_run",
    "sinabs_run",
    "spinnaker2_run",
    "version",
]

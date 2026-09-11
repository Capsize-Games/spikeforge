"""Deprecated alias for :mod:`snn_targets.energy`.

ARCH-0001 Phase 3 moved ``snn_interpreter.energy`` to the ``snn-targets``
distribution and the ``snn_targets.energy`` import root. This module
re-exports the new package and is kept for one minor release; import
``snn_targets.energy`` directly instead.
"""

import warnings

warnings.warn(
    "snn_interpreter.energy moved to snn_targets.energy; "
    "install the 'snn-targets' distribution.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from snn_targets.energy import *  # noqa: F403
except ModuleNotFoundError as error:  # pragma: no cover - satellite absent
    raise ImportError(
        "snn_interpreter.energy moved to the 'snn-targets' distribution "
        "(import root 'snn_targets.energy'); install it with "
        "`pip install snn-targets`."
    ) from error

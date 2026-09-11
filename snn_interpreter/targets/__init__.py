"""Deprecated alias for :mod:`snn_targets`.

ARCH-0001 Phase 3 moved ``snn_interpreter.targets`` to the ``snn-targets``
distribution and the ``snn_targets`` import root. This module re-exports the
new package and is kept for one minor release; import ``snn_targets`` directly
instead.
"""

import warnings

warnings.warn(
    "snn_interpreter.targets moved to snn_targets; "
    "install the 'snn-targets' distribution.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from snn_targets import *  # noqa: F403
except ModuleNotFoundError as error:  # pragma: no cover - satellite absent
    raise ImportError(
        "snn_interpreter.targets moved to the 'snn-targets' distribution "
        "(import root 'snn_targets'); install it with "
        "`pip install snn-targets`."
    ) from error

"""Deprecated alias for :mod:`snn_hub`.

ARCH-0001 Phase 4 moved ``snn_interpreter.hub`` to the ``snn-hub``
distribution and the ``snn_hub`` import root. This module re-exports the
new package and is kept for one minor release; import ``snn_hub`` directly
instead.
"""

import warnings

warnings.warn(
    "snn_interpreter.hub moved to snn_hub; "
    "install the 'snn-hub' distribution.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from snn_hub import *  # noqa: F403
except ModuleNotFoundError as error:  # pragma: no cover - satellite absent
    raise ImportError(
        "snn_interpreter.hub moved to the 'snn-hub' distribution "
        "(import root 'snn_hub'); install it with "
        "`pip install snn-hub`."
    ) from error

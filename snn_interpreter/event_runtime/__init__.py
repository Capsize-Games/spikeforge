"""Deprecated alias for :mod:`snn_targets.event_runtime`.

ARCH-0001 Phase 3 moved ``snn_interpreter.event_runtime`` to the
``snn-targets`` distribution and the ``snn_targets.event_runtime`` import
root. This module re-exports the new package and is kept for one minor
release; import ``snn_targets.event_runtime`` directly instead.
"""

import warnings

warnings.warn(
    "snn_interpreter.event_runtime moved to snn_targets.event_runtime; "
    "install the 'snn-targets' distribution.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from snn_targets.event_runtime import *  # noqa: F403
except ModuleNotFoundError as error:  # pragma: no cover - satellite absent
    raise ImportError(
        "snn_interpreter.event_runtime moved to the 'snn-targets' "
        "distribution (import root 'snn_targets.event_runtime'); install it "
        "with `pip install snn-targets`."
    ) from error

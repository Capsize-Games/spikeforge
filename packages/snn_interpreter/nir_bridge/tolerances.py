"""Default drift tolerances for the snnTorch <-> NIR validation.

Spike trains are expected to match exactly (agreement ``1.0``, no differing
events). Membrane potentials and readouts carry only float32 rounding, so a
small epsilon absorbs the order of floating-point operations without hiding
a real mismatch.
"""

from typing import Dict, Mapping, Optional

#: Tolerances keyed by ``<quantity>_<metric>``.
DEFAULT_TOLERANCES: Dict[str, float] = {
    "spike_max_abs": 0.0,
    "spike_mean_abs": 0.0,
    "spike_agreement": 1.0,
    "membrane_max_abs": 1e-5,
    "membrane_mean_abs": 1e-5,
    "readout_max_abs": 1e-6,
    "readout_mean_abs": 1e-6,
}


def resolved(
    overrides: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Return the default tolerances updated with ``overrides``."""
    settings = dict(DEFAULT_TOLERANCES)
    if overrides:
        settings.update(overrides)
    return settings

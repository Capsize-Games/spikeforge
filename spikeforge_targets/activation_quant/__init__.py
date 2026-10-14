"""Simulated fixed-point activation and membrane quantization.

The core step path stays scheme-agnostic: it accepts an optional ``post_step``
transform, and this package supplies one. :class:`ActivationQuantizer` is that
transform. It maps every floating activation (a stage output, and the
previous-step mirror a delayed edge reads) and every floating membrane or
input-current tensor in the carried state onto a symmetric fixed-point grid
with ``2 ** (bits - 1) - 1`` levels, recording the per-stage ranges and the
absolute error the grid introduced.

Weight-only quantization remains the default everywhere; nothing is applied
until a caller passes a quantizer to
:meth:`spikeforge.serving.session.InferenceSession.load`. A scheme name this
package does not implement is refused in the report rather than approximated,
and ``none`` is an explicit no-op. :func:`calibrate` is the calibration-dataset
hook: feed it observed tensors once, then hand the resulting ranges to the
quantizer so the grid is chosen from data instead of per-step extremes.

The same schemes, grid, and report shape serve the NIR side: the
:class:`~spikeforge_targets.activation_quant_graph.GraphActivationQuantizer`
hook applies them to a reference-interpreter run.

This package is the import root: the schemes live in
:mod:`spikeforge_targets.activation_quant.scheme`, the calibration in
:mod:`spikeforge_targets.activation_quant.calibration`, and the quantizer in
:mod:`spikeforge_targets.activation_quant.quantizer`. Everything is
re-exported here so importers keep using this path.
"""

from spikeforge_targets.activation_quant.calibration import (
    Calibration,
    Ranges,
    calibrate,
    calibrated_bound,
    fold_range,
    is_float,
)
from spikeforge_targets.activation_quant.quantizer import ActivationQuantizer
from spikeforge_targets.activation_quant.scheme import (
    NO_ACTIVATION_SCHEME,
    SCHEMES,
    TARGET_ACTIVATION,
    TARGET_BOTH,
    TARGET_MEMBRANE,
    TARGETS,
    UNKNOWN_ACTIVATION_SCHEME,
    FixedPointScheme,
    refused_report,
    scheme_for,
    scheme_names,
    scheme_report,
)

__all__ = [
    "NO_ACTIVATION_SCHEME",
    "SCHEMES",
    "TARGET_ACTIVATION",
    "TARGET_BOTH",
    "TARGET_MEMBRANE",
    "TARGETS",
    "UNKNOWN_ACTIVATION_SCHEME",
    "ActivationQuantizer",
    "Calibration",
    "FixedPointScheme",
    "Ranges",
    "calibrate",
    "calibrated_bound",
    "fold_range",
    "is_float",
    "refused_report",
    "scheme_for",
    "scheme_names",
    "scheme_report",
]

"""Stage kinds with no faithful NIR rendering in the installed ``nir``.

Each entry names the honest reason export cannot represent the kind, mirroring
the ``alpha`` neuron precedent: the mapper raises the typed
:class:`~snn_interpreter.nir_bridge.errors.UnsupportedStageError` naming the
kind and this detail rather than dropping or approximating the stage.
"""

from typing import Dict

_NO_NORM = "the installed nir has no normalisation primitive"
_NO_ATTENTION = "the installed nir has no attention primitive"
_NO_MAXPOOL = "the installed nir pooling primitives are average and sum only"

#: Kind -> reason it cannot be mapped to the installed ``nir``.
UNMAPPABLE_STAGES: Dict[str, str] = {
    "embedding": (
        "the installed nir has no Embedding primitive, so a token lookup "
        "table cannot be represented; the stage is simulation-only"
    ),
    "maxpool1d": _NO_MAXPOOL,
    "maxpool2d": _NO_MAXPOOL,
    "layer_norm": _NO_NORM,
    "batch_norm": _NO_NORM,
    "positional_encoding": (
        "the installed nir has no constant node to carry a fixed sinusoidal "
        "encoding; the stage is simulation-only"
    ),
    "attention": _NO_ATTENTION,
    "multihead_attention": _NO_ATTENTION,
}

"""Windowing and normalization surface for ``spikeforge-io``.

The plan places "windowing/normalization" on the ``spikeforge-io``
distribution. The *implementation*, however, stays in the core
:mod:`spikeforge.streaming.window_spec` module, because core must never depend
on a satellite: the training recipe, the serving session, and the UC-1 tests
all consume the frozen :class:`~spikeforge.streaming.window_spec.WindowSpec`
directly. This module therefore **re-exports** the one canonical contract
rather than copying it, so there is a single definition of "how a stream
becomes a normalized window"; ``spikeforge_io.windows(...)`` is byte-for-byte
the core behaviour.

A caller that only wants the I/O surface can import everything here:

    from spikeforge_io.windowing import WindowSpec, window_stream
"""

from typing import Any

from spikeforge.streaming.encoding import (
    delta_over_window,
    encode_windows,
)
from spikeforge.streaming.window_spec import (
    WINDOW_SPEC_VERSION,
    WindowSpec,
    fit_window_spec,
    normalize_windows,
    window_stream,
)

__all__ = [
    "WINDOW_SPEC_VERSION",
    "WindowSpec",
    "delta_over_window",
    "encode_windows",
    "fit_window_spec",
    "normalize_windows",
    "window_spec_from_bundle",
    "window_stream",
]


def window_spec_from_bundle(bundle: Any) -> WindowSpec:
    """Return the frozen window contract a bundle carries.

    A bundle whose ``preprocessing`` block has no ``window`` entry cannot be
    windowed by this helper; the caller must fit a spec instead. The returned
    spec is validated, so a corrupt recorded contract is refused rather than
    used.
    """
    preprocessing = dict(getattr(bundle, "preprocessing", None) or {})
    recorded = preprocessing.get("window")
    if not recorded:
        raise ValueError(
            "bundle carries no frozen window contract; fit a WindowSpec "
            "with fit_window_spec"
        )
    spec = WindowSpec.from_dict(recorded)
    spec.validate()
    return spec


def windows(bundle: Any, stream: Any) -> Any:
    """Return ``stream``'s normalized ``[W, L, D]`` windows for ``bundle``.

    This is the one-call path an adapter uses: the window geometry and the
    z-score statistics both come from the bundle's frozen contract, so a
    recorded stream is normalized exactly as training was.
    """
    return window_spec_from_bundle(bundle).windows(stream)

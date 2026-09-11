"""Normalise an image or sensor geometry declaration to an ``(H, W)`` pair.

A square sensor may be declared as a single ``int`` (its side, the historical
form); any other geometry is an ``(H, W)`` pair. Both normalise here so the
transform, the presets, and the simulator reshape agree on one convention.
"""

from typing import Tuple, Union

#: A geometry declared as a square side or an explicit ``(H, W)`` pair.
SizeLike = Union[int, Tuple[int, int]]
#: The historical 28x28 grayscale sample geometry.
DEFAULT_SIZE: Tuple[int, int] = (28, 28)


def as_size(value: SizeLike) -> Tuple[int, int]:
    """Return ``value`` as an ``(H, W)`` pair; an int means a square side."""
    if isinstance(value, int):
        return (int(value), int(value))
    height, width = value
    return (int(height), int(width))

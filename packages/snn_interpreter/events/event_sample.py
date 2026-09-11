"""Sparse neuromorphic event streams and their coordinate conventions.

An :class:`EventSample` holds an asynchronous address-event stream:

* ``x`` is the 0-based column index (width axis): ``0 <= x < W``.
* ``y`` is the 0-based row index (height axis): ``0 <= y < H``.
* ``t`` is the 0-based time-bin index: ``0 <= t < num_steps``.
* ``p`` is the polarity: ``+1`` for an ON (brightness increase) event and
  ``-1`` for an OFF (brightness decrease) event.

Events are unordered; ``shape`` is the ``(H, W)`` sensor layout and
``num_steps`` the number of bins a dense accumulation spans. Every dense
form is time-major ``[T, ...]`` so it feeds the simulator's single loop.
"""

from typing import Tuple

import torch


class EventSample:
    """A validated sparse event stream in ``(x, y, t, p)`` form."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        p: torch.Tensor,
        shape: Tuple[int, int],
        num_steps: int,
    ) -> None:
        """Validate and store the coordinate tensors and sensor layout."""
        self._x = _index(x, "x")
        self._y = _index(y, "y")
        self._t = _index(t, "t")
        self._p = _index(p, "p")
        self._shape = (int(shape[0]), int(shape[1]))
        self._num_steps = int(num_steps)
        self._check()

    def _check(self) -> None:
        """Raise ``ValueError`` when the stream breaks a convention."""
        if len({item.numel() for item in self._tensors()}) != 1:
            raise ValueError("x/y/t/p must have equal length")
        self._check_layout()
        self._check_ranges()

    def _check_layout(self) -> None:
        """Require a positive sensor shape and at least one time bin."""
        height, width = self._shape
        if height <= 0 or width <= 0 or self._num_steps < 1:
            raise ValueError("shape and num_steps must be positive")

    def _check_ranges(self) -> None:
        """Require in-range x/y/t indices and a valid polarity."""
        height, width = self._shape
        _check_range(self._x, width, "x")
        _check_range(self._y, height, "y")
        _check_range(self._t, self._num_steps, "t")
        _check_polarity(self._p)

    def _tensors(self) -> Tuple[torch.Tensor, ...]:
        """Return the coordinate tensors in ``x, y, t, p`` order."""
        return (self._x, self._y, self._t, self._p)

    @property
    def x(self) -> torch.Tensor:
        """Return the 0-based column index of every event."""
        return self._x

    @property
    def y(self) -> torch.Tensor:
        """Return the 0-based row index of every event."""
        return self._y

    @property
    def t(self) -> torch.Tensor:
        """Return the 0-based time-bin index of every event."""
        return self._t

    @property
    def p(self) -> torch.Tensor:
        """Return the polarity (``+1`` ON, ``-1`` OFF) of every event."""
        return self._p

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the ``(H, W)`` sensor layout."""
        return self._shape

    @property
    def num_steps(self) -> int:
        """Return the number of time bins a dense form spans."""
        return self._num_steps

    @property
    def num_events(self) -> int:
        """Return the number of events in the stream."""
        return int(self._x.numel())


def _index(value: torch.Tensor, name: str) -> torch.Tensor:
    """Return ``value`` as a flat 1-D long index tensor."""
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    return value.reshape(-1).long()


def _check_range(values: torch.Tensor, size: int, name: str) -> None:
    """Raise unless every ``values`` entry lies in ``[0, size)``."""
    if values.numel() == 0:
        return
    if int(values.min()) < 0 or int(values.max()) >= size:
        raise ValueError(f"{name} must be within [0, {size})")


def _check_polarity(values: torch.Tensor) -> None:
    """Raise unless every polarity is ON (``+1``) or OFF (``-1``)."""
    if values.numel() == 0:
        return
    valid = (values == 1) | (values == -1)
    if not bool(valid.all()):
        raise ValueError("p must be +1 (ON) or -1 (OFF)")

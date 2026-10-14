"""DoS bounds on serve request frames.

``frames_from`` rejects an over-count or oversized request before
any tensor is allocated from its frames.
"""

from typing import Any

import pytest

from spikeforge_serve.payloads import (
    MAX_FRAME_ELEMENTS,
    MAX_FRAMES,
    frames_from,
)


def test_small_frame_passes() -> None:
    """A realistic 28x28 raw sample is accepted unchanged."""
    sample = [[[0.0] * 28] * 28]
    assert frames_from({"frames": sample}) == sample


def test_max_frame_count_accepted() -> None:
    """Exactly MAX_FRAMES tiny frames are still accepted."""
    payload = {"frames": [[0.0]] * MAX_FRAMES}
    assert len(frames_from(payload)) == MAX_FRAMES


def test_frame_count_capped() -> None:
    """One frame past MAX_FRAMES is rejected, unexpanded."""
    payload = {"frames": [[0.0]] * (MAX_FRAMES + 1)}
    with pytest.raises(ValueError, match="at most"):
        frames_from(payload)


def test_huge_frame_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame past the element cap fails before tensor conversion."""
    monkeypatch.setattr(
        "spikeforge_serve.payloads.MAX_FRAME_ELEMENTS", 16
    )
    with pytest.raises(ValueError, match="exceeds"):
        frames_from({"frames": [[0.0] * 17]})


def test_capped_elements_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame exactly at the element cap is still accepted."""
    monkeypatch.setattr(
        "spikeforge_serve.payloads.MAX_FRAME_ELEMENTS", 16
    )
    assert frames_from({"frames": [[0.0] * 16]}) == [[0.0] * 16]


def test_ten_megabyte_payload_rejected() -> None:
    """A ~10MB single-frame payload is refused under the real cap."""
    ten_mb_of_floats = 10 * 1024 * 1024 // 4
    assert ten_mb_of_floats > MAX_FRAME_ELEMENTS
    with pytest.raises(ValueError, match="exceeds"):
        frames_from({"frames": [[0.0] * ten_mb_of_floats]})


def test_deep_nesting_rejected() -> None:
    """A nesting bomb is refused without recursing into it."""
    frame: Any = [0.0]
    for _ in range(32):
        frame = [frame]
    with pytest.raises(ValueError, match="nesting"):
        frames_from({"frames": [frame]})


def test_huge_string_frame_rejected() -> None:
    """A 10MB string frame counts its length against the cap."""
    with pytest.raises(ValueError, match="exceeds"):
        frames_from({"frames": ["x" * (10 * 1024 * 1024)]})


def test_empty_frames_still_rejected() -> None:
    """The pre-existing empty/non-list guards are unchanged."""
    with pytest.raises(ValueError):
        frames_from({"frames": []})
    with pytest.raises(ValueError):
        frames_from({"frames": "nope"})

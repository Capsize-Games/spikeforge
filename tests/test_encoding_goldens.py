"""Issue #160 acceptance: committed golden vectors for the spike codings.

Each test re-encodes a committed fixture input through the frozen
encode path and compares against the committed output bit-for-bit, so
any change to the encoder math or the seeding order fails loudly. The
fixtures double as the oracle a future C encoder verifies against;
``documentation/encoding-goldens.md`` specifies the math they pin.
"""

import json
from pathlib import Path
from typing import Any, Dict

import torch

from spikeforge.serving import preprocess
from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.streaming.encoding import delta_over_window
from spikeforge.streaming.window_spec import WindowSpec

_CASES = Path(__file__).resolve().parent / "fixtures" / "encoding_goldens"

_CODINGS = ("rate", "latency", "delta", "random")


def _load(name: str) -> Dict[str, Any]:
    """Return the parsed ``<name>.json`` golden case."""
    return json.loads((_CASES / f"{name}.json").read_text(encoding="utf-8"))


def _tensor(payload: Dict[str, Any]) -> torch.Tensor:
    """Return ``payload`` as a float32 tensor of its recorded shape."""
    return torch.tensor(payload["values"], dtype=torch.float32).reshape(
        *payload["shape"]
    )


def _check_spikes(actual: torch.Tensor, case: Dict[str, Any]) -> None:
    """Assert ``actual`` equals the case's committed spike tensor."""
    expected = _tensor(case["output"])
    assert list(actual.shape) == case["output"]["shape"]
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_rate_matches_golden() -> None:
    """A seeded rate code reproduces its committed spike tensor."""
    case = _load("rate")
    _check_spikes(preprocess.encode(_tensor(case["input"]), case["spec"]),
                  case)


def test_latency_matches_golden() -> None:
    """The deterministic latency code reproduces its golden tensor."""
    case = _load("latency")
    _check_spikes(preprocess.encode(_tensor(case["input"]), case["spec"]),
                  case)


def test_delta_matches_golden() -> None:
    """The deterministic delta code reproduces its golden tensor."""
    case = _load("delta")
    _check_spikes(preprocess.encode(_tensor(case["input"]), case["spec"]),
                  case)


def test_random_matches_golden() -> None:
    """A seeded random code reproduces its committed spike tensor."""
    case = _load("random")
    _check_spikes(preprocess.encode(_tensor(case["input"]), case["spec"]),
                  case)


def test_delta_over_window_matches_golden() -> None:
    """The first-difference pre-step reproduces its golden output."""
    case = _load("delta_over_window")
    actual = delta_over_window(_tensor(case["input"]))
    assert list(actual.shape) == case["output"]["shape"]
    assert torch.equal(actual, _tensor(case["output"]))


def test_window_normalize_matches_golden() -> None:
    """A frozen spec z-scores the fixture windows onto the golden."""
    case = _load("window_normalize")
    spec = WindowSpec.from_dict(case["spec"])
    assert spec.digest() == case["spec_digest"]
    actual = spec.normalize(_tensor(case["input"]))
    assert list(actual.shape) == case["output"]["shape"]
    assert torch.equal(actual, _tensor(case["output"]))


def test_coding_spec_digests_match() -> None:
    """Each committed spec digest matches the spec it was frozen from."""
    for coding in _CODINGS:
        case = _load(coding)
        assert EncodeSpec.from_dict(case["spec"]).digest() == (
            case["spec_digest"]
        )


def test_rate_seed_change_breaks_match() -> None:
    """A different seed must not reproduce the rate golden (sensitivity)."""
    case = _load("rate")
    spec = dict(case["spec"])
    spec["random_seed"] = int(spec["random_seed"]) + 1
    other = preprocess.encode(_tensor(case["input"]), spec)
    assert not torch.equal(other, _tensor(case["output"]))


def test_random_seed_change_breaks_match() -> None:
    """A different seed must not reproduce the random golden."""
    case = _load("random")
    spec = dict(case["spec"])
    spec["random_seed"] = int(spec["random_seed"]) + 1
    other = preprocess.encode(_tensor(case["input"]), spec)
    assert not torch.equal(other, _tensor(case["output"]))


def test_random_ignores_input_pixels() -> None:
    """The random code depends on the input shape, not its pixels."""
    case = _load("random")
    zeros = torch.zeros_like(_tensor(case["input"]))
    _check_spikes(preprocess.encode(zeros, case["spec"]), case)

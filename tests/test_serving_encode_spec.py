"""The frozen, versioned encode contract backing train/serve parity."""

import pytest

from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.serving.encode_spec import (
    CODINGS,
    ENCODE_SPEC_VERSION,
    EncodeSpec,
)
from spikeforge.serving.errors import EncodeSpecError


def test_defaults_are_valid_and_carry_a_version() -> None:
    """The default spec is usable and pinned to the runtime's version."""
    spec = EncodeSpec()
    spec.validate()
    assert spec.coding == "rate"
    assert spec.spec_version == ENCODE_SPEC_VERSION
    assert spec.to_dict()["spec_version"] == ENCODE_SPEC_VERSION


def test_round_trip_preserves_every_field() -> None:
    """A canonical dict reloads into an equal spec."""
    spec = EncodeSpec(
        coding="latency",
        num_steps=7,
        tau=3.0,
        input_size=(32, 24),
        random_seed=11,
    )
    assert EncodeSpec.from_dict(spec.to_dict()) == spec


def test_unknown_keys_are_ignored() -> None:
    """Parsing is tolerant, matching the schema's additionalProperties."""
    spec = EncodeSpec.from_dict(
        {"coding": "delta", "num_steps": 6, "dataset": "mnist", "junk": 1}
    )
    assert spec.coding == "delta"
    assert spec.num_steps == 6
    assert "junk" not in spec.to_dict()


def test_from_mapping_accepts_none_and_model_dump() -> None:
    """``None`` yields defaults; a ``model_dump()``-able config is read."""

    class _Config:
        def model_dump(self) -> dict:
            return {"coding": "latency", "num_steps": 9}

    assert EncodeSpec.from_mapping(None) == EncodeSpec()
    parsed = EncodeSpec.from_mapping(_Config())
    assert parsed.coding == "latency"
    assert parsed.num_steps == 9


def test_rejects_an_unknown_coding() -> None:
    """A coding outside ``CODINGS`` is refused, never silently accepted."""
    with pytest.raises(EncodeSpecError):
        EncodeSpec.from_dict({"coding": "bogus"}).validate()


@pytest.mark.parametrize("value", [0, -1])
def test_rejects_a_non_positive_num_steps(value: int) -> None:
    """``num_steps`` must be at least one."""
    with pytest.raises(EncodeSpecError):
        EncodeSpec(num_steps=value).validate()


@pytest.mark.parametrize("value", [0.0, -2.5])
def test_rejects_a_non_positive_tau(value: float) -> None:
    """``tau`` must be strictly positive."""
    with pytest.raises(EncodeSpecError):
        EncodeSpec(tau=value).validate()


def test_rejects_an_unsupported_spec_version() -> None:
    """A version this runtime cannot speak is refused."""
    with pytest.raises(EncodeSpecError):
        EncodeSpec(spec_version=ENCODE_SPEC_VERSION + 1).validate()


def test_accepts_an_input_size_pair() -> None:
    """A two-int geometry is accepted and dumped as a JSON pair."""
    spec = EncodeSpec(input_size=(32, 24))
    spec.validate()
    assert spec.to_dict()["input_size"] == [32, 24]


@pytest.mark.parametrize(
    "size",
    [(0, 28), (28, 0), (28,), (28, 28, 1), (-1, 28)],
)
def test_rejects_bad_geometry(size: tuple) -> None:
    """A geometry that is not a pair of positive ints is refused."""
    with pytest.raises(EncodeSpecError):
        EncodeSpec(input_size=size).validate()


def test_digest_is_stable_and_key_order_independent() -> None:
    """The digest depends on values, not on mapping insertion order."""
    first = EncodeSpec.from_dict(
        {"coding": "latency", "num_steps": 5, "tau": 5.0}
    )
    second = EncodeSpec.from_dict(
        {"tau": 5.0, "num_steps": 5, "coding": "latency"}
    )
    third = EncodeSpec(coding="latency", num_steps=5)
    assert first.digest() == second.digest() == third.digest()


def test_digest_changes_with_parameters() -> None:
    """A different frozen parameter yields a different digest."""
    assert EncodeSpec(num_steps=5).digest() != EncodeSpec(num_steps=6).digest()


def test_unsupported_fields_are_reported_not_dropped() -> None:
    """Inert fields are listed honestly and still survive ``to_dict``."""
    spec = EncodeSpec(coding="rate", tau=9.0)
    unsupported = spec.unsupported()
    assert "tau" in unsupported
    assert "num_steps" not in unsupported
    assert spec.to_dict()["tau"] == 9.0


def test_unsupported_is_coding_specific() -> None:
    """A latency code applies ``tau`` but not ``delta_threshold``."""
    latency = EncodeSpec(coding="latency")
    assert "tau" not in latency.unsupported()
    assert "delta_threshold" in latency.unsupported()


def test_to_encoder_delegates_to_the_core_encoder() -> None:
    """The spec builds the same encoder the core owns, without new math."""
    encoder = EncodeSpec(coding="latency", num_steps=7, tau=3.0).to_encoder()
    assert isinstance(encoder, SpikeEncoder)
    assert encoder.coding == "latency"
    assert encoder.num_steps == 7
    assert encoder.tau == 3.0


def test_codings_constant_lists_the_supported_names() -> None:
    """The public coding tuple is the encoder's own set."""
    assert CODINGS == ("rate", "latency", "delta", "random")

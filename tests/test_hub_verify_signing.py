"""HMAC signing of the outgoing verification report."""

import pytest
from hub_verify import signing


def test_matching_secret_verifies() -> None:
    """A signature computed and checked with the same secret matches."""
    body = b'{"passed": true}'
    header = signing.sign_body(body, "sekrit")
    assert header.startswith("sha256=")
    assert signing.signature_matches(body, "sekrit", header)


def test_wrong_secret_does_not_verify() -> None:
    """A signature checked under a different secret does not match."""
    body = b'{"passed": true}'
    header = signing.sign_body(body, "sekrit")
    assert not signing.signature_matches(body, "wrong", header)


def test_tampered_body_does_not_verify() -> None:
    """A body byte changed after signing breaks the signature."""
    body = b'{"passed": true}'
    header = signing.sign_body(body, "sekrit")
    tampered = b'{"passed": false}'
    assert not signing.signature_matches(tampered, "sekrit", header)


def test_empty_secret_is_refused() -> None:
    """Signing under an empty secret is refused rather than silently weak."""
    with pytest.raises(ValueError):
        signing.sign_body(b"{}", "")

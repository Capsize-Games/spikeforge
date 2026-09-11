"""Opt-in JSON logging: parseable lines, human fallback, clean reset."""

import io
import json
import logging
from typing import Iterator

import pytest

from spikeforge.observability.logging_setup import (
    LOGGER_NAME,
    configure_logging,
    logging_enabled,
    reset_logging,
)

_LOGGER = logging.getLogger(LOGGER_NAME)


@pytest.fixture(autouse=True)
def _clean_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear the logging environment and always reset the logger."""
    for name in ("SPIKEFORGE_LOG_JSON", "SPIKEFORGE_LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)
    reset_logging()
    yield
    reset_logging()


def test_json_mode_emits_parseable_lines_with_expected_keys() -> None:
    """A record rendered in JSON mode carries the documented keys."""
    stream = io.StringIO()
    configure_logging(json_mode=True, force=True, stream=stream)
    _LOGGER.info(
        "train.started",
        extra={
            "run_id": "run-1",
            "config_hash": "abc",
            "fields": {"epoch": 2},
        },
    )
    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "train.started"
    assert payload["level"] == "INFO"
    assert payload["run_id"] == "run-1"
    assert payload["config_hash"] == "abc"
    assert payload["fields"] == {"epoch": 2}
    assert "timestamp" in payload


def test_json_mode_reads_the_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SPIKEFORGE_LOG_JSON`` alone enables JSON output."""
    monkeypatch.setenv("SPIKEFORGE_LOG_JSON", "1")
    assert logging_enabled() is True
    stream = io.StringIO()
    assert configure_logging(stream=stream) is not None
    _LOGGER.warning("boom")
    assert json.loads(stream.getvalue().strip())["level"] == "WARNING"


def test_json_disabled_stays_human_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With only a level set the output stays human-readable text."""
    monkeypatch.setenv("SPIKEFORGE_LOG_LEVEL", "INFO")
    stream = io.StringIO()
    configure_logging(stream=stream)
    _LOGGER.info("hello world")
    text = stream.getvalue().strip()
    assert "hello world" in text
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_no_environment_means_no_configuration() -> None:
    """Nothing is attached unless asked, so defaults are unchanged."""
    assert logging_enabled() is False
    assert configure_logging() is None


def test_reset_restores_logger_without_leaking() -> None:
    """Reset returns the logger to its exact pre-configuration state."""
    before = (_LOGGER.level, _LOGGER.propagate, len(_LOGGER.handlers))
    configure_logging(json_mode=True, force=True, stream=io.StringIO())
    assert _LOGGER.handlers
    reset_logging()
    after = (_LOGGER.level, _LOGGER.propagate, len(_LOGGER.handlers))
    assert after == before

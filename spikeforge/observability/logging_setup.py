"""Opt-in structured logging that is reversible and never forced.

Nothing here runs at import time and no entry point must call it, so the
default (unconfigured) logging behaviour is unchanged. Enable JSON lines
with ``SPIKEFORGE_LOG_JSON=1`` and a level with
``SPIKEFORGE_LOG_LEVEL=DEBUG``, then call :func:`configure_logging`;
:func:`reset_logging` removes the handler and
restores the logger exactly as it was.
"""

import logging
import os
from typing import Any, Optional, Tuple

from spikeforge.observability.json_formatter import JsonFormatter

#: Package logger structured logging attaches to (never the root logger).
LOGGER_NAME = "spikeforge"
#: Default level when neither the argument nor ``SPIKEFORGE_LOG_LEVEL`` is set.
DEFAULT_LEVEL = "INFO"
#: Values that turn an environment flag off.
_FALSEY = ("", "0", "false", "no", "off")
_HUMAN_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

_logger = logging.getLogger(LOGGER_NAME)
_handler: Optional[logging.Handler] = None
_saved: Optional[Tuple[int, bool]] = None


def _truthy(value: Optional[str]) -> bool:
    """Return True when an environment flag is set to a truthy value."""
    if value is None:
        return False
    return value.strip().lower() not in _FALSEY


def logging_enabled() -> bool:
    """Return True when the environment asks for logging to be configured."""
    if _truthy(os.environ.get("SPIKEFORGE_LOG_JSON")):
        return True
    return bool(os.environ.get("SPIKEFORGE_LOG_LEVEL"))


def _level(name: Optional[str]) -> int:
    """Resolve a level name from the argument, environment, or default."""
    resolved = name or os.environ.get("SPIKEFORGE_LOG_LEVEL") or DEFAULT_LEVEL
    return getattr(logging, str(resolved).upper(), logging.INFO)


def _handler_for(use_json: bool, stream: Any) -> logging.Handler:
    """Return a stream handler carrying the JSON or human formatter."""
    handler = logging.StreamHandler(stream)
    formatter = (
        JsonFormatter() if use_json else logging.Formatter(_HUMAN_FORMAT)
    )
    handler.setFormatter(formatter)
    return handler


def _use_json(json_mode: Optional[bool]) -> bool:
    """Resolve the JSON decision from the argument or the environment."""
    if json_mode is not None:
        return json_mode
    return _truthy(os.environ.get("SPIKEFORGE_LOG_JSON"))


def _install(
    use_json: bool, level: Optional[str], stream: Any
) -> logging.Handler:
    """Install the handler and apply the resolved level and formatter."""
    global _handler, _saved
    _saved = (_logger.level, _logger.propagate)
    _handler = _handler_for(use_json, stream)
    _logger.addHandler(_handler)
    _logger.setLevel(_level(level))
    _logger.propagate = False
    return _handler


def configure_logging(
    level: Optional[str] = None,
    json_mode: Optional[bool] = None,
    force: bool = False,
    stream: Any = None,
) -> Optional[logging.Handler]:
    """Attach the opt-in handler, or return None when logging is disabled.

    ``level``/``json_mode`` override
    ``SPIKEFORGE_LOG_LEVEL``/``SPIKEFORGE_LOG_JSON``;
    ``force`` configures even when no environment variable is set, and
    ``stream`` redirects output (used by tests). Repeating the call is safe:
    any previously installed handler is removed first.
    """
    if not force and not logging_enabled():
        return None
    reset_logging()
    return _install(_use_json(json_mode), level, stream)


def reset_logging() -> None:
    """Remove the installed handler and restore the previous logger state."""
    global _handler, _saved
    if _handler is not None:
        _logger.removeHandler(_handler)
        _handler = None
    if _saved is not None:
        _logger.setLevel(_saved[0])
        _logger.propagate = _saved[1]
        _saved = None

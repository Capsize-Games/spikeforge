"""A logging formatter that renders each record as one JSON line."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

#: Record attributes copied into the payload when the caller supplies them.
_IDENTIFIERS = ("run_id", "config_id", "config_hash")


class JsonFormatter(logging.Formatter):
    """Format a log record as a single JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the record as compact, parseable JSON."""
        return json.dumps(self._payload(record), default=str)

    def _payload(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Assemble the timestamp/level/event and identifier fields."""
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "logger": record.name,
        }
        for key in _IDENTIFIERS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        fields = getattr(record, "fields", None)
        if fields:
            payload["fields"] = fields
        return payload

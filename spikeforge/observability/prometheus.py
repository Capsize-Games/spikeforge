"""Prometheus text-exposition exporter for the in-process metrics registry.

This is the core-side, dependency-free half of PT-W6: it renders a
:class:`~spikeforge.observability.registry.MetricsRegistry` (or a JSON
snapshot of one) into the Prometheus text exposition format (version 0.0.4)
without pulling in any client library. The web-facing ``/metrics`` route in
``spikeforge-serve`` calls :func:`render` over the shared registry.

Mapping rules:

* a counter renders as ``<name>_total`` (the conventional suffix);
* a gauge renders verbatim;
* a timer renders as a histogram with cumulative ``<name>_bucket`` lines, a
  ``<name>_sum``, and a ``<name>_count``.

Dotted metric names such as ``serve.request_seconds`` are sanitised to
Prometheus-valid names (``serve_request_seconds``). An OpenTelemetry
exporter is deliberately *not* provided here: it would need the OTLP/
protobuf stack, which belongs in a satellite distribution rather than the
headless core, so PT-W6 ships the Prometheus MVP only.
"""

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

#: ``Content-Type`` a Prometheus scraper expects from an exposition endpoint.
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

#: Default histogram bucket upper bounds, in seconds.
DEFAULT_BUCKETS: Tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

_INVALID = re.compile(r"[^a-zA-Z0-9_:]")
_TOTAL = "_total"


def sanitize(name: str) -> str:
    """Return ``name`` as a Prometheus-valid metric name."""
    cleaned = _INVALID.sub("_", str(name))
    if not cleaned:
        return "_"
    if cleaned[0].isdigit():
        return "_" + cleaned
    return cleaned


def _number(value: Any) -> str:
    """Render ``value`` as a compact Prometheus number literal."""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return repr(number)


def _escape_help(text: Any) -> str:
    """Escape backslashes and newlines in a ``# HELP`` description."""
    return str(text).replace("\\", "\\\\").replace("\n", "\\n")


def _counter_name(base: str) -> str:
    """Return ``base`` with the conventional ``_total`` counter suffix."""
    return base if base.endswith(_TOTAL) else base + _TOTAL


def _help(metric: str, kind: str, include_help: bool) -> List[str]:
    """Return the optional ``# HELP`` line for ``metric``."""
    if not include_help:
        return []
    return [f"# HELP {metric} spikeforge {kind}: {_escape_help(metric)}"]


def _counter_lines(
    name: str, value: Any, include_help: bool
) -> List[str]:
    """Return the exposition lines for one counter."""
    metric = _counter_name(sanitize(name))
    return [
        *_help(metric, "counter", include_help),
        f"# TYPE {metric} counter",
        f"{metric} {_number(value)}",
    ]


def _gauge_lines(
    name: str, value: Any, include_help: bool
) -> List[str]:
    """Return the exposition lines for one gauge."""
    metric = sanitize(name)
    return [
        *_help(metric, "gauge", include_help),
        f"# TYPE {metric} gauge",
        f"{metric} {_number(value)}",
    ]


def _histogram_lines(
    name: str,
    summary: Mapping[str, Any],
    samples: Optional[Sequence[float]],
    buckets: Sequence[float],
    include_help: bool,
) -> List[str]:
    """Return the exposition lines for one timer as a histogram."""
    metric = sanitize(name)
    count = int(summary.get("count", 0) or 0)
    total = float(summary.get("total_seconds", 0.0) or 0.0)
    lines = [
        *_help(metric, "histogram (seconds)", include_help),
        f"# TYPE {metric} histogram",
    ]
    if samples is not None:
        for bound in buckets:
            cumulative = sum(1 for sample in samples if sample <= bound)
            lines.append(
                f'{metric}_bucket{{le="{_number(bound)}"}} {cumulative}'
            )
    lines.append(f'{metric}_bucket{{le="+Inf"}} {count}')
    lines.append(f"{metric}_sum {_number(total)}")
    lines.append(f"{metric}_count {count}")
    return lines


def _snapshot_of(source: Any) -> Dict[str, Any]:
    """Return the registry snapshot for a registry or a mapping."""
    if hasattr(source, "snapshot"):
        return source.snapshot()
    return dict(source)


def _raw_timers(source: Any) -> Dict[str, List[float]]:
    """Return raw timer samples for a registry, or an empty mapping."""
    if hasattr(source, "timers"):
        return source.timers()
    return {}


def render(
    source: Union[Any, Mapping[str, Any]],
    buckets: Sequence[float] = DEFAULT_BUCKETS,
    include_help: bool = True,
) -> str:
    """Render a registry (or its snapshot) as Prometheus exposition text.

    ``source`` may be a :class:`MetricsRegistry` or the mapping returned by
    its :meth:`snapshot`. Raw timer samples are used for histogram buckets
    when available; a snapshot mapping that only carries the per-timer
    summary still renders a valid histogram with a single ``+Inf`` bucket.
    """
    snapshot = _snapshot_of(source)
    raw = _raw_timers(source)
    lines: List[str] = []
    for name, value in sorted((snapshot.get("counters") or {}).items()):
        lines.extend(_counter_lines(name, value, include_help))
    for name, value in sorted((snapshot.get("gauges") or {}).items()):
        lines.extend(_gauge_lines(name, value, include_help))
    for name, summary in sorted((snapshot.get("timers") or {}).items()):
        lines.extend(
            _histogram_lines(
                name, summary, raw.get(name), buckets, include_help
            )
        )
    return "".join(f"{line}\n" for line in lines)

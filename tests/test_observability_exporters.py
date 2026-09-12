"""Issue #7 acceptance: the Prometheus exporter over the metrics registry."""

from spikeforge.observability import prometheus
from spikeforge.observability.registry import MetricsRegistry


def test_counter_gauge_and_histogram_render_valid_text() -> None:
    """A registry renders counters, gauges, and timers as exposition text."""
    registry = MetricsRegistry()
    registry.counter("serve.requests", 3)
    registry.gauge("serve.in_flight", 2)
    registry.observe("serve.request_seconds", 0.003)
    registry.observe("serve.request_seconds", 0.2)
    text = prometheus.render(registry)
    assert "# TYPE serve_requests_total counter" in text
    assert "serve_requests_total 3" in text
    assert "# TYPE serve_in_flight gauge" in text
    assert "serve_in_flight 2" in text
    assert "# TYPE serve_request_seconds histogram" in text
    assert 'serve_request_seconds_bucket{le="0.005"} 1' in text
    assert 'serve_request_seconds_bucket{le="0.25"} 2' in text
    assert 'serve_request_seconds_bucket{le="+Inf"} 2' in text
    assert "serve_request_seconds_count 2" in text
    assert "serve_request_seconds_sum 0.203" in text


def test_every_metric_has_a_help_and_type_line() -> None:
    """Each family carries one HELP and one TYPE header."""
    registry = MetricsRegistry()
    registry.counter("a")
    registry.gauge("b", 1.0)
    registry.observe("c", 0.01)
    text = prometheus.render(registry)
    assert text.count("# HELP ") == 3
    assert text.count("# TYPE ") == 3
    assert text.endswith("\n")


def test_metric_names_are_sanitised() -> None:
    """Dotted and hyphenated names become Prometheus-valid identifiers."""
    registry = MetricsRegistry()
    registry.counter("train.steps-total")
    registry.gauge("9lives", 1.0)
    text = prometheus.render(registry)
    assert "train_steps_total" in text
    assert "_9lives" in text
    assert prometheus.sanitize("serve.request.seconds") == (
        "serve_request_seconds"
    )


def test_empty_registry_renders_nothing() -> None:
    """An empty registry produces an empty exposition body."""
    assert prometheus.render(MetricsRegistry()) == ""


def test_snapshot_mapping_renders_without_raw_samples() -> None:
    """A summary-only snapshot still yields a valid single-bucket histogram."""
    snapshot = {
        "counters": {"x": 1.0},
        "gauges": {},
        "timers": {"t": {"count": 2, "total_seconds": 0.5}},
    }
    text = prometheus.render(snapshot)
    assert "x_total 1" in text
    assert 't_bucket{le="+Inf"} 2' in text
    assert "t_count 2" in text
    assert "t_sum 0.5" in text


def test_content_type_matches_a_prometheus_scraper() -> None:
    """The advertised content type is the 0.0.4 text exposition type."""
    assert prometheus.CONTENT_TYPE == (
        "text/plain; version=0.0.4; charset=utf-8"
    )

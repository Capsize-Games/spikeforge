"""Assemble the standalone hub page from its head, rows, and footer."""

from typing import Any, Dict, List

from build_hub_page_cells import (
    _notice,
    _row,
    _trained,
    _verified,
)
from build_hub_page_style import _HEAD, _STYLE, _WORDMARK, _header


def _page_head() -> str:
    """Return the document head: doctype, meta, styles, and open body."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>spikeforge model hub</title>
<meta name="description" content="Browsable, offline-generated listing of \
the spikeforge model-hub catalog.">
{_HEAD}
<style>{_STYLE}</style>
</head>
<body>
"""


def _page_intro(
    entries: List[Dict[str, Any]], trained_count: int, verified_count: int
) -> str:
    """Return the site header, title, and the two sub-lines."""
    return f"""{_header()}
<h1>{_WORDMARK} model hub</h1>
<p class="sub">{len(entries)} catalog entries -- {trained_count} with trained
weights, {verified_count} with a verified source and license.</p>
<p class="sub">This page is generated from the catalog, not a live view of
what is downloaded or cached.</p>
"""


def _page_table(rows: str) -> str:
    """Return the scrollable catalog table for ``rows``."""
    return f"""<div class="table-scroll">
<table>
<thead><tr><th>Name / id</th><th>Weights</th><th>Framework</th><th>Kind</th>
<th>Source</th><th>License</th><th>Training data</th>
<th>Notes</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</div>
"""


def _page_footer() -> str:
    """Return the footer and the closing document tags."""
    return """<footer class="hub-footer">
Browse locally with <code>pip install spikeforge-hub &amp;&amp;
spikeforge-hub list</code>, or from the
<a href="https://dash.spikeforge.net/">spikeforge</a>
dashboard's Hub panel.
<a href="/privacy/">Privacy policy</a>
</footer>
</body>
</html>
"""


def render(entries: List[Dict[str, Any]]) -> str:
    """Return the full standalone HTML page for ``entries``."""
    rows = "\n".join(_row(entry) for entry in entries)
    verified_count = sum(1 for entry in entries if _verified(entry))
    trained_count = sum(1 for entry in entries if _trained(entry))
    return (
        _page_head()
        + _page_intro(entries, trained_count, verified_count)
        + f"{_notice(trained_count)}\n"
        + _page_table(rows)
        + _page_footer()
    )

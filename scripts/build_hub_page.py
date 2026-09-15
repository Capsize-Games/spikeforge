#!/usr/bin/env python3
"""Render the model-hub catalog to a static, publicly browsable HTML page.

The hub's catalog (``spikeforge_hub/models.json``) is otherwise only visible
through a local CLI or a locally-run dashboard panel, so nothing about it is
indexable or shareable the way a Hugging Face model page is. This script
closes that gap with the cheapest possible fix: render the same catalog a
search engine or a casual visitor can open without installing anything.

Deliberately stdlib-only (no ``spikeforge_hub`` import) so it can run in the
landing-page deploy job without pulling in the torch-backed core distribution
just to build a static page. It re-implements the small "is this entry
verified?" rule from :mod:`spikeforge_hub.entry` rather than the full
``available()`` computation, which depends on the *viewer's* installed
extras and can't be known ahead of time for a pre-rendered page.

Usage::

    python scripts/build_hub_page.py [--out build/pages/hub/index.html]
"""

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "spikeforge_hub" / "models.json"
DEFAULT_OUT = REPO_ROOT / "build" / "pages" / "hub" / "index.html"
UNVERIFIED_CANDIDATE = "unverified-candidate"
#: The one catalog source whose entries carry trained weights.
TRAINED_SOURCE = "reference"

_STYLE = """
body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem
  auto; padding: 0 1rem; color: #1a1a1a; background: #fff; }
h1 { margin-bottom: 0.25rem; }
.sub { color: #555; margin-top: 0; }
.notice { background: #fff8e1; border: 1px solid #e0c46c; border-radius: 6px;
  padding: 0.75rem 1rem; margin: 1.25rem 0; }
table { border-collapse: collapse; width: 100%; margin-top: 1.5rem; }
th, td { text-align: left; padding: 0.5rem 0.6rem; border-bottom: 1px solid
  #ddd; vertical-align: top; font-size: 0.92rem; }
th { background: #f5f5f5; }
.badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px;
  font-size: 0.78rem; font-weight: 600; }
.badge-verified { background: #e3f6e5; color: #1b6d2f; }
.badge-unverified { background: #fde3e3; color: #a12222; }
.badge-trained { background: #e4edfb; color: #1d4a8f; }
.badge-untrained { background: #f0f0f0; color: #555; }
.score { font-variant-numeric: tabular-nums; font-weight: 600; }
/* The weights column is the one a visitor scans first; give it room so the
   score does not wrap one word per line. */
th:nth-child(2), td:nth-child(2) { min-width: 12rem; }
td:nth-child(2) { font-size: 0.86rem; line-height: 1.45; }
code { background: #f0f0f0; padding: 0.1rem 0.3rem; border-radius: 3px; }
footer { margin-top: 2rem; color: #666; font-size: 0.85rem; }
"""


def _load_entries() -> List[Dict[str, Any]]:
    """Return the raw catalog entries, or an empty list on a bad file."""
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    return entries if isinstance(entries, list) else []


def _verified(entry: Dict[str, Any]) -> bool:
    """Return False only for the explicit unverified-candidate marker."""
    return entry.get("license") != UNVERIFIED_CANDIDATE


def _trained(entry: Dict[str, Any]) -> bool:
    """Return True only for an entry that carries trained weights."""
    return entry.get("source") == TRAINED_SOURCE


def _weights_cell(entry: Dict[str, Any]) -> str:
    """Return the cell that says whether this entry is a usable model.

    This is the column a visitor actually came for. A catalog of untrained
    shapes and a catalog of trained models are different products, so the
    page states which each row is instead of leaving it to the notes.
    """
    if not _trained(entry):
        return (
            '<span class="badge badge-untrained">structure only</span>'
        )
    accuracy = entry.get("test_accuracy")
    samples = entry.get("test_samples")
    dataset = html.escape(str(entry.get("dataset", "")))
    score = (
        f'<span class="score">{accuracy}%</span> on {samples} held-out '
        f"{dataset} samples"
        if accuracy is not None
        else ""
    )
    return (
        '<span class="badge badge-trained">trained weights</span><br>'
        f"{score}"
    )


def _row(entry: Dict[str, Any]) -> str:
    """Return one ``<tr>`` for ``entry``, HTML-escaping every field."""
    esc = html.escape
    verified = _verified(entry)
    badge_class = "badge-verified" if verified else "badge-unverified"
    badge_text = "verified source" if verified else "unverified candidate"
    locator = entry.get("hf_repo") or entry.get("url") or entry.get(
        "topology"
    ) or ""
    return f"""<tr>
  <td><strong>{esc(str(entry.get("name", "")))}</strong><br>
    <code>{esc(str(entry.get("id", "")))}</code></td>
  <td>{_weights_cell(entry)}</td>
  <td>{esc(str(entry.get("framework", "")))}</td>
  <td>{esc(str(entry.get("kind", "")))}</td>
  <td>{esc(str(entry.get("source", "")))}<br>
    <code>{esc(str(locator))}</code></td>
  <td>{esc(str(entry.get("license", "")))}
    <span class="badge {badge_class}">{badge_text}</span></td>
  <td>{esc(str(entry.get("notes", "")))}</td>
</tr>"""


def _notice(trained_count: int) -> str:
    """Return the standing honesty notice, sized to what the catalog holds."""
    curation = (
        "https://github.com/Capsize-Games/spikeforge/blob/main/"
        "spikeforge_hub/CURATION.md"
    )
    if trained_count:
        body = (
            f"<strong>{trained_count} entries carry trained weights</strong> "
            "-- checkpoints this project trained itself, shipped with "
            "<code>spikeforge-hub</code>, each showing what it scores on the "
            "complete held-out test split. They are reference configurations "
            "with stock hyperparameters, <strong>not</strong> tuned attempts "
            "at state of the art, and every one names the command that "
            "reproduces it. The remaining entries are "
            "<code>bundled</code>: this project's own NIR graph presets -- "
            "structure, with freshly-initialised weights. No third-party "
            "weights are redistributed."
        )
    else:
        body = (
            "This catalog is <strong>metadata-only</strong>: entries marked "
            "<code>bundled</code> are this project's own NIR graph presets "
            "(structure, not third-party trained weights); no third-party "
            "weights are redistributed."
        )
    return (
        f'<div class="notice">\n{body}\nSee\n'
        f'<a href="{curation}">CURATION.md</a>\n'
        "for the verification policy, and open a pull request there to "
        "propose a\nreal, checked entry.\n</div>"
    )


def render(entries: List[Dict[str, Any]]) -> str:
    """Return the full standalone HTML page for ``entries``."""
    rows = "\n".join(_row(entry) for entry in entries)
    verified_count = sum(1 for entry in entries if _verified(entry))
    trained_count = sum(1 for entry in entries if _trained(entry))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>spikeforge model hub catalog</title>
<meta name="description" content="Browsable, offline-generated listing of \
the spikeforge model-hub catalog.">
<style>{_STYLE}</style>
</head>
<body>
<h1>spikeforge model hub</h1>
<p class="sub">{len(entries)} catalog entries -- {trained_count} with trained
weights, {verified_count} with a verified source and license. Generated from
<code>spikeforge_hub/models.json</code> by
<code>scripts/build_hub_page.py</code> -- not a live view of what is
downloaded or cached.</p>
{_notice(trained_count)}
<table>
<thead><tr><th>Name / id</th><th>Weights</th><th>Framework</th><th>Kind</th>
<th>Source</th><th>License</th><th>Notes</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
<footer>
Browse locally with <code>pip install spikeforge-hub &amp;&amp;
spikeforge-hub list</code>, or from the
<a href="https://github.com/Capsize-Games/spikeforge">spikeforge</a>
dashboard's Hub panel.
</footer>
</body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    """Render the catalog to ``--out`` (default ``build/pages/hub``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    entries = _load_entries()
    page = render(entries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"build_hub_page: wrote {len(entries)} entries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

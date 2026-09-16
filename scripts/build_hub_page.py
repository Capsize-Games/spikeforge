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

The page links ``../assets/capsize-landing.css`` for its color tokens (the
``:root`` custom properties only) rather than hard-coding a second copy of
the palette, so the two pages cannot silently drift apart on color the way
they had before. That path is fixed by ``.github/workflows/docs-deploy.yml``,
which assembles ``build/pages/assets/`` and ``build/pages/hub/`` as siblings;
change one and change the other. Everything else here -- layout, the header,
badges -- is this page's own, so an edit to the landing page's markup cannot
break this one.

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

#: The site logo, colored the same way the landing page and dashboard style
#: the wordmark: the "spike" half in the shared accent color.
_WORDMARK = '<span class="wordmark-spike">spike</span>forge'

_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;\
600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/capsize-landing.css">
"""

#: This page's own layout and components. Only ``var(--bg)`` etc. come from
#: the linked stylesheet; nothing here depends on its selectors.
_STYLE = """
* { box-sizing: border-box; }
body { font-family: "Inter", ui-sans-serif, system-ui, sans-serif;
  max-width: 1180px; margin: 0 auto; padding: 0 20px 3rem; color: var(--text);
  background: var(--bg); line-height: 1.55; }
a { color: var(--accent); }
code { font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas,
  monospace; background: var(--panel-alt); padding: 0.1rem 0.35rem;
  border-radius: 4px; font-size: 0.92em; }
.hub-header { display: flex; align-items: center; justify-content:
  space-between; flex-wrap: wrap; gap: 10px; padding-block: 18px;
  border-bottom: 1px solid var(--border); margin-bottom: 2rem; }
.hub-brand { display: inline-flex; align-items: center; gap: 8px;
  font-weight: 800; font-size: 17px; color: var(--text); text-decoration:
  none; }
.wordmark-spike { color: var(--accent); }
.hub-nav { display: flex; gap: 20px; font-size: 13px; flex-wrap: wrap; }
.hub-nav a { color: var(--muted); text-decoration: none; }
.hub-nav a:hover { color: var(--accent); }
h1 { margin: 0 0 0.25rem; font-size: 28px; letter-spacing: -0.01em; }
.sub { color: var(--muted); margin-top: 0; font-size: 14px; max-width: 68ch; }
.notice { background: var(--panel); border: 1px solid var(--border);
  border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0;
  padding: 0.9rem 1.1rem; margin: 1.5rem 0; font-size: 13.5px;
  color: var(--muted); }
.notice strong { color: var(--text); }
.notice a { color: var(--accent); }
/* Eight columns do not fit a phone; the table scrolls inside its own box
   rather than making the whole page scroll sideways. */
.table-scroll { overflow-x: auto; margin-top: 1.5rem; border: 1px solid
  var(--border); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { text-align: left; padding: 0.6rem 0.7rem; border-bottom: 1px solid
  var(--border); vertical-align: top; }
th { font-family: ui-monospace, monospace; font-size: 11px; letter-spacing:
  0.04em; text-transform: uppercase; color: var(--muted); background:
  var(--panel-head); }
tbody tr:hover { background: var(--panel-alt); }
tbody tr:last-child td { border-bottom: none; }
.badge { display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px;
  font-size: 0.75rem; font-weight: 600; white-space: nowrap; }
.badge-verified { background: rgba(63,185,80,0.16); color: #7ee2a8; }
.badge-unverified { background: rgba(248,81,73,0.16); color: #ff9d97; }
.badge-trained { background: rgba(165,184,255,0.18); color: var(--accent); }
.badge-untrained { background: rgba(160,177,204,0.16); color: var(--muted); }
.score { font-variant-numeric: tabular-nums; font-weight: 600; }
/* The requested credit line is often a full citation; keep it legible but
   subordinate to the licence id above it. */
.attribution { color: var(--muted); font-size: 0.8rem; line-height: 1.4;
  display: inline-block; margin-top: 0.15rem; }
.muted { color: var(--muted); }
/* The weights column is the one a visitor scans first; give it room so the
   score does not wrap one word per line. */
th:nth-child(2), td:nth-child(2) { min-width: 12rem; }
td:nth-child(2) { font-size: 0.86rem; line-height: 1.45; }
/* Training data: the citation needs width, or every row grows a tall
   one-word-per-line column. */
th:nth-child(7), td:nth-child(7) { min-width: 13rem; }
.hub-footer { margin-top: 2rem; color: var(--muted); font-size: 0.85rem; }
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


def _dataset_cell(entry: Dict[str, Any]) -> str:
    """Return the cell naming the training data's own terms and credit.

    The `License` column beside this one describes the *weights* -- this
    project's artifact. This one describes the data they were trained on,
    which is a different licence with a different holder. Showing only the
    first invites a reader to assume it covers both.
    """
    esc = html.escape
    dataset = entry.get("dataset")
    if not dataset:
        return '<span class="muted">not trained on a dataset</span>'
    license_id = str(entry.get("dataset_license") or "")
    unverified = license_id == UNVERIFIED_CANDIDATE
    badge = (
        f'<span class="badge badge-unverified">{esc(license_id)}</span>'
        if unverified
        else f"<code>{esc(license_id)}</code>"
    )
    attribution = esc(str(entry.get("dataset_attribution") or ""))
    return (
        f"<strong>{esc(str(dataset))}</strong><br>{badge}<br>"
        f'<span class="attribution">{attribution}</span>'
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
  <td>{_dataset_cell(entry)}</td>
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


def _header() -> str:
    """Return the site header: the wordmark and a link back to the site."""
    return f"""<header class="hub-header">
  <a class="hub-brand" href="https://spikeforge.net/">{_WORDMARK}</a>
  <nav class="hub-nav">
    <a href="https://spikeforge.net/">Home</a>
    <a href="https://docs.spikeforge.net/">Documentation</a>
    <a href="https://dash.spikeforge.net/">Dashboard</a>
    <a href="https://github.com/Capsize-Games/spikeforge">GitHub</a>
  </nav>
</header>"""


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
<title>spikeforge model hub</title>
<meta name="description" content="Browsable, offline-generated listing of \
the spikeforge model-hub catalog.">
{_HEAD}
<style>{_STYLE}</style>
</head>
<body>
{_header()}
<h1>{_WORDMARK} model hub</h1>
<p class="sub">{len(entries)} catalog entries -- {trained_count} with trained
weights, {verified_count} with a verified source and license.</p>
<p class="sub">This page is generated from the catalog, not a live view of
what is downloaded or cached.</p>
{_notice(trained_count)}
<div class="table-scroll">
<table>
<thead><tr><th>Name / id</th><th>Weights</th><th>Framework</th><th>Kind</th>
<th>Source</th><th>License</th><th>Training data</th>
<th>Notes</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</div>
<footer class="hub-footer">
Browse locally with <code>pip install spikeforge-hub &amp;&amp;
spikeforge-hub list</code>, or from the
<a href="https://dash.spikeforge.net/">spikeforge</a>
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

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
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from build_hub_page_cells import _verified as _verified
from build_hub_page_render import render as render

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "spikeforge_hub" / "models.json"
DEFAULT_OUT = REPO_ROOT / "build" / "pages" / "hub" / "index.html"


def _load_entries() -> List[Dict[str, Any]]:
    """Return the raw catalog entries, or an empty list on a bad file."""
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    return entries if isinstance(entries, list) else []


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

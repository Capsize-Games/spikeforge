#!/usr/bin/env python3
"""Fail when the published documentation site would ship a dead link.

``docs.spikeforge.net`` serves the GitHub wiki, which ``build_wiki.py``
regenerates from ``documentation/`` and ``plans/`` on every push to ``main``.
A wiki is a flat set of pages with no repository behind it, so a relative link
that is not another wiki page cannot resolve -- and GitHub does not 404 it,
it silently renders the wiki Home page instead, teleporting the reader back to
the documentation index with no explanation.

``check_docs_links.py`` cannot see any of this: it checks ``.md``-suffixed
links against the MkDocs tree in ``docs/``, which is built, strict-validated,
and published nowhere. This gate checks the artifact that is actually
published, by building the wiki into a temporary directory and auditing every
link the reader would be able to click.

Usage::

    python scripts/check_wiki_links.py

Exits non-zero, naming each page and target, when a link would be dead.
"""

import re
import sys
import tempfile
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_wiki import build  # noqa: E402

_LINK = re.compile(r"\]\(([^)\s]+)\)")
#: Wiki-relative targets that are always resolvable: an absolute URL, an
#: in-page anchor, or a mail link.
_ABSOLUTE = ("http://", "https://", "mailto:", "#")


def _dead_links(wiki: Path) -> List[str]:
    """Return every link on a generated page that the wiki cannot resolve."""
    pages = {page.stem for page in wiki.glob("*.md")}
    dead: List[str] = []
    for page in sorted(wiki.glob("*.md")):
        for target in _LINK.findall(page.read_text(encoding="utf-8")):
            if target.startswith(_ABSOLUTE):
                continue
            if target.split("#", 1)[0] in pages:
                continue
            dead.append(f"{page.stem} -> {target}")
    return dead


def main() -> int:
    """Build the wiki and report every link a reader could not follow."""
    with tempfile.TemporaryDirectory() as directory:
        wiki = Path(directory)
        build(wiki)
        pages = len(list(wiki.glob("*.md")))
        dead = _dead_links(wiki)
    for item in dead:
        print(f"dead link on the published docs site: {item}")
    if dead:
        print(
            f"\n{len(dead)} dead link(s) across {pages} generated wiki pages. "
            "Every relative link must resolve to another wiki page or be "
            "rewritten to an absolute repository URL by build_wiki.py -- a "
            "link build_wiki.py leaves alone is one whose target does not "
            "exist in this repository."
        )
        return 1
    print(f"published docs site OK: {pages} wiki pages, no dead links.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

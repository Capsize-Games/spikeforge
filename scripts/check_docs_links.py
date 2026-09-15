"""Fail when a documentation page links to a missing relative page.

The plan documents intentionally reference repository source files (for
example ``spikeforge/...py:14``), which are not documentation pages and
which MkDocs cannot distinguish from a broken page link. This checker narrows
the contract to what matters for a browsable site: every relative link that
targets another markdown page must resolve inside the generated docs tree.

Scope note: this guards the **MkDocs** tree in ``docs/``, which
``scripts/build_docs.sh`` generates and strict-mode validates. The site
readers actually land on -- ``docs.spikeforge.net``, which serves the GitHub
wiki -- is guarded by ``scripts/check_wiki_links.py``, which audits every
link on the generated wiki rather than only the ``.md`` ones. Keep both: this
one proves the MkDocs nav is sound, that one proves the published pages are.
"""

import re
import sys
from pathlib import Path
from typing import List, Sequence

_LINK = re.compile(r"\]\(([^)\s]+)\)")
_PAGE_SUFFIXES = (".md", ".markdown")


def _is_page(target: str) -> bool:
    """Return True for a relative link that targets a markdown page."""
    if target.startswith(("http://", "https://", "#", "/", "mailto:")):
        return False
    path = target.split("#", 1)[0]
    return path.endswith(_PAGE_SUFFIXES)


def _broken_links(docs: Path) -> List[str]:
    """Return a description of every markdown-page link that is missing."""
    broken: List[str] = []
    for page in sorted(docs.rglob("*.md")):
        for target in _LINK.findall(page.read_text(encoding="utf-8")):
            if not _is_page(target):
                continue
            resolved = (page.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists():
                broken.append(f"{page.relative_to(docs)} -> {target}")
    return broken


def main(argv: Sequence[str]) -> int:
    """Check a docs directory, printing broken links and returning an exit."""
    docs = Path(argv[1]) if len(argv) > 1 else Path("docs")
    broken = _broken_links(docs)
    for item in broken:
        print(f"broken documentation link: {item}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

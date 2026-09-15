#!/usr/bin/env python3
"""Build the Spikeforge GitHub Wiki working tree from repository docs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: The wiki is a flat set of pages with no repository behind it, so a relative
#: link that is not another wiki page cannot resolve there. GitHub silently
#: serves the wiki Home page for any unknown wiki path, so such a link does not
#: even 404 -- it teleports the reader back to the documentation home with no
#: explanation. Every non-page reference is therefore rewritten to an absolute
#: URL into the repository, which turns a dead link into a working one.
REPO_BLOB = "https://github.com/Capsize-Games/spikeforge/blob/main/"
REPO_TREE = "https://github.com/Capsize-Games/spikeforge/tree/main/"
REPO_RAW = "https://raw.githubusercontent.com/Capsize-Games/spikeforge/main/"
SOURCE_GROUPS = (
    ("Guides", ROOT / "documentation"),
    ("Plans", ROOT / "plans"),
)
#: Link syntax, covering the image form so a relative ``src`` is rewritten too.
_LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")
#: A ``file.py:16`` reference, which GitHub spells ``file.py#L16``.
_SOURCE_LINE = re.compile(r":(\d+)$")
#: Link targets that already address something outside the repository tree.
_ABSOLUTE = ("http://", "https://", "mailto:", "#", "/")
SPECIAL_PAGES = (
    ("Reference", ROOT / "COOKBOOK.md"),
    ("Reference", ROOT / "examples" / "README.md"),
    ("Reference", ROOT / "protocol" / "README.md"),
)


def page_name(path: Path) -> str:
    """Return the stable wiki page name for a Markdown source file."""
    if path == ROOT / "documentation" / "README.md":
        return "Home"
    if path == ROOT / "plans" / "index.md":
        return "Plans"
    if path == ROOT / "examples" / "README.md":
        return "Examples"
    if path == ROOT / "protocol" / "README.md":
        return "Protocol"
    if path == ROOT / "COOKBOOK.md":
        return "Cookbook"
    return path.stem.replace("_", "-")


def page_title(name: str) -> str:
    """Turn a wiki page slug into a readable navigation label."""
    return name.replace("-", " ").title()


def source_pages() -> list[tuple[str, Path]]:
    """Collect every Markdown document exposed by the landing page."""
    pages = list(SPECIAL_PAGES)
    for group, directory in SOURCE_GROUPS:
        pages.extend((group, path) for path in sorted(directory.glob("*.md")))
    return pages


def markdown_link_map(pages: list[tuple[str, Path]]) -> dict[str, str]:
    """Map source paths to wiki links for cross-document rewrites."""
    links: dict[str, str] = {}
    for _, path in pages:
        name = page_name(path)
        relative = path.relative_to(ROOT).as_posix()
        links[relative] = name
        links[path.name] = name
    return links


def repository_path(target: str, source: Path) -> str | None:
    """Resolve a relative link to its repository path, or None if unknown.

    The documents address repository files two ways -- relative to the
    document (``../examples/``) and relative to the repository root
    (``plans/arch-0001-target-topology.md``) -- so both are tried.
    """
    for base in (source.parent, ROOT):
        try:
            resolved = (base / target).resolve()
            relative = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            continue
        if resolved.exists():
            return relative
    return None


def absolute_url(target: str, source: Path, is_image: bool) -> str | None:
    """Return the absolute repository URL for a non-page relative link.

    Source references carry a ``:LINE`` suffix (``spec.py:16``), which is a
    path on GitHub only when it becomes the ``#L16`` fragment.
    """
    path, _, fragment = target.partition("#")
    line = _SOURCE_LINE.search(path)
    if line:
        path = path[: line.start()]
    relative = repository_path(path, source)
    if relative is None:
        return None
    if is_image:
        return f"{REPO_RAW}{relative}"
    if (ROOT / relative).is_dir():
        return f"{REPO_TREE}{relative}/"
    if line:
        return f"{REPO_BLOB}{relative}#L{line.group(1)}"
    return f"{REPO_BLOB}{relative}{'#' + fragment if fragment else ''}"


def rewrite_links(text: str, source: Path, links: dict[str, str]) -> str:
    """Rewrite every local link to a wiki page or an absolute repository URL.

    A link to another generated page becomes that page's wiki name; every
    other relative link -- source files, directories, images, repository
    documents the wiki does not carry -- becomes an absolute GitHub URL.
    """

    def replace(match: re.Match[str]) -> str:
        bang, label, target = match.groups()
        if target.startswith(_ABSOLUTE):
            return match.group(0)
        path, _, fragment = target.partition("#")
        path = path.split("?", 1)[0]
        if not bang and path.endswith(".md"):
            relative = repository_path(path, source)
            page = links.get(relative or "") or links.get(Path(path).name)
            if page:
                return f"[{label}]({page}{'#' + fragment if fragment else ''})"
        url = absolute_url(target, source, bool(bang))
        return f"{bang}[{label}]({url})" if url else match.group(0)

    return _LINK.sub(replace, text)


def write_header(destination: Path) -> None:
    """Write the shared top-level product navigation for every wiki page."""
    navigation = (
        "[⚡ spikeforge](https://spikeforge.net/) · "
        "[Features](https://spikeforge.net/#features) · "
        "[Status](https://spikeforge.net/#status) · "
        "[Documentation](https://docs.spikeforge.net/) · "
        "[Dashboard](https://dash.spikeforge.net/) · "
        "[GitHub](https://github.com/Capsize-Games/spikeforge)\n"
    )
    (destination / "_Header.md").write_text(navigation, encoding="utf-8")


def write_sidebar(
    destination: Path,
    pages: list[tuple[str, Path]],
) -> None:
    """Write grouped navigation for the generated wiki."""
    lines = ["## Spikeforge documentation", ""]
    for group in ("Guides", "Plans", "Reference"):
        entries = [path for item, path in pages if item == group]
        if not entries:
            continue
        lines.extend((f"### {group}", ""))
        for path in entries:
            name = page_name(path)
            lines.append(f"- [{page_title(name)}]({name})")
        lines.append("")
    sidebar = destination / "_Sidebar.md"
    sidebar.write_text("\n".join(lines), encoding="utf-8")


def build(destination: Path) -> None:
    """Create a complete wiki working tree at destination."""
    destination.mkdir(parents=True, exist_ok=True)
    for old_page in destination.glob("*.md"):
        old_page.unlink()
    pages = source_pages()
    links = markdown_link_map(pages)
    for _, source in pages:
        text = rewrite_links(source.read_text(encoding="utf-8"), source, links)
        target = destination / f"{page_name(source)}.md"
        target.write_text(text, encoding="utf-8")
    write_header(destination)
    write_sidebar(destination, pages)


def main() -> None:
    """Parse the output directory and build wiki pages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.destination.resolve())


if __name__ == "__main__":
    main()

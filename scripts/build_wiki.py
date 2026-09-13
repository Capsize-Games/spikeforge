#!/usr/bin/env python3
"""Build the Spikeforge GitHub Wiki working tree from repository docs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_GROUPS = (
    ("Guides", ROOT / "documentation"),
    ("Plans", ROOT / "plans"),
)
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


def rewrite_links(text: str, source: Path, links: dict[str, str]) -> str:
    """Rewrite local Markdown document links to their wiki page names."""
    pattern = re.compile(r"\]\((?!https?://|#)([^)]+\.md)(#[^)]+)?\)")

    def replace(match: re.Match[str]) -> str:
        target = match.group(1).split("?", 1)[0]
        resolved = (source.parent / target).resolve()
        try:
            relative = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            return match.group(0)
        page = links.get(relative) or links.get(Path(target).name)
        return f"]({page}{match.group(2) or ''})" if page else match.group(0)

    return pattern.sub(replace, text)


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
    (destination / "_Sidebar.md").write_text("\n".join(lines), encoding="utf-8")


def build(destination: Path) -> None:
    """Create a complete wiki working tree at destination."""
    destination.mkdir(parents=True, exist_ok=True)
    for old_page in destination.glob("*.md"):
        old_page.unlink()
    pages = source_pages()
    links = markdown_link_map(pages)
    for _, source in pages:
        text = rewrite_links(source.read_text(encoding="utf-8"), source, links)
        (destination / f"{page_name(source)}.md").write_text(text, encoding="utf-8")
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

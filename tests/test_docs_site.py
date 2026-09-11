"""Docs site configuration, build script, and documentation link checking."""

from pathlib import Path

from scripts.check_docs_links import main

_ROOT = Path(__file__).resolve().parent.parent


def test_mkdocs_config_covers_every_plan() -> None:
    """The Material config names every plan document in its nav."""
    text = (_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "docs_dir: docs" in text
    assert "name: material" in text
    for plan in sorted((_ROOT / "plans").glob("*.md")):
        assert plan.name in text


def test_build_script_runs_a_strict_check() -> None:
    """The build script is present and wires the strict link check."""
    script = _ROOT / "scripts" / "build_docs.sh"
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "mkdocs build --strict" in text
    assert "--check" in text
    assert "check_docs_links.py" in text


def test_link_checker_passes_when_pages_resolve(tmp_path: Path) -> None:
    """A resolvable page link is accepted."""
    (tmp_path / "index.md").write_text("[ok](guide.md)\n", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Guide\n", encoding="utf-8")
    assert main(["check", str(tmp_path)]) == 0


def test_link_checker_fails_on_a_broken_page_link(tmp_path: Path) -> None:
    """A missing documentation page fails the check."""
    (tmp_path / "index.md").write_text("[bad](missing.md)\n", encoding="utf-8")
    assert main(["check", str(tmp_path)]) == 1


def test_link_checker_ignores_source_references(tmp_path: Path) -> None:
    """Repository source references are not documentation page links."""
    page = tmp_path / "index.md"
    page.write_text("[src](spikeforge/a.py:14)\n", encoding="utf-8")
    assert main(["check", str(tmp_path)]) == 0

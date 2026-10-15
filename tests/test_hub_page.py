"""Static model-hub page generator (scripts/build_hub_page.py)."""

from pathlib import Path

from scripts.build_hub_page import _load_entries, _verified, main, render
from scripts.build_hub_page_cells import _download_links


def test_loads_the_real_catalog() -> None:
    """The generator reads the same catalog the hub package ships."""
    entries = _load_entries()
    assert entries
    assert all("id" in entry for entry in entries)


def test_verified_flags_the_unverified_candidate_marker() -> None:
    """Only the explicit marker is reported as an unverified candidate."""
    assert _verified({"license": "BSD-3-Clause"}) is True
    assert _verified({"license": "unverified-candidate"}) is False


def test_render_escapes_untrusted_fields() -> None:
    """A malicious-looking field never lands as raw HTML in the page."""
    page = render(
        [
            {
                "id": "x/y",
                "name": "<script>evil()</script>",
                "framework": "nir",
                "kind": "nir_graph",
                "source": "bundled",
                "license": "BSD-3-Clause",
                "notes": "safe",
            }
        ]
    )
    assert "<script>evil()</script>" not in page
    assert "&lt;script&gt;" in page


def test_render_includes_every_entry_once() -> None:
    """Each entry id appears exactly once in the rendered table."""
    entries = _load_entries()
    page = render(entries)
    for entry in entries:
        assert page.count(str(entry["id"])) == 1


def test_trained_rows_link_their_bytes() -> None:
    """Every trained row anchors its weights file for download."""
    entries = _load_entries()
    trained = [e for e in entries if e.get("source") == "reference"]
    assert trained
    for entry in trained:
        links = _download_links(entry)
        assert "download weights" in links, entry["id"]
        assert entry["weights"] in links, entry["id"]


def test_bundled_rows_link_nothing() -> None:
    """Structure-only rows offer no bytes to download."""
    entries = _load_entries()
    bundled = [e for e in entries if e.get("source") != "reference"]
    assert bundled
    for entry in bundled:
        assert _download_links(entry) == "", entry["id"]


def test_main_writes_the_output_file(tmp_path: Path) -> None:
    """``main`` writes a real HTML file to the requested path."""
    out = tmp_path / "hub" / "index.html"
    exit_code = main(["--out", str(out)])
    assert exit_code == 0
    assert out.exists()
    assert "<table>" in out.read_text(encoding="utf-8")

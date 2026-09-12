"""Save/load/list/delete for pipeline graph files."""

from pathlib import Path

import pytest

from spikeforge_serve import pipeline_store

_GRAPH = {
    "version": 1,
    "name": "digits-then-risk",
    "nodes": [
        {"id": "n1", "checkpoint": "digits", "position": {"x": 0, "y": 0}},
    ],
    "edges": [],
}


@pytest.fixture(autouse=True)
def _pipelines_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect pipeline reads/writes into a per-test directory."""
    monkeypatch.setattr(pipeline_store, "PIPELINES_DIR", str(tmp_path))


def test_save_then_load_round_trips() -> None:
    """A saved pipeline loads back byte-for-byte the same structure."""
    pipeline_store.save("digits-then-risk", _GRAPH)
    assert pipeline_store.load("digits-then-risk") == _GRAPH


def test_load_missing_pipeline_raises_a_named_error() -> None:
    """A name with no saved file is a clear FileNotFoundError, not a crash."""
    with pytest.raises(FileNotFoundError, match="nope"):
        pipeline_store.load("nope")


def test_list_pipelines_is_empty_then_reflects_saves() -> None:
    """An empty directory lists nothing; a save shows up with real counts."""
    assert pipeline_store.list_pipelines() == []
    pipeline_store.save("digits-then-risk", _GRAPH)
    listed = pipeline_store.list_pipelines()
    assert len(listed) == 1
    assert listed[0]["name"] == "digits-then-risk"
    assert listed[0]["node_count"] == 1
    assert listed[0]["edge_count"] == 0


def test_delete_removes_a_saved_pipeline() -> None:
    """Deleting an existing pipeline reports True and it's gone after."""
    pipeline_store.save("digits-then-risk", _GRAPH)
    assert pipeline_store.delete("digits-then-risk") is True
    assert pipeline_store.list_pipelines() == []


def test_delete_a_missing_pipeline_reports_false() -> None:
    """Deleting a name that was never saved is a no-op, not an error."""
    assert pipeline_store.delete("nope") is False


def test_name_is_sanitised_for_the_filesystem() -> None:
    """Unsafe characters in the name don't produce path traversal or errors."""
    pipeline_store.save("weird name/with slash", _GRAPH)
    listed = pipeline_store.list_pipelines()
    assert listed[0]["name"] == "weird_name_with_slash"


def test_a_stray_non_json_file_is_ignored(tmp_path: Path) -> None:
    """A non-.json file in the pipelines dir doesn't break listing."""
    (tmp_path / "notes.txt").write_text("hello")
    pipeline_store.save("digits-then-risk", _GRAPH)
    listed = pipeline_store.list_pipelines()
    assert len(listed) == 1

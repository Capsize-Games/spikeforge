"""Bundle traversal: hostile member names never reach the filesystem.

``plans/hub_accounts_plan.md`` §10 records that the serving bundle
reader is safe against path traversal because it reads named zip members
and never calls ``extractall`` -- "keep it that way, and add a test that
asserts it". These tests feed the read path a malicious zip (a ``../``
member and an absolute-path member) and assert no file materializes
outside the intended destination, plus a static assertion that the
serving package performs no archive extraction at all.
"""

import ast
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest
from _bundle_helpers import _written

import spikeforge.serving.bundle_read as bundle_read
from spikeforge.network import model_store
from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.errors import BundleIntegrityError

#: A member that escapes the destination when extracted naively.
_RELATIVE = "../escape-rel-spkf.txt"
#: An absolute member lands outside any destination when extracted.
_ABSOLUTE_NAME = "escape-abs-spkf.txt"


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint reads and writes into a per-test directory.

    Also runs the test with the scratch directory as the cwd, so any
    extraction relative to the cwd lands inside the snapshotted tree
    instead of the repository.
    """
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def _snapshot(root: Any) -> Set[Path]:
    """Return every file under ``root`` as a relative path."""
    base = Path(str(root))
    return {
        path.relative_to(base)
        for path in base.rglob("*")
        if path.is_file()
    }


def _rewrite_with(
    path: str, extra: Dict[str, bytes], listed: bool
) -> None:
    """Add ``extra`` members to the bundle at ``path``.

    When ``listed`` the members are checksummed into SHA256SUMS (a
    re-signed hostile bundle); otherwise they ride along unlisted.
    """
    with zipfile.ZipFile(path) as archive:
        entries = {
            name: archive.read(name) for name in archive.namelist()
        }
    entries.update(extra)
    if listed:
        payloads = {
            name: data
            for name, data in entries.items()
            if name != bm.SUMS_NAME
        }
        entries[bm.SUMS_NAME] = bm.sums_text(payloads).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


def _hostile(tmp_path: Any) -> Dict[str, bytes]:
    """Return traversal members: a parent escape and an absolute path."""
    absolute = str(Path(str(tmp_path)).parent / _ABSOLUTE_NAME)
    return {_RELATIVE: b"escape", absolute: b"escape"}


def _assert_confined(tmp_path: Any, before: Set[Path]) -> None:
    """Assert loading wrote nothing inside ``tmp_path`` or outside it."""
    assert _snapshot(tmp_path) == before
    parent = Path(str(tmp_path)).parent
    assert not (parent / "escape-rel-spkf.txt").exists()
    assert not (parent / _ABSOLUTE_NAME).exists()


def test_unlisted_traversal_members_are_refused(tmp_path: Any) -> None:
    """Hostile members riding along unlisted are an integrity error."""
    out = _written(tmp_path)
    _rewrite_with(out, _hostile(tmp_path), listed=False)
    before = _snapshot(tmp_path)
    with pytest.raises(BundleIntegrityError):
        DeploymentBundle.load(out)
    _assert_confined(tmp_path, before)


def test_listed_traversal_members_stay_confined(tmp_path: Any) -> None:
    """Even a re-signed hostile bundle writes nothing outside itself."""
    out = _written(tmp_path)
    _rewrite_with(out, _hostile(tmp_path), listed=True)
    before = _snapshot(tmp_path)
    DeploymentBundle.load(out)
    _assert_confined(tmp_path, before)


def _extraction_calls(module: Path) -> List[str]:
    """Return every ``extract``/``extractall`` call site in ``module``."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"extract", "extractall"}
        ):
            calls.append(f"{module.name}:{node.lineno}")
    return calls


def test_the_serving_reader_extracts_no_archive_member() -> None:
    """The bundle read path reads named members; it never extracts."""
    package = Path(str(bundle_read.__file__)).resolve().parent
    calls = []
    for module in sorted(package.glob("*.py")):
        calls.extend(_extraction_calls(module))
    assert calls == []

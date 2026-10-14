"""Torch-free ingestion through a subprocess without torch (issue #161).

The suite runs with torch installed, so this test blocks the ``torch``
import in a child interpreter and proves ``spikeforge_io.adapters``
imports and CSV/JSON/in-memory/NPY ingestion works there.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

# Probe script run with torch blocked; prints ``TORCHFREE-OK`` on success.
_CHILD_SCRIPT = textwrap.dedent(
    """\
    import csv
    import json
    import sys
    import tempfile
    from pathlib import Path

    sys.modules["torch"] = None  # simulate torch not installed

    import numpy as np

    import spikeforge_io as sio
    import spikeforge_io.adapters as adapters
    from spikeforge_io.errors import AdapterDependencyError

    tmp = Path(tempfile.mkdtemp())
    rows = [[1.0, 2.0], [3.0, 4.0]]

    csv_path = tmp / "r.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["c0", "c1"])
        writer.writerows(rows)
    json_path = tmp / "r.json"
    json_path.write_text(json.dumps(rows), encoding="utf-8")
    npy_path = tmp / "r.npy"
    np.save(str(npy_path), np.asarray(rows, dtype=np.float32))

    expected = np.asarray(rows, dtype=np.float32)
    views = [
        adapters.CsvAdapter(str(csv_path)).stream_numpy(),
        adapters.JsonAdapter(str(json_path)).stream_numpy(),
        sio.InMemoryAdapter(rows).stream_numpy(),
        adapters.NpyAdapter(str(npy_path)).stream_numpy(),
    ]
    for view in views:
        assert view.dtype == np.float32, view.dtype
        assert view.shape == (2, 2), view.shape
        assert bool(np.array_equal(view, expected)), view
    assert len(adapters.CsvAdapter(str(csv_path))) == 2
    assert "torch" not in sys.modules or sys.modules["torch"] is None

    try:
        adapters.InMemoryAdapter(rows).stream()
    except AdapterDependencyError as error:
        assert error.dependency == "torch", error.dependency
    else:
        raise AssertionError("torch stream did not report torch absent")

    try:
        sio.WindowSpec
    except ImportError:
        pass
    else:
        raise AssertionError("windowing resolved without torch")

    print("TORCHFREE-OK")
    """
)


def test_ingestion_without_torch() -> None:
    """Adapters import and ingest in a child without torch installed."""
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "TORCHFREE-OK" in proc.stdout

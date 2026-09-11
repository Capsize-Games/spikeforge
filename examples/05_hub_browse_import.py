"""Browse, inspect, and import a model from the bundled hub catalog.

The curated catalog renders fully offline. Live Hugging Face search is
opt-in behind the ``hub`` extra; this script reports that availability
honestly instead of failing when the extra is absent.

Run from the repository root::

    venv/bin/python examples/05_hub_browse_import.py

Promoting the bundled ``nir/fc_legacy`` graph writes a checkpoint into
``MODEL_DIR`` (the gitignored ``build/models``). The equivalent shell
commands are ``snn-hub list``, ``snn-hub inspect nir/fc_legacy`` and
``snn-hub import nir/fc_legacy``.
"""

from typing import Any, Dict, List

from snn_interpreter.hub import available, catalog, get, import_model, inspect

#: Bundled NIR entry exercised by this example.
ENTRY_ID = "nir/fc_legacy"


def browse() -> List[Dict[str, Any]]:
    """Print a short catalog overview and return the entry cards."""
    cards = catalog()
    frameworks = sorted({str(card["framework"]) for card in cards})
    print("catalog entries:", len(cards))
    print("frameworks:", frameworks)
    print("live Hugging Face search available:", available())
    if not available():
        print("  (live search needs the `hub` extra: huggingface_hub)")
    return cards


def promote() -> None:
    """Inspect and import the bundled entry, printing the outcome."""
    entry = get(ENTRY_ID)
    print("entry:", entry.id if entry is not None else None)
    report = inspect(ENTRY_ID)
    print("inspect kind:", report["kind"])
    result = import_model(ENTRY_ID)
    print("verdict:", result["verdict"]["verdict"])
    print("promoted:", result["promoted"])
    print("destination:", result["destination"])


def main() -> int:
    """Run the catalog browse and import journey."""
    browse()
    promote()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

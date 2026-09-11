"""Bundled per-target declared cost tables.

Each ``costs/<target>.json`` travels with the target's capability matrix: it
declares energy per SOP/MAC/AC and latency per timestep plus its source and a
``measured: false`` flag. A target with no bundled table is simply absent, so
the accounting layer reports ``basis: "unavailable"`` instead of guessing.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from spikeforge_targets.energy.cost_table import CostTable
from spikeforge_targets.energy.errors import EnergyCostError

#: Directory of the bundled declared cost tables, shipped inside the package.
COSTS_DIR = Path(__file__).with_name("costs")


@lru_cache(maxsize=None)
def _tables() -> Dict[str, CostTable]:
    """Load every bundled cost table once, keyed by target name."""
    tables: Dict[str, CostTable] = {}
    for path in sorted(COSTS_DIR.glob("*.json")):
        tables[path.stem] = CostTable.from_dict(path.stem, _read(path))
    return tables


def _read(path: Path) -> Dict[str, object]:
    """Return the parsed JSON object at ``path`` or raise a typed error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnergyCostError(path.stem, f"cannot read: {exc}") from exc
    if not isinstance(data, dict):
        raise EnergyCostError(path.stem, "cost table root must be an object")
    return data


def load(target: str) -> Optional[CostTable]:
    """Return the declared cost table for ``target``, or ``None``."""
    return _tables().get(target)


def declared(target: str) -> bool:
    """Return True when ``target`` has a bundled declared cost table."""
    return target in _tables()


def names() -> List[str]:
    """Return the targets with a bundled declared cost table, sorted."""
    return sorted(_tables())

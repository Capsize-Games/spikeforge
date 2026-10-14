"""Case envelopes and wrapped-JSON fixtures for the encoding goldens.

A case is a plain dict: its name, a description, generator provenance,
the frozen spec plus digest, the procedure in words, and the input and
output tensors as shapes with row-major values. Spike tensors store
``0``/``1`` integers; anything else stores floats. Values wrap eight
per line so a spike raster stays readable in a diff.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def payload(tensor: torch.Tensor) -> Dict[str, Any]:
    """Return ``tensor`` as a shape plus row-major values dict."""
    flat = [float(value) for value in tensor.flatten().tolist()]
    values: List[Any] = flat
    if all(value in (0.0, 1.0) for value in flat):
        values = [int(value) for value in flat]
    return {"shape": list(tensor.shape), "values": values}


def envelope(
    name: str, description: str, provenance: Dict[str, str]
) -> Dict[str, Any]:
    """Return the case envelope with generator provenance filled in."""
    return {
        "name": name,
        "description": description,
        "generator": dict(provenance),
    }


def _wrapped(values: List[Any], pad: str, width: int = 8) -> str:
    """Format ``values`` as a JSON array wrapped ``width`` per line."""
    rows = [
        ", ".join(json.dumps(value) for value in values[i : i + width])
        for i in range(0, len(values), width)
    ]
    lines = ",\n".join(pad + row for row in rows)
    return "[\n" + lines + "\n    ]"


def _tensor_block(label: str, data: Dict[str, Any]) -> str:
    """Format an input/output payload with a wrapped values array."""
    shape = json.dumps(data["shape"])
    values = _wrapped(data["values"], "      ")
    lines = [
        f'  "{label}": {{',
        f'    "shape": {shape},',
        f'    "values": {values}',
        "  }",
    ]
    return "\n".join(lines)


def case_text(case: Dict[str, Any]) -> str:
    """Render ``case`` as JSON with wrapped tensor value arrays."""
    head = {
        key: value
        for key, value in case.items()
        if key not in ("input", "output")
    }
    text = json.dumps(head, indent=2)
    body = text[: text.rindex("}")].rstrip()
    blocks = "\n".join(
        [
            _tensor_block("input", case["input"]) + ",",
            _tensor_block("output", case["output"]),
        ]
    )
    return body + ",\n" + blocks + "\n}\n"


def write_cases(cases: List[Dict[str, Any]], out: Path) -> List[Path]:
    """Write ``cases`` as ``<name>.json`` under ``out``; return paths."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for case in cases:
        path = out / f"{case['name']}.json"
        text = case_text(case)
        json.loads(text)
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths

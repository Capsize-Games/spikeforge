"""Render training records as a Markdown summary."""

from typing import Any, Dict, List


def _markdown_row(record: Dict[str, Any]) -> str:
    """Return one Markdown table row for a trained config's result."""
    return (
        f"| `{record['name']}` | {record['dataset']} | "
        f"`{record['topology']}` | **{record['test_accuracy']}%** | "
        f"{record['epochs']} | {record['num_steps']} | "
        f"{record['train_seconds']} s |"
    )


def _markdown_footer(hardware: Dict[str, Any]) -> str:
    """Return the hardware footer line for the results table."""
    return (
        f"Hardware: {hardware['processor']}, "
        f"{hardware['torch_threads']} torch threads, CPU only. "
        f"Python {hardware['python']}, torch {hardware['torch']}."
    )


def _markdown(records: List[Dict[str, Any]], hardware: Dict[str, Any]) -> str:
    """Render the results as the table the documentation page carries."""
    lines = [
        "| Configuration | Dataset | Topology | Test accuracy | Epochs | "
        "Steps | Train time |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(_markdown_row(record) for record in records)
    lines.extend(["", _markdown_footer(hardware)])
    return "\n".join(lines)


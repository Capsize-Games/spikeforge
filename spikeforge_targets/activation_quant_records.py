"""Per-tensor range and error records of a simulated quantization."""

from typing import Any, Dict, Tuple

import torch


def tensor_range(tensor: torch.Tensor) -> Tuple[float, float]:
    """Return the (min, max) of ``tensor`` as floats, or zeros when empty."""
    if not tensor.numel():
        return (0.0, 0.0)
    return (float(tensor.min()), float(tensor.max()))


class RangeRecords:
    """Fold before/after tensors into one JSON-able record per key.

    Each record keeps the running value range before and after snapping and
    the largest and mean absolute error the grid introduced, so a report can
    show where a scheme's resolution was lost. Private counters never leave
    :meth:`layers`.
    """

    def __init__(self) -> None:
        """Start with no records."""
        self._records: Dict[str, Dict[str, Any]] = {}

    def __len__(self) -> int:
        """Return how many keys have been folded."""
        return len(self._records)

    def _new(self, key: str, kind: str) -> Dict[str, Any]:
        """Return the empty record for ``key``."""
        record = {
            "name": key,
            "kind": kind,
            "before": [float("inf"), float("-inf")],
            "after": [float("inf"), float("-inf")],
            "max_abs": 0.0,
            "_abs_sum": 0.0,
            "_count": 0,
        }
        self._records[key] = record
        return record

    def fold(
        self,
        key: str,
        kind: str,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> None:
        """Fold one tensor's ranges and error into the record for ``key``."""
        record = self._records.get(key)
        if record is None:
            record = self._new(key, kind)
        for field, tensor in (("before", before), ("after", after)):
            low, high = tensor_range(tensor)
            record[field][0] = min(record[field][0], low)
            record[field][1] = max(record[field][1], high)
        difference = (after - before).abs()
        if difference.numel():
            record["max_abs"] = max(
                record["max_abs"], float(difference.max())
            )
            record["_abs_sum"] += float(difference.sum())
            record["_count"] += int(difference.numel())

    @staticmethod
    def _layer(record: Dict[str, Any]) -> Dict[str, Any]:
        """Return the JSON-able form of one record."""
        count = int(record["_count"])
        return {
            "name": record["name"],
            "kind": record["kind"],
            "before": [float(record["before"][0]), float(record["before"][1])],
            "after": [float(record["after"][0]), float(record["after"][1])],
            "max_abs": float(record["max_abs"]),
            "mean_abs": (
                float(record["_abs_sum"]) / count if count else 0.0
            ),
        }

    def layers(self) -> Tuple[Dict[str, Any], ...]:
        """Return every record, sorted by key, without private counters."""
        return tuple(
            self._layer(record)
            for record in sorted(
                self._records.values(), key=lambda item: item["name"]
            )
        )

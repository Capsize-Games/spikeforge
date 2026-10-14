"""Adapter reading a ``.csv`` file of numeric rows into a stream.

Split out of the ``adapters`` package root so each adapter lives in its
own file.
"""

import csv
from typing import List, Optional, Sequence

import torch

from spikeforge_io.adapters.base import StreamAdapter, as_stream
from spikeforge_io.errors import AdapterFormatError, AdapterShapeError


def _read_rows(path: str, delimiter: str) -> List[List[str]]:
    """Read a CSV file into raw rows, or raise a typed format error."""
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.reader(handle, delimiter=delimiter))
    except OSError as error:
        raise AdapterFormatError(path, f"cannot read: {error}") from None


def _parse_csv_row(path: str, index: int, row: Sequence[str]) -> List[float]:
    """Parse one CSV row into floats, naming the first bad field."""
    values: List[float] = []
    for column, cell in enumerate(row):
        text = cell.strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError:
            raise AdapterFormatError(
                path,
                f"row {index} column {column} is not numeric: {cell!r}",
                position=index,
            ) from None
    if not values:
        raise AdapterFormatError(
            path, f"row {index} has no numeric fields", position=index
        )
    return values


def _uniform_width(source: str, parsed: List[List[float]]) -> int:
    """Return the shared row width, or raise on a ragged row."""
    width = len(parsed[0])
    for index, row in enumerate(parsed):
        if len(row) != width:
            raise AdapterFormatError(
                source,
                f"row {index} has {len(row)} fields, expected {width}",
                position=index,
            )
    return width


class CsvAdapter(StreamAdapter):
    """Read a ``.csv`` file of numeric rows into an ``[N, D]`` stream."""

    def __init__(
        self,
        path: str,
        *,
        delimiter: str = ",",
        has_header: bool = True,
        channels: Optional[int] = None,
    ) -> None:
        """Record the ``path``, delimiter, header flag, and expected width."""
        self.source = path
        self._delimiter = delimiter
        self._has_header = has_header
        self._channels = channels

    def _rows(self) -> List[List[float]]:
        """Return the CSV's parsed numeric rows at a uniform width."""
        rows = _read_rows(self.source, self._delimiter)
        if self._has_header and rows:
            rows = rows[1:]
        if not rows:
            raise AdapterShapeError(self.source, "CSV has no data rows")
        parsed = [
            _parse_csv_row(self.source, index, row)
            for index, row in enumerate(rows)
        ]
        _uniform_width(self.source, parsed)
        return parsed

    def stream(self) -> torch.Tensor:
        """Return the CSV's numeric rows as a ``[N, D]`` tensor."""
        parsed = self._rows()
        width = len(parsed[0])
        if self._channels is not None and width != self._channels:
            raise AdapterShapeError(
                self.source,
                f"expected {self._channels} channels, got {width}",
            )
        return as_stream(parsed, self.source)

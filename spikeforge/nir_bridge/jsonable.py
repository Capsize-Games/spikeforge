"""Coerce NIR node fields into plain JSON-serialisable values."""

import dataclasses
from typing import Any, Dict, Mapping

#: Derived type/shape fields omitted from a summary's node parameters.
RESERVED_FIELDS = ("input_type", "output_type", "metadata")


def to_jsonable(value: Any) -> Any:
    """Return ``value`` with arrays converted to nested lists.

    Numpy arrays and torch tensors both expose ``tolist``, so this never
    leaves a tensor in the result.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return str(value)


def node_params(node: Any) -> Dict[str, Any]:
    """Return a node's defining parameters as plain JSON values.

    Derived shape metadata is skipped so the summary describes only the
    parameters a consumer would rebuild the node from.
    """
    params: Dict[str, Any] = {}
    for field in dataclasses.fields(node):
        if field.name in RESERVED_FIELDS:
            continue
        params[field.name] = to_jsonable(getattr(node, field.name))
    return params

"""Tolerant parsing: coerce untyped mappings into encode-spec fields."""

from typing import Any, Dict, Mapping, Optional, Tuple

from spikeforge.serving.errors import EncodeSpecError

#: Every field carried by a spec, in canonical order.
_FIELDS: Tuple[str, ...] = (
    "coding",
    "num_steps",
    "tau",
    "threshold",
    "clip",
    "normalize",
    "linear",
    "delta_threshold",
    "random_scale",
    "random_seed",
    "gain",
    "off_spike",
    "input_size",
    "spec_version",
)


def _as_int(value: Any, name: str) -> int:
    """Return ``value`` as an ``int`` or raise a typed spec error."""
    if isinstance(value, bool):
        raise EncodeSpecError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise EncodeSpecError(f"{name} must be an integer") from None


def _as_float(value: Any, name: str) -> float:
    """Return ``value`` as a ``float`` or raise a typed spec error."""
    if isinstance(value, bool):
        raise EncodeSpecError(f"{name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise EncodeSpecError(f"{name} must be a number") from None


def _as_bool(value: Any, name: str) -> bool:
    """Return ``value`` as a ``bool`` (used only for tolerant parsing)."""
    if isinstance(value, bool):
        return value
    raise EncodeSpecError(f"{name} must be a boolean")


def _as_optional_int(value: Any, name: str) -> Optional[int]:
    """Return ``value`` as an ``int`` or ``None``."""
    if value is None:
        return None
    return _as_int(value, name)


def _as_size(value: Any) -> Optional[Tuple[int, int]]:
    """Return ``value`` as an ``(H, W)`` pair or ``None``."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise EncodeSpecError("input_size must be an (H, W) pair")
    try:
        items = tuple(value)
    except TypeError:
        raise EncodeSpecError("input_size must be an (H, W) pair") from None
    if len(items) != 2:
        raise EncodeSpecError("input_size must be an (H, W) pair")
    return (_as_int(items[0], "input_size"), _as_int(items[1], "input_size"))


_COERCERS = {
    "coding": lambda value: str(value),
    "num_steps": lambda value: _as_int(value, "num_steps"),
    "tau": lambda value: _as_float(value, "tau"),
    "threshold": lambda value: _as_float(value, "threshold"),
    "clip": lambda value: _as_bool(value, "clip"),
    "normalize": lambda value: _as_bool(value, "normalize"),
    "linear": lambda value: _as_bool(value, "linear"),
    "delta_threshold": lambda value: _as_float(value, "delta_threshold"),
    "random_scale": lambda value: _as_float(value, "random_scale"),
    "random_seed": lambda value: _as_optional_int(value, "random_seed"),
    "gain": lambda value: _as_float(value, "gain"),
    "off_spike": lambda value: _as_bool(value, "off_spike"),
    "input_size": _as_size,
    "spec_version": lambda value: _as_int(value, "spec_version"),
}


def _coerce(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the known ``data`` keys coerced to their field types."""
    fields: Dict[str, Any] = {}
    for name in _FIELDS:
        if name in data:
            fields[name] = _COERCERS[name](data[name])
    return fields


def _from_attrs(cls: Any, cfg: Any) -> Any:
    """Return a spec built from ``cfg``'s attributes matching known fields."""
    data = {
        name: getattr(cfg, name) for name in _FIELDS if hasattr(cfg, name)
    }
    if not data:
        raise EncodeSpecError(
            f"cannot read an encode spec from {type(cfg).__name__}"
        )
    return cls.from_dict(data)

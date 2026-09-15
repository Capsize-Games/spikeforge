"""Frozen, versioned encode contract shared by training and serving.

Encoding used to be re-derived independently on the training and serving
paths, which is the classic train/serve skew. :class:`EncodeSpec` freezes the
:class:`~spikeforge.encoding.spike_encoder.SpikeEncoder` parameters into a
single typed, hashable value that both paths hand to
:func:`spikeforge.serving.preprocess.encode`. The spec carries its own
``spec_version`` so a bundle built by an older runtime can be refused instead
of silently re-encoded.

The module is pure stdlib plus the core encoder: it never imports pydantic or
the server stack, so a headless core install can freeze and compare a spec.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.serving.errors import EncodeSpecError

#: Version of the encode contract; a mismatch is refused, never re-encoded.
ENCODE_SPEC_VERSION: int = 1
#: Coding names the encoder understands; an unknown name is refused.
CODINGS: Tuple[str, ...] = ("rate", "latency", "delta", "random")

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

#: Fields that are metadata or geometry rather than coding math.
_META_FIELDS = frozenset({"spec_version", "input_size"})

#: Encode-math fields each coding actually applies.
_APPLIED: Dict[str, frozenset] = {
    "rate": frozenset({"coding", "num_steps"}),
    "latency": frozenset(
        {
            "coding",
            "num_steps",
            "tau",
            "threshold",
            "clip",
            "normalize",
            "linear",
        }
    ),
    "delta": frozenset({"coding", "num_steps", "delta_threshold"}),
    "random": frozenset(
        {"coding", "num_steps", "random_scale", "random_seed"}
    ),
}


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


def _check_coding(coding: Any) -> None:
    """Raise unless ``coding`` is a known coding name."""
    if not isinstance(coding, str) or coding not in CODINGS:
        raise EncodeSpecError(
            f"unsupported coding {coding!r}; expected one of {CODINGS}"
        )


def _check_num_steps(steps: Any) -> None:
    """Raise unless ``steps`` is a positive int."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise EncodeSpecError("num_steps must be an integer >= 1")


def _check_random_seed(seed: Any) -> None:
    """Raise unless ``seed`` is an int or None."""
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int)
    ):
        raise EncodeSpecError("random_seed must be an integer or null")


def _check_spec_version(version: Any) -> None:
    """Raise unless ``version`` matches the runtime's encode contract."""
    if version != ENCODE_SPEC_VERSION:
        raise EncodeSpecError(
            f"unsupported encode spec version {version!r}; this runtime "
            f"speaks {ENCODE_SPEC_VERSION}"
        )


@dataclass(frozen=True)
class EncodeSpec:
    """The frozen encode parameters a bundle pins and both paths share.

    The fields mirror :class:`server.schemas.encode_config.EncodeConfig` for
    the subset the encoder actually consumes. Unknown keys are ignored when
    parsing, matching the schema's ``additionalProperties: true``, so a newer
    config never fails on an older runtime.
    """

    coding: str = "rate"
    num_steps: int = 100
    tau: float = 5.0
    threshold: float = 0.01
    clip: bool = False
    normalize: bool = True
    linear: bool = True
    delta_threshold: float = 4.0
    random_scale: float = 0.5
    random_seed: Optional[int] = None
    gain: float = 1.0
    off_spike: bool = False
    input_size: Optional[Tuple[int, int]] = None
    spec_version: int = ENCODE_SPEC_VERSION

    @classmethod
    def from_mapping(cls, cfg: Any) -> "EncodeSpec":
        """Return a spec from a mapping, a ``model_dump()``-able, or ``None``.

        ``None`` yields the defaults; unknown keys are ignored and known
        keys are coerced to their field type.
        """
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        if isinstance(cfg, Mapping):
            return cls.from_dict(cfg)
        dump = getattr(cfg, "model_dump", None)
        if callable(dump):
            return cls.from_dict(dump())
        return _from_attrs(cls, cfg)

    @classmethod
    def from_dict(cls, data: Any) -> "EncodeSpec":
        """Return a spec from a mapping, ignoring unknown keys."""
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise EncodeSpecError("encode spec must be a mapping")
        return cls(**_coerce(data))

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical, JSON-ready spec including ``spec_version``."""
        fields = self._coding_fields()
        fields.update(self._meta_fields())
        return fields

    def _coding_fields(self) -> Dict[str, Any]:
        """Return the coding-math fields of the canonical dict."""
        return {
            "coding": self.coding,
            "num_steps": int(self.num_steps),
            "tau": float(self.tau),
            "threshold": float(self.threshold),
            "clip": bool(self.clip),
            "normalize": bool(self.normalize),
            "linear": bool(self.linear),
            "delta_threshold": float(self.delta_threshold),
            "random_scale": float(self.random_scale),
            "random_seed": (
                None if self.random_seed is None else int(self.random_seed)
            ),
            "gain": float(self.gain),
            "off_spike": bool(self.off_spike),
        }

    def _meta_fields(self) -> Dict[str, Any]:
        """Return the geometry/version metadata fields of the dict."""
        size: Optional[list] = None
        if self.input_size is not None:
            size = [int(self.input_size[0]), int(self.input_size[1])]
        return {"input_size": size, "spec_version": int(self.spec_version)}

    def to_encoder(self) -> SpikeEncoder:
        """Return the encoder this spec pins, without reimplementing math."""
        return SpikeEncoder.from_encode_config(self)

    def digest(self) -> str:
        """Return the hex SHA-256 of the canonical (sorted-key) spec JSON."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def unsupported(self) -> Tuple[str, ...]:
        """Return recognised fields the active coding does not apply.

        Reporting them instead of dropping them keeps the frozen spec honest:
        a caller can see that, say, ``tau`` was frozen but is inert under a
        ``rate`` coding. ``spec_version`` and ``input_size`` are metadata and
        geometry, not coding math, so they are never listed.
        """
        applied = _APPLIED.get(self.coding, frozenset())
        known = set(_FIELDS) - _META_FIELDS
        return tuple(sorted(known - applied))

    def validate(self) -> None:
        """Raise :class:`EncodeSpecError` when the spec is unusable."""
        _check_coding(self.coding)
        _check_num_steps(self.num_steps)
        _require_positive(self.tau, "tau")
        _require_positive(self.threshold, "threshold")
        _require_non_negative(self.delta_threshold, "delta_threshold")
        _require_non_negative(self.random_scale, "random_scale")
        _check_random_seed(self.random_seed)
        if self.input_size is not None:
            _require_size(self.input_size)
        _check_spec_version(self.spec_version)


def _is_number(value: Any) -> bool:
    """Return True when ``value`` is a real (non-bool) number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_positive(value: Any, name: str) -> None:
    """Raise when ``value`` is not a positive number."""
    if not _is_number(value) or value <= 0:
        raise EncodeSpecError(f"{name} must be a positive number")


def _require_non_negative(value: Any, name: str) -> None:
    """Raise when ``value`` is not a non-negative number."""
    if not _is_number(value) or value < 0:
        raise EncodeSpecError(f"{name} must be a non-negative number")


def _require_size(value: Any) -> None:
    """Raise when ``value`` is not a pair of positive integers."""
    if isinstance(value, (str, bytes)):
        raise EncodeSpecError("input_size must be an (H, W) pair")
    try:
        items = tuple(value)
    except TypeError:
        raise EncodeSpecError(
            "input_size must be an (H, W) pair"
        ) from None
    if len(items) != 2:
        raise EncodeSpecError("input_size must be an (H, W) pair")
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise EncodeSpecError(
                "input_size must be an (H, W) pair of positive integers"
            )


#: A spec or anything :meth:`EncodeSpec.from_mapping` accepts.
SpecLike = Union[EncodeSpec, Mapping[str, Any], Any]

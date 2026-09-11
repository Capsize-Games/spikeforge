"""Isolated capability probe and I/O helpers for the optional ``onnx`` stack.

This is the only module in the project that imports ``onnx`` or
``onnxruntime``. Imports happen inside functions, never at module load, so a
missing dependency never breaks importing this package; callers instead read
:func:`capability` or receive a typed
:class:`~spikeforge.onnx_bridge.errors.OnnxExtraMissingError` from
:func:`require`.

Export deliberately uses torch's TorchScript exporter (``dynamo=False``) so the
``onnx`` extra alone is sufficient: the newer dynamo exporter pulls in a
separate ``onnxscript`` dependency the declared extra does not include.
"""

from importlib import import_module
from typing import Any, Dict, List, Mapping, Optional

import torch

from spikeforge.onnx_bridge.errors import (
    OnnxExportError,
    OnnxExtraMissingError,
    OnnxGraphFileError,
    OnnxGraphNotFoundError,
)

#: The pip extra that enables this bridge.
EXTRA = "onnx"
#: Default ONNX opset written by the exporter.
DEFAULT_OPSET = 17
#: Graph input/output names used by every export.
INPUT_NAME = "spikes"
OUTPUT_NAME = "readout"


def _safe_import(name: str) -> Optional[Any]:
    """Import ``name``, returning ``None`` when it is unavailable."""
    try:
        return import_module(name)
    except ImportError:
        return None


def _version_of(module: Optional[Any]) -> Optional[str]:
    """Return the module's version string when one is exposed."""
    if module is None:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version else None


def available() -> bool:
    """Return True when the ``onnx`` package can be imported."""
    return _safe_import("onnx") is not None


def onnxruntime_available() -> bool:
    """Return True when the ``onnxruntime`` package can be imported."""
    return _safe_import("onnxruntime") is not None


def capability() -> Dict[str, Any]:
    """Return a JSON-able report of the ONNX dependency surface."""
    onnx_module = _safe_import("onnx")
    runtime = _safe_import("onnxruntime")
    return {
        "onnx_available": onnx_module is not None,
        "onnxruntime_available": runtime is not None,
        "onnx_version": _version_of(onnx_module),
        "onnxruntime_version": _version_of(runtime),
        "opset": DEFAULT_OPSET,
    }


def require() -> Any:
    """Return the ``onnx`` module or raise the typed missing-extra error."""
    onnx_module = _safe_import("onnx")
    if onnx_module is None:
        raise OnnxExtraMissingError(EXTRA)
    return onnx_module


def export_onnx(
    module: Any,
    args: Any,
    path: str,
    opset: int = DEFAULT_OPSET,
) -> None:
    """Trace ``module`` over ``args`` and write the ONNX graph to ``path``."""
    require()
    try:
        torch.onnx.export(
            module,
            args,
            path,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            opset_version=opset,
            dynamo=False,
        )
    except (RuntimeError, ValueError, TypeError) as error:
        raise OnnxExportError(str(error)) from None


def load_onnx(path: str) -> Any:
    """Load the ONNX model stored at ``path``, raising typed errors."""
    onnx_module = require()
    try:
        return onnx_module.load(path)
    except FileNotFoundError:
        raise OnnxGraphNotFoundError(path) from None
    except Exception as error:
        # onnx.load raises protobuf DecodeError, which has no narrow base.
        raise OnnxGraphFileError(path, str(error)) from None


def save_onnx(model: Any, path: str) -> None:
    """Write ``model`` back to ``path``."""
    require().save(model, path)


def metadata_values(model: Any) -> Dict[str, str]:
    """Return the model's metadata properties as a plain string mapping."""
    return {entry.key: entry.value for entry in model.metadata_props}


def set_metadata(model: Any, values: Mapping[str, str]) -> None:
    """Replace the model's metadata properties with ``values``."""
    del model.metadata_props[:]
    for key, value in values.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = str(value)


def op_types(model: Any) -> List[str]:
    """Return the sorted, de-duplicated op types used by ``model``."""
    return sorted({node.op_type for node in model.graph.node})


def _plain(value: Any) -> Any:
    """Return a protobuf attribute value as plain lists and numbers."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _initializer_dims(model: Any) -> Dict[str, List[int]]:
    """Map every initializer name to its shape as a plain int list."""
    return {
        init.name: [int(dim) for dim in init.dims]
        for init in model.graph.initializer
    }


def _attributes(helper: Any, node: Any) -> Dict[str, Any]:
    """Return a node's ONNX attributes as plain values."""
    return {
        attr.name: _plain(helper.get_attribute_value(attr))
        for attr in node.attribute
    }


def _descriptor(
    helper: Any, dims: Dict[str, List[int]], node: Any
) -> Dict[str, Any]:
    """Return the plain descriptor for one ONNX node."""
    second = node.input[1] if len(node.input) > 1 else None
    return {
        "op_type": node.op_type,
        "name": node.name,
        "inputs": list(node.input),
        "attrs": _attributes(helper, node),
        "weight_shape": dims.get(second) if second else None,
    }


def graph_ops(model: Any) -> List[Dict[str, Any]]:
    """Return one plain descriptor per ONNX node in graph order.

    Each descriptor carries ``op_type``, ``name``, ``inputs``, ``attrs`` and
    the shape of a second-input ``weight_shape`` when that input is an
    initializer, which is all :mod:`import_onnx` needs to size a stage.
    """
    helper = require().helper
    dims = _initializer_dims(model)
    return [_descriptor(helper, dims, node) for node in model.graph.node]

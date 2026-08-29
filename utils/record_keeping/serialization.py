"""Serialization utilities for saving optimization hyperparameters and objects to JSON."""
from __future__ import annotations

import inspect
import json
from typing import Any

import torch
from torchvision.transforms import v2 as T_v2


def get_fully_qualified_class_name(obj_or_type: Any) -> str:
    """Return the fully qualified import name for a class or instance.

    Args:
        obj_or_type: Class type or instantiated object.

    Returns:
        String module and class name (e.g. 'torchvision.transforms.v2.RandomAffine').
    """
    typ = obj_or_type if inspect.isclass(obj_or_type) else obj_or_type.__class__
    return f"{typ.__module__}.{typ.__name__}"


def _is_torchvision_v2_transform(obj: Any) -> bool:
    """Check whether an object is an instance of a torchvision v2 Transform."""
    try:
        return isinstance(obj, T_v2.Transform)
    except Exception:
        return getattr(obj.__class__, "__module__", "").startswith("torchvision.transforms.v2")


def _tensor_summary(t: Any, max_numel: int = 1000) -> Any:
    """Build a JSON-serializable dictionary summary of a PyTorch tensor."""
    if torch is None:
        return str(t)
    info = {
        "_type": "torch.Tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
    }
    numel = t.numel()
    if numel <= max_numel:
        info["data"] = t.detach().cpu().tolist()
    else:
        info["data"] = f"<omitted: {numel} values>"
    return info


def _serialize_transform(transform: Any) -> dict[str, Any]:
    """Serialize a torchvision v2 transform into a JSON-compatible dictionary."""
    result: dict[str, Any] = {"_type": get_fully_qualified_class_name(transform)}

    inner = None
    for attr in ("transforms", "_transforms", "ops"):
        if hasattr(transform, attr):
            val = getattr(transform, attr)
            if isinstance(val, (list, tuple)) and val and all(callable(x) for x in val):
                inner = val
                break
    if inner is not None:
        result["transforms"] = [
            _serialize_transform(t) if _is_torchvision_v2_transform(t) else {"_type": get_fully_qualified_class_name(t), "repr": repr(t)}
            for t in inner
        ]

    try:
        attrs = {k: v for k, v in vars(transform).items() if not k.startswith("_")}
    except TypeError:
        attrs = {}

    params: dict[str, Any] = {}
    for k, v in attrs.items():
        params[k] = to_serializable(v)
    if params:
        result["params"] = params
    else:
        result["repr"] = repr(transform)

    return result


def to_serializable(obj: Any) -> Any:
    """Recursively convert an arbitrary Python object into JSON-serializable structures.

    Args:
        obj: Python object to convert.

    Returns:
        JSON-compatible primitive (dict, list, string, number, bool, or None).
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if torch is not None:
        if isinstance(obj, torch.Tensor):
            return _tensor_summary(obj)
        if isinstance(obj, torch.device):
            return {"_type": "torch.device", "value": str(obj)}
        if isinstance(obj, torch.dtype):
            return {"_type": "torch.dtype", "value": str(obj)}

    if _is_torchvision_v2_transform(obj):
        return _serialize_transform(obj)

    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        return [to_serializable(v) for v in seq]

    if inspect.isclass(obj) or callable(obj):
        try:
            result = {"_type": "callable", "name": get_fully_qualified_class_name(obj)}
            try:
                source = inspect.getsource(obj)
                result["source"] = source
            except (OSError, TypeError):
                pass
            try:
                sig = inspect.signature(obj)
                result["signature"] = str(sig)
            except (ValueError, TypeError):
                pass
            return result
        except Exception:
            return str(obj)

    try:
        return {"_type": get_fully_qualified_class_name(obj), "state": {k: to_serializable(v) for k, v in vars(obj).items() if not k.startswith("_")}}
    except Exception:
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

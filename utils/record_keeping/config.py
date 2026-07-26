"""
Utilities for serializing and managing configuration settings.

This module provides:
- to_serializable: recursively converts Python objects into JSON-serializable
  structures with sensible representations for tensors, torch devices/dtypes,
  torchvision v2 transforms (including Compose), and common Python objects.
- get_fully_qualified_class_name: fully qualified class name for instances or types.
"""
from __future__ import annotations

import inspect
import json
from typing import Any

import torch
from torchvision.transforms import v2 as T_v2


def get_fully_qualified_class_name(obj_or_type: Any) -> str:
    """Return fully-qualified class name for an object or a type.

    Example: torchvision.transforms.v2.RandomAffine
    """
    typ = obj_or_type if inspect.isclass(obj_or_type) else obj_or_type.__class__
    return f"{typ.__module__}.{typ.__name__}"


def _is_torchvision_v2_transform(obj: Any) -> bool:
    try:
        # v2 exposes a base class Transform
        return isinstance(obj, T_v2.Transform)
    except Exception:
        # Fallback to module check
        return getattr(obj.__class__, "__module__", "").startswith("torchvision.transforms.v2")


def _tensor_summary(t: Any, max_numel: int = 1000) -> Any:
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
    """Serialize a torchvision v2 transform (including Compose) into a dict.

    Tries to extract relevant constructor/attribute parameters when possible and
    falls back to repr(). Supports nested Compose recursively.
    """
    result: dict[str, Any] = {"_type": get_fully_qualified_class_name(transform)}

    # Handle Compose-like containers by looking for a list/tuple of inner transforms
    inner = None
    for attr in ("transforms", "_transforms", "ops"):
        if hasattr(transform, attr):
            val = getattr(transform, attr)
            if isinstance(val, (list, tuple)) and val and all(callable(x) for x in val):
                inner = val
                break
    if inner is not None:
        result["transforms"] = [_serialize_transform(t) if _is_torchvision_v2_transform(t) else {"_type": get_fully_qualified_class_name(t), "repr": repr(t)} for t in inner]

    # Capture public attributes as params, filtering out callables/modules
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
        # Fallback to repr if no public params found
        result["repr"] = repr(transform)

    return result


def to_serializable(obj: Any) -> Any:
    """Recursively convert object to something JSON-serializable.

    Special handling for:
    - torch.Tensor
    - torch.device, torch.dtype
    - torchvision v2 transforms (including Compose)
    - dicts, lists, tuples, sets
    - callables and classes (represented by fully-qualified names)
    """
    # Basic types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # torch-specific types
    if torch is not None:
        if isinstance(obj, torch.Tensor):
            return _tensor_summary(obj)
        if isinstance(obj, torch.device):
            return {"_type": "torch.device", "value": str(obj)}
        if isinstance(obj, torch.dtype):
            return {"_type": "torch.dtype", "value": str(obj)}

    # torchvision v2 transforms
    if _is_torchvision_v2_transform(obj):
        return _serialize_transform(obj)

    # dict-like
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        return [to_serializable(v) for v in seq]

    # callable or class
    if inspect.isclass(obj) or callable(obj):
        try:
            result = {"_type": "callable", "name": get_fully_qualified_class_name(obj)}
            # Try to get the source code
            try:
                source = inspect.getsource(obj)
                result["source"] = source
            except (OSError, TypeError):
                # Source not available (built-in, C extension, or dynamically created)
                pass
            # Try to get the signature
            try:
                sig = inspect.signature(obj)
                result["signature"] = str(sig)
            except (ValueError, TypeError):
                # Signature not available
                pass
            return result
        except Exception:
            return str(obj)

    # objects with __dict__
    try:
        return {"_type": get_fully_qualified_class_name(obj), "state": {k: to_serializable(v) for k, v in vars(obj).items() if not k.startswith("_")}}
    except Exception:
        # Fallback to string representation
        try:
            json.dumps(obj)  # type: ignore
            return obj
        except Exception:
            return str(obj)

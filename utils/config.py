"""
Utilities for serializing and managing configuration settings.

This module provides:
- to_serializable: recursively converts Python objects into JSON-serializable
  structures with sensible representations for tensors, torch devices/dtypes,
  torchvision v2 transforms (including Compose), and common Python objects.
- deep_merge: deep dictionary merge (b overwrites/extends a).
- fqcn: fully qualified class name for instances or types.
"""
from __future__ import annotations

from typing import Any, Dict
import json
import inspect

try:
    # torchvision.transforms.v2 is optional in some environments
    from torchvision.transforms import v2 as T_v2
except Exception:  # pragma: no cover - only for environments without torchvision
    T_v2 = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def fqcn(obj_or_type: Any) -> str:
    """Return fully-qualified class name for an object or a type.

    Example: torchvision.transforms.v2.RandomAffine
    """
    typ = obj_or_type if inspect.isclass(obj_or_type) else obj_or_type.__class__
    return f"{typ.__module__}.{typ.__name__}"


def is_torchvision_v2_transform(obj: Any) -> bool:
    if T_v2 is None:
        return False
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


def serialize_transform(transform: Any) -> Dict[str, Any]:
    """Serialize a torchvision v2 transform (including Compose) into a dict.

    Tries to extract relevant constructor/attribute parameters when possible and
    falls back to repr(). Supports nested Compose recursively.
    """
    result: Dict[str, Any] = {"_type": fqcn(transform)}

    # Handle Compose-like containers by looking for a list/tuple of inner transforms
    inner = None
    for attr in ("transforms", "_transforms", "ops"):
        if hasattr(transform, attr):
            val = getattr(transform, attr)
            if isinstance(val, (list, tuple)) and val and all(hasattr(x, "__call__") for x in val):
                inner = val
                break
    if inner is not None:
        result["transforms"] = [serialize_transform(t) if is_torchvision_v2_transform(t) else {"_type": fqcn(t), "repr": repr(t)} for t in inner]

    # Capture public attributes as params, filtering out callables/modules
    try:
        attrs = {k: v for k, v in vars(transform).items() if not k.startswith("_")}
    except TypeError:
        attrs = {}

    params: Dict[str, Any] = {}
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
    if is_torchvision_v2_transform(obj):
        return serialize_transform(obj)

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
            result = {"_type": "callable", "name": fqcn(obj)}
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
        return {"_type": fqcn(obj), "state": {k: to_serializable(v) for k, v in vars(obj).items() if not k.startswith("_")}}
    except Exception:
        # Fallback to string representation
        try:
            json.dumps(obj)  # type: ignore
            return obj
        except Exception:
            return str(obj)


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dict b into dict a (does not mutate inputs)."""
    result = dict(a)
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

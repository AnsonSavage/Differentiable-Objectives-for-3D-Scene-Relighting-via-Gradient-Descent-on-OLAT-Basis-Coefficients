"""Utilities for preprocessing images."""

# NOTE:
# These methods were created because `preprocess` expects a PIL Image. However, our pipeline requires back propagating through the `preprocess` method in order to update the light parameters.
# This module simply replicates, using differentiable PyTorch methods, what the `preprocess` method would do in some cases (e.g., resizing, normalization), but cannot handle arbitrary transforms.
# This code could be avoided if external libraries are updated to take PyTorch tensors directly.
from typing import Any

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode, Normalize

_DEFAULT_IMAGE_PROCESSOR_MEAN = (0.48145466, 0.4578275, 0.40821073)
_DEFAULT_IMAGE_PROCESSOR_STD = (0.26862954, 0.26130258, 0.27577711)

def _extract_preprocess_meta(preprocess: Any) -> dict[str, Any]:
    """Best-effort extraction of image_size and normalization from a preprocess pipeline.

    Supports common torchvision Compose pipelines returned by open_clip.create_transform.

    Args:
        preprocess: Preprocessing pipeline object.

    Returns:
        Dictionary containing extracted metadata keys (image_size, mean, std, resize_mode, interpolation).
    """
    meta: dict[str, Any] = {
        "image_size": None,
        "mean": None,
        "std": None,
        "resize_mode": None,
        "interpolation": None,
    }
    transforms = getattr(preprocess, "transforms", None)
    if transforms is None and hasattr(preprocess, "_transforms"):
        transforms = getattr(preprocess, "_transforms", None)

    if transforms is not None:
        for t in transforms:
            cls_name = t.__class__.__name__
            if cls_name == "Resize":
                size = getattr(t, "size", None)
                if isinstance(size, int):
                    meta["image_size"] = size
                elif isinstance(size, (tuple, list)) and len(size) > 0:
                    meta["image_size"] = int(size[0])
                interp = getattr(t, "interpolation", None)
                if interp is not None:
                    meta["interpolation"] = interp
                if isinstance(size, int):
                    meta["resize_mode"] = meta["resize_mode"] or "shortest"
                else:
                    meta["resize_mode"] = meta["resize_mode"] or "squash"
            elif cls_name == "CenterCrop":
                size = getattr(t, "size", None)
                if isinstance(size, int):
                    meta["image_size"] = meta["image_size"] or size
                elif isinstance(size, (tuple, list)) and len(size) > 0:
                    meta["image_size"] = meta["image_size"] or int(size[0])
                meta["resize_mode"] = meta["resize_mode"] or "shortest"
            elif cls_name == "Normalize":
                mean = getattr(t, "mean", None)
                std = getattr(t, "std", None)
                if mean is not None and std is not None:
                    meta["mean"] = tuple(float(x) for x in mean)
                    meta["std"] = tuple(float(x) for x in std)
            elif cls_name == "ResizeKeepRatio":
                size = getattr(t, "size", None)
                if isinstance(size, int):
                    meta["image_size"] = meta["image_size"] or size
                elif isinstance(size, (tuple, list)) and len(size) > 0:
                    meta["image_size"] = meta["image_size"] or int(size[0])
                interp = getattr(t, "interpolation", None)
                if interp is not None:
                    meta["interpolation"] = interp
                longest = getattr(t, "longest", 0.0)
                meta["resize_mode"] = meta["resize_mode"] or ("longest" if longest else "shortest")
            elif cls_name == "CenterCropOrPad":
                size = getattr(t, "size", None)
                if isinstance(size, int):
                    meta["image_size"] = meta["image_size"] or size
                elif isinstance(size, (tuple, list)) and len(size) > 0:
                    meta["image_size"] = meta["image_size"] or int(size[0])
                meta["resize_mode"] = meta["resize_mode"] or "longest"
    return meta


def preprocess_image_tensor(
    tensor: torch.Tensor,
    n_px: int | None = None,
    *,
    preprocess: Any = None,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
    fallback_default_image_size: int = 224,
) -> torch.Tensor:
    """Preprocess a tensor for vision backbones using extracted transform metadata or defaults.

    Args:
        tensor: Tensor of shape [C, H, W] or [N, C, H, W] in [0, 1].
        n_px: Optional explicit output resolution override.
        preprocess: Optional torchvision Compose transform to extract parameters from.
        mean: Optional channel-wise normalization mean.
        std: Optional channel-wise normalization std.
        fallback_default_image_size: Default pixel resolution if not specified.

    Returns:
        Preprocessed image tensor matching input batch dimensionality.

    Raises:
        ValueError: If input tensor dimension is not 3 or 4.
    """
    if tensor.dim() == 3:
        batched = False
        batch = tensor.unsqueeze(0)
    elif tensor.dim() == 4:
        batched = True
        batch = tensor
    else:
        raise ValueError("Expected tensor of shape [C,H,W] or [N,C,H,W]")

    meta = _extract_preprocess_meta(preprocess) if preprocess is not None else {}
    _size = n_px or meta.get("image_size") or fallback_default_image_size
    if isinstance(_size, int):
        target_h, target_w = _size, _size
    elif isinstance(_size, (tuple, list)) and len(_size) >= 2:
        target_h, target_w = int(_size[0]), int(_size[1])
    else:
        target_h, target_w = fallback_default_image_size, fallback_default_image_size
    m = mean or meta.get("mean") or _DEFAULT_IMAGE_PROCESSOR_MEAN
    s = std or meta.get("std") or _DEFAULT_IMAGE_PROCESSOR_STD
    resize_mode = meta.get("resize_mode") or "shortest"
    interp = meta.get("interpolation")
    if isinstance(interp, InterpolationMode):
        interpolation = interp
    elif isinstance(interp, str):
        interpolation = InterpolationMode.BICUBIC if interp == "bicubic" else InterpolationMode.BILINEAR
    else:
        interpolation = InterpolationMode.BICUBIC

    def _center_crop_or_pad(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        _, h, w = x.shape
        pad_l = max((out_w - w) // 2, 0)
        pad_r = max(out_w - w - pad_l, 0)
        pad_t = max((out_h - h) // 2, 0)
        pad_b = max(out_h - h - pad_t, 0)
        if pad_l or pad_r or pad_t or pad_b:
            x = F.pad(x, [pad_l, pad_t, pad_r, pad_b])
            _, h, w = x.shape
        top = max((h - out_h) // 2, 0)
        left = max((w - out_w) // 2, 0)
        return F.crop(x, top, left, out_h, out_w)

    processed = []
    for img in batch:
        c, h, w = img.shape
        if resize_mode == "squash":
            x = F.resize(img, [target_h, target_w], interpolation=interpolation, antialias=True)
        elif resize_mode == "shortest":
            scale = max(target_h / h, target_w / w)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            x = F.resize(img, [new_h, new_w], interpolation=interpolation, antialias=True)
            x = F.center_crop(x, [target_h, target_w])
        else:  # 'longest'
            scale = min(target_h / h, target_w / w)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            x = F.resize(img, [new_h, new_w], interpolation=interpolation, antialias=True)
            x = _center_crop_or_pad(x, target_h, target_w)
        x = Normalize(mean=m, std=s)(x)
        processed.append(x)
    out = torch.stack(processed, dim=0)
    return out if batched else out.squeeze(0)


def preprocess_image_input(
    image: torch.Tensor | Image.Image,
    *,
    preprocess: Any,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Preprocess PIL Image or PyTorch Tensor input for vision backbones differentiably.

    Args:
        image: PIL Image or Tensor [C, H, W] / [N, C, H, W].
        preprocess: Preprocessing transform pipeline.
        device: Target PyTorch device.

    Returns:
        Batched tensor of shape [N, C, H, W] on target device.

    Raises:
        TypeError: If input type is neither PIL Image nor torch.Tensor.
    """
    if isinstance(image, Image.Image):
        t = preprocess(image).unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        t = preprocess_image_tensor(image, preprocess=preprocess)
        if t.dim() == 3:
            t = t.unsqueeze(0)
    else:
        raise TypeError(f"Expected PIL Image or torch.Tensor, got {type(image)}")

    if device is not None:
        t = t.to(device)
    return t

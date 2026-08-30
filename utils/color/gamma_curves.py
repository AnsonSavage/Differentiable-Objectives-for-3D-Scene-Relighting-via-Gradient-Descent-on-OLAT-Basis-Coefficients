"""Color space conversion helpers for linear Rec.709 to sRGB."""
import numpy as np
import torch


def linear_to_srgb(linear_rgb: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """Convert a linear RGB tensor to standard sRGB.

    Args:
        linear_rgb: Tensor of arbitrary shape with values in [0, 1].
        clamp: Whether to clamp output values to [0, 1].

    Returns:
        Tensor of same shape in sRGB color space.
    """
    threshold = 0.0031308
    below = linear_rgb <= threshold
    above = ~below

    srgb = torch.zeros_like(linear_rgb)
    srgb[below] = 12.92 * linear_rgb[below]
    srgb[above] = 1.055 * torch.pow(linear_rgb[above], 1.0 / 2.4) - 0.055

    if clamp:
        srgb = srgb.clamp(0.0, 1.0)
    return srgb


def linear_to_srgb_numpy(linear_rgb: np.ndarray) -> np.ndarray:
    """Convert a linear RGB numpy array to sRGB via PyTorch.

    Args:
        linear_rgb: NumPy array with float values in [0, 1].

    Returns:
        NumPy array of same shape in sRGB color space.
    """
    arr = (
        linear_rgb
        if np.issubdtype(linear_rgb.dtype, np.floating)
        else linear_rgb.astype(np.float32, copy=False)
    )
    t = torch.from_numpy(arr)
    srgb_t = linear_to_srgb(t, clamp=True)
    return srgb_t.detach().cpu().numpy()

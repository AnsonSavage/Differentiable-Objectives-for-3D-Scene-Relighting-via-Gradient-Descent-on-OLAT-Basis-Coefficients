"""Differentiable color space conversion between HSV and RGB."""
import torch


def _differentiable_hsv_to_rgb_google(img: torch.Tensor) -> torch.Tensor:
    """Convert an image tensor from HSV to RGB using a differentiable piecewise formulation. This approach was coded by Gemini in Fall 2025.

    Args:
        img: Tensor of shape (..., 3, ...) where channels represent HSV with values in [0, 1].

    Returns:
        Tensor of identical shape in RGB color space.
    """
    h, s, v = img.unbind(dim=-3)
    h = h % 1.0
    h6 = h * 6.0

    p = v * (1.0 - s)

    f0 = h6
    r0, g0, b0 = v, v * (1.0 - s * (1.0 - f0)), p

    f1 = h6 - 1.0
    r1, g1, b1 = v * (1.0 - s * f1), v, p

    f2 = h6 - 2.0
    r2, g2, b2 = p, v, v * (1.0 - s * (1.0 - f2))

    f3 = h6 - 3.0
    r3, g3, b3 = p, v * (1.0 - s * f3), v

    f4 = h6 - 4.0
    r4, g4, b4 = v * (1.0 - s * (1.0 - f4)), p, v

    f5 = h6 - 5.0
    r5, g5, b5 = v, p, v * (1.0 - s * f5)

    r = torch.where(h6 < 5.0, r4, r5)
    g = torch.where(h6 < 5.0, g4, g5)
    b = torch.where(h6 < 5.0, b4, b5)
    r = torch.where(h6 < 4.0, r3, r)
    g = torch.where(h6 < 4.0, g3, g)
    b = torch.where(h6 < 4.0, b3, b)
    r = torch.where(h6 < 3.0, r2, r)
    g = torch.where(h6 < 3.0, g2, g)
    b = torch.where(h6 < 3.0, b2, b)

    r = torch.where(h6 < 2.0, r1, r)
    g = torch.where(h6 < 2.0, g1, g)
    b = torch.where(h6 < 2.0, b1, b)
    r = torch.where(h6 < 1.0, r0, r)
    g = torch.where(h6 < 1.0, g0, g)
    b = torch.where(h6 < 1.0, b0, b)

    return torch.stack((r, g, b), dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert an image tensor from HSV to RGB color space differentiably.

    Args:
        image: Tensor of shape (..., 3, H, W) or (..., 3) with HSV values in [0, 1].

    Returns:
        Tensor of identical shape in RGB color space.
    """
    return _differentiable_hsv_to_rgb_google(image)
"""Converters for transforming linear Rec.709 color to display sRGB."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

import torch

from utils.color.gamma_curves import linear_to_srgb
from utils.color.tonemapping.agx_looks import AgXLook
from utils.color.tonemapping.agx_utils import applyAgXLogTorch, applyAgxLutTorch


class LinearRec709TosRGB(ABC):
    """Abstract base class for linear Rec. 709 to sRGB converters."""

    @abstractmethod
    def convert(self, linear_rgb: torch.Tensor) -> torch.Tensor:
        """Convert linear Rec. 709 RGB to sRGB.

        Args:
            linear_rgb: Tensor of shape (3, H, W) or (N, 3, H, W), with values in [0, 1].

        Returns:
            Tensor of same shape in sRGB color space.
        """

    def __call__(self, linear_rec_709: torch.Tensor) -> torch.Tensor:
        """Apply converter by forwarding to convert().

        Args:
            linear_rec_709: Input tensor in linear Rec. 709.

        Returns:
            Converted sRGB tensor.
        """
        return self.convert(linear_rec_709)

    def settings_info(self) -> dict:
        """Return a serializable description dictionary for experiment logging.

        Returns:
            Dictionary with converter name and parameters.
        """
        return {"name": self.__class__.__name__}


class SimpleGammaCurve(LinearRec709TosRGB):
    """Linear Rec. 709 to sRGB converter using standard IEC 61966-2-1 piecewise gamma."""

    @override
    def convert(self, linear_rgb: torch.Tensor) -> torch.Tensor:
        """Convert linear Rec. 709 RGB to sRGB using standard piecewise gamma.

        Args:
            linear_rgb: Tensor of shape (3, H, W) or (N, 3, H, W).

        Returns:
            Tensor of same shape in sRGB.

        Raises:
            ValueError: If input tensor dimension is not 3 or 4.
        """
        if linear_rgb.dim() == 3:
            linear_rgb_batched = linear_rgb.unsqueeze(0)
            squeeze_back = True
        elif linear_rgb.dim() == 4:
            linear_rgb_batched = linear_rgb
            squeeze_back = False
        else:
            raise ValueError(f"Expected input of shape (3,H,W) or (N,3,H,W), got {tuple(linear_rgb.shape)}")

        srgb = linear_to_srgb(linear_rgb_batched.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return srgb.squeeze(0) if squeeze_back else srgb


class LinearRec709ToAgXBase(LinearRec709TosRGB):
    """Linear Rec. 709 to sRGB converter using a differentiable implementation of the AgX view transform created for Blender (https://developer.blender.org/docs/release_notes/4.0/color_management/)."""

    def __init__(self, look: AgXLook | None = None):
        """Initialize AgX tonemapping converter.

        Args:
            look: Optional AgXLook modification (e.g., AgXPunchyLook).
        """
        self.look = look

    @override
    def convert(self, linear_rgb: torch.Tensor) -> torch.Tensor:
        """Convert linear Rec. 709 RGB to sRGB using AgX tonemapping.

        Args:
            linear_rgb: Tensor of shape (3, H, W) or (N, 3, H, W).

        Returns:
            Tonemapped sRGB tensor of same shape.

        Raises:
            ValueError: If input tensor dimension is not 3 or 4.
        """
        if linear_rgb.dim() == 3:
            linear_rgb_batched = linear_rgb.unsqueeze(0)
            squeeze_back = True
        elif linear_rgb.dim() == 4:
            linear_rgb_batched = linear_rgb
            squeeze_back = False
        else:
            raise ValueError(f"Expected input of shape (3,H,W) or (N,3,H,W), got {tuple(linear_rgb.shape)}")

        log = applyAgXLogTorch(linear_rgb_batched.permute(0, 2, 3, 1))
        agx_base = applyAgxLutTorch(log)
        if self.look is not None:
            agx_base = self.look(agx_base)
        srgb = agx_base.permute(0, 3, 1, 2)
        return srgb.squeeze(0) if squeeze_back else srgb

    @override
    def settings_info(self) -> dict:
        """Return serialized converter description including active look.

        Returns:
            Dictionary containing converter and look names.
        """
        info = super().settings_info()
        if self.look is not None:
            look_name = getattr(self.look, "name", None)
            if look_name is None:
                look_name = self.look.__class__.__name__
            info["look"] = look_name
        return info
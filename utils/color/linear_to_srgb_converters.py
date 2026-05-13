"""Base classes related to Rec.709 to sRGB conversion."""
from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from utils.color.gamma_curves import linear_to_srgb
from utils.color.tonemapping.agx_utils import applyAgXLogTorch, applyAgxLutTorch
from utils.color.tonemapping.agx_looks import AgXLook


class LinearRec709TosRGB(ABC):
    """Abstract base class for linear Rec. 709 to sRGB conversion."""

    @abstractmethod
    def convert(self, linear_rgb: torch.Tensor) -> torch.Tensor:
        """Convert linear Rec. 709 RGB to sRGB.

        Args:
            linear_rgb: Tensor of shape (3, H, W) or (N, 3, H, W), values in [0, 1]

        Returns:
            Tensor of same shape, values in [0, 1]
        """
        raise NotImplementedError

    def __call__(self, linear_rec_709: torch.Tensor) -> torch.Tensor:
        """Enable calling the converter instance directly.

        This simply forwards to .convert(linear_rgb).
        """
        return self.convert(linear_rec_709)

    def settings_info(self) -> dict:
        """Return a serializable description of this converter for settings.json.

        Base implementation reports only the converter name; subclasses can
        extend with additional fields (e.g., look).
        """
        return {"name": self.__class__.__name__}

class SimpleGammaCurve(LinearRec709TosRGB):
    """Simple gamma curve implementation of LinearRec709TosRGB."""

    def convert(self, linear_rgb: torch.Tensor) -> torch.Tensor:
        """Convert linear Rec. 709 RGB to sRGB using a simple gamma curve.

        Args:
            linear_rgb: Tensor of shape (3, H, W) or (N, 3, H, W), values in [0, 1]

        Returns:
            srgb: Tensor of same shape, values in [0, 1]
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
    """Linear Rec. 709 to sRGB conversion using AgX tonemapping."""

    def __init__(self, look: AgXLook=None):
        """Initialize with a specific AgX look.

        Args:
            look: An instance of AgXLook defining the post processing
        """
        self.look = look

    def convert(self, linear_rgb: torch.Tensor) -> torch.Tensor:
        """Convert linear Rec. 709 RGB to sRGB using AgX tonemapping.

        Args:
            linear_rgb: Tensor of shape (3, H, W) or (N, 3, H, W), values in [0, 1]

        Returns:
            srgb: Tensor of same shape, values in [0, 1]
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

    def settings_info(self) -> dict:
        info = super().settings_info()
        if self.look is not None:
            look_name = getattr(self.look, "name", None)
            if look_name is None:
                look_name = self.look.__class__.__name__
            info["look"] = look_name
        return info
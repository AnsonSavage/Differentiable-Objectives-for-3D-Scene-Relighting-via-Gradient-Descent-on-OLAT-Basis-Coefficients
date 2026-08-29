"""AgX creative look transformations (e.g. Punchy)."""
from abc import ABC, abstractmethod

import torch

from utils.color.tonemapping.agx_utils import applyLookPunchyTorch


class AgXLook(ABC):
    """Abstract base class for creative look modifications applied to AgX Base."""

    def __call__(self, agx_base: torch.Tensor) -> torch.Tensor:
        """Apply look modification by forwarding to apply().

        Args:
            agx_base: Tensor in AgX Base color space with values in [0, 1].

        Returns:
            Modified tensor in [0, 1].
        """
        return self.apply(agx_base)

    @abstractmethod
    def apply(self, array: torch.Tensor) -> torch.Tensor:
        """Apply the AgX look modification to the input tensor.

        Args:
            array: Tensor of shape (..., 3), values in [0, 1].

        Returns:
            Tensor of same shape, values in [0, 1].
        """


class AgXPunchyLook(AgXLook):
    """AgX Punchy look implementing contrast (power) and saturation boost."""

    def __init__(self, punchy_gamma: float = 1.3, punchy_saturation: float = 1.2, preserve_range: bool = True):
        """Initialize AgX Punchy look.

        Args:
            punchy_gamma: Gamma exponent power for contrast adjustment (default 1.3).
            punchy_saturation: Saturation boost factor (default 1.2).
            preserve_range: If True, uses gamut-safe saturation clamping.
        """
        self.punchy_gamma = punchy_gamma
        self.punchy_saturation = punchy_saturation
        self.preserve_range = preserve_range

    def apply(self, agx_base: torch.Tensor) -> torch.Tensor:
        """Apply the AgX punchy look to the input tensor.

        Args:
            agx_base: Tensor of shape (..., 3), values in [0, 1].

        Returns:
            Contrast- and saturation-boosted tensor in [0, 1].
        """
        return applyLookPunchyTorch(agx_base, self.punchy_gamma, self.punchy_saturation, self.preserve_range)
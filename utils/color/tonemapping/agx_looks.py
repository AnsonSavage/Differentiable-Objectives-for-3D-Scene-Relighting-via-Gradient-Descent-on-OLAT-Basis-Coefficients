from abc import ABC, abstractmethod
import torch
from utils.color.tonemapping.agx_utils import applyLookPunchyTorch

class AgXLook(ABC):
    """Abstract base class for AgX look modifications."""
    def __call__(self, agx_base: torch.Tensor) -> torch.Tensor:
        return self.apply(agx_base)
    
    @abstractmethod
    def apply(self, array: torch.Tensor) -> torch.Tensor:
        """Apply the AgX look modification to the input tensor.

        Args:
            array: Tensor of shape (..., 3), values in [0, 1]

        Returns:
            Tensor of same shape, values in [0, 1]
        """

class AgXPunchyLook(AgXLook):
    """AgX punchy look implementation."""
    def __init__(self, punchy_gamma: float = 1.3, punchy_saturation: float = 1.2, preserve_range: bool = True):
        self.punchy_gamma = punchy_gamma
        self.punchy_saturation = punchy_saturation
        self.preserve_range = preserve_range

    def apply(self, agx_base: torch.Tensor) -> torch.Tensor:
        """Apply the AgX punchy look to the input tensor.

        Args:
            array: Tensor of shape (..., 3), values in [0, 1]

        Returns:
            Tensor of same shape, values in [0, 1]
        """
        return applyLookPunchyTorch(agx_base, self.punchy_gamma, self.punchy_saturation, self.preserve_range)
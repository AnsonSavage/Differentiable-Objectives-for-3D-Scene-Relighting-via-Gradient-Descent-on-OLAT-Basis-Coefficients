"""Synthetic lightweight dummy scene for fast unit and workflow testing."""
import torch

from utils.color.linear_to_srgb_converters import (
    LinearRec709ToAgXBase,
    LinearRec709TosRGB,
)
from utils.scene import Scene


class DummyScene(Scene):
    """A lightweight in-memory OLAT scene for fast headless testing.

    Provides synthetic linear multi-light tensors without reading EXR files from disk.
    """

    def __init__(
        self,
        num_lights: int = 4,
        height: int = 32,
        width: int = 32,
        has_non_optimized: bool = True,
        has_alpha_mask: bool = True,
        device: str = "cpu",
        name: str = "dummy_scene",
        description: str = "A synthetic dummy scene for testing",
    ):
        self.name = name
        self.description = description
        self.device = device
        self._num_lights = num_lights
        self._height = height
        self._width = width

        # Deterministic synthetic tensors for testing
        torch.manual_seed(42)
        self._images = (
            torch.rand(num_lights, height, width, 3, device=device, dtype=torch.float32) * 0.2 + 0.05
        )
        self._non_optimized = (
            torch.rand(height, width, 3, device=device, dtype=torch.float32) * 0.05
            if has_non_optimized
            else None
        )
        self._alpha_mask = (
            torch.ones(height, width, 1, device=device, dtype=torch.float32)
            if has_alpha_mask
            else None
        )
        self._light_names = [f"light_{i:03d}" for i in range(num_lights)]

    def get_optimizable_images(self) -> torch.Tensor:
        """Returns optimizable light images tensor of shape (num_lights, H, W, 3)."""
        return self._images

    def get_non_optimized_lights(self) -> torch.Tensor | None:
        """Returns non-optimized ambient/base light tensor of shape (H, W, 3) or None."""
        return self._non_optimized

    def get_alpha_mask(self) -> torch.Tensor | None:
        """Returns alpha mask tensor of shape (H, W, 1) or None."""
        return self._alpha_mask

    def get_light_name_list(self) -> list[str]:
        """Returns list of light names."""
        return list(self._light_names)

    def to(self, device: str) -> DummyScene:
        """Move tensors to the specified device."""
        self.device = device
        self._images = self._images.to(device=device)
        if self._non_optimized is not None:
            self._non_optimized = self._non_optimized.to(device=device)
        if self._alpha_mask is not None:
            self._alpha_mask = self._alpha_mask.to(device=device)
        return self

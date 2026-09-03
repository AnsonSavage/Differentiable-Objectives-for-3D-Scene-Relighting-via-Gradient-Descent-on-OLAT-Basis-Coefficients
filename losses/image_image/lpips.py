"""Learned Perceptual Image Patch Similarity (LPIPS) loss."""
from typing import override

import torch
from PIL import Image

from losses.image_image.base import ImageImageLoss


class LPIPSLoss(ImageImageLoss):
    """Learned Perceptual Image Patch Similarity (LPIPS) loss."""

    def __init__(
        self,
        reference_image: torch.Tensor | Image.Image | str,
        comparison_height: int | None = None,
        comparison_width: int | None = None,
        device: str = "cuda",
        backbone: str = "vgg",
    ):
        """Initialize LPIPS perceptual loss.

        Args:
            reference_image: Reference target image.
            comparison_height: Optional comparison height.
            comparison_width: Optional comparison width.
            device: PyTorch device.
            backbone: Backbone feature network ("vgg", "alex", etc.).
        """
        super().__init__(reference_image, comparison_height, comparison_width, device)
        from lpips import LPIPS

        self.lpips = LPIPS(net=backbone).to(device)

    @override
    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
        target = self.processed_target_image
        if incoming_image.dim() == 3:
            incoming_image = incoming_image.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        # normalize=True internally normalizes [0, 1] to [-1, 1] range
        return self.lpips(incoming_image, target, normalize=True).squeeze()

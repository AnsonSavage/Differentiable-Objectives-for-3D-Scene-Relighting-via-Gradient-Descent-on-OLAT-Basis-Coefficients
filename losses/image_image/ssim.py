"""Structural Similarity Index (SSIM) loss."""
from typing import override

import torch
from PIL import Image

from losses.image_image.base import ImageImageLoss


class SSIMLoss(ImageImageLoss):
    """Mean Structural Similarity Index (SSIM) loss (1 - SSIM)."""

    def __init__(
        self,
        reference_image: torch.Tensor | Image.Image | str,
        comparison_height: int | None = None,
        comparison_width: int | None = None,
        device: str = "cuda",
    ):
        """Initialize SSIM loss.

        Args:
            reference_image: Reference target image.
            comparison_height: Optional comparison height.
            comparison_width: Optional comparison width.
            device: PyTorch device.
        """
        super().__init__(reference_image, comparison_height, comparison_width, device)
        from pytorch_msssim import SSIM

        self.ssim_metric = SSIM(data_range=1.0, size_average=False, channel=3)

    @override
    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
        target = self.processed_target_image
        if incoming_image.dim() == 3:
            incoming_image = incoming_image.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        return torch.tensor(1.0) - self.ssim_metric(incoming_image, target)

"""Pixel-level image-to-image losses (MSE, L1)."""
from typing import override

import torch
from torch import nn

from losses.image_image.base import ImageImageLoss


class MSELossWithReferenceImage(ImageImageLoss):
    """Mean Squared Error (MSE) loss with reference image target."""

    @override
    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(incoming_image, self.processed_target_image)


class L1LossWithReferenceImage(ImageImageLoss):
    """L1 Loss (Mean Absolute Error) with reference image target."""

    @override
    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
        return nn.functional.l1_loss(incoming_image, self.processed_target_image)

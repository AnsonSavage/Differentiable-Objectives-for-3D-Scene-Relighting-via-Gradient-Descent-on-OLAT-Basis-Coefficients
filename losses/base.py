"""Base classes for loss functions used in image optimization."""
from abc import ABC, abstractmethod

from torch import nn


class BaseLoss(nn.Module, ABC):
    """Base class for all loss functions."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, image):
        """Calculate loss for the given image.

        Args:
            image: Image tensor to evaluate, shape [C, H, W] or [N, C, H, W].

        Returns:
            Loss value as a scalar tensor.
        """

    @abstractmethod
    def get_prompt_info(self) -> dict:
        """Get prompt and configuration information for logging.

        Returns:
            Dictionary containing metadata specific to this loss.
        """


class UpdatableLoss(BaseLoss):
    """Base class for losses with step-dependent parameters."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def update_parameters(self, current_step: int, total_steps: int, **kwargs):
        """Update internal parameters based on optimization progress.

        Args:
            current_step: Current optimization step.
            total_steps: Total number of optimization steps.
            **kwargs: Additional keyword arguments for parameter updates.
        """
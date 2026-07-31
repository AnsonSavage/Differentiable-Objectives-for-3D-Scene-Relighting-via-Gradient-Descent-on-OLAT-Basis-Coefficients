"""
Base classes for loss functions used in image optimization.
"""
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
            image: Image tensor to evaluate, shape [C, H, W]

        Returns:
            Loss value as a scalar tensor
        """

    @abstractmethod
    def get_prompt_info(self):
        """Get prompt information from this loss for logging.

        Returns:
            Dictionary containing prompt information specific to this loss
        """


class UpdatableLoss(BaseLoss):
    """Base class for losses that have internal parameters that can be updated based on the number of steps into the optimization."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def update_parameters(self, current_step: int, total_steps: int, **kwargs):
        """Update the internal parameters of the loss function.

        Args:
            current_step: Current optimization step
            total_steps: Total number of optimization steps
            kwargs: Additional arguments needed for updating parameters
        """
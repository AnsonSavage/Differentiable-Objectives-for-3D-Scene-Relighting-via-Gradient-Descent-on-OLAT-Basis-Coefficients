"""Base class for image-to-image losses."""
from abc import ABC, abstractmethod
from typing import override

import torch
import torchvision.transforms.functional as F
from PIL import Image

from losses.base import BaseLoss
from utils.image.image import resize_then_crop


class ImageImageLoss(BaseLoss, ABC):
    """Abstract base class for comparing generated images against reference targets."""

    def __init__(
        self,
        reference_image: torch.Tensor | Image.Image | str,
        comparison_height: int | None = None,
        comparison_width: int | None = None,
        device: str = "cuda",
    ):
        """Initialize ImageImageLoss with a reference target image.

        Args:
            reference_image: Target reference as a tensor, PIL Image, or file path.
            comparison_height: Optional height to resize both images to for comparison.
            comparison_width: Optional width to resize both images to for comparison.
            device: PyTorch device.
        """
        super().__init__()
        device_obj = torch.device(device)
        self.device = device_obj

        self.image_path = None
        if isinstance(reference_image, str):
            self.image_path = reference_image
            reference_image = Image.open(reference_image).convert("RGB")

        if isinstance(reference_image, Image.Image):
            reference_image = F.to_tensor(reference_image)

        self.reference_image = reference_image.float()
        self.comparison_height, self.comparison_width = comparison_height, comparison_width
        self.processed_target_image = self.preprocess(self.reference_image)
        if self.comparison_height is None or self.comparison_width is None:
            self.comparison_height, self.comparison_width = (
                self.processed_target_image.shape[-2],
                self.processed_target_image.shape[-1],
            )
            print(
                f"Initialized {self.__class__.__name__} with comparison resolution: "
                f"{self.comparison_height}x{self.comparison_width}"
            )

    def preprocess(self, img: torch.Tensor | Image.Image, verbose: bool = False) -> torch.Tensor:
        """Preprocess an image or batch for loss comparison.

        Args:
            img: PIL Image or torch.Tensor of shape [H, W], [C, H, W], or [N, C, H, W].
            verbose: If True, prints resizing information.

        Returns:
            Preprocessed 4D tensor on the target device.
        """
        if isinstance(img, Image.Image):
            img = F.to_tensor(img)
        if img.dim() == 2:
            # In this case, the image is grayscale, so we add a channel dimension
            img = img.unsqueeze(0)
        assert img.shape[-3] in (1, 3), "Image must have 1 or 3 channels"

        img = img.float().to(self.device)

        # Add batch dimension if necessary
        img = img.unsqueeze(0) if img.dim() == 3 else img

        if self.comparison_height is not None and self.comparison_width is not None:
            if hasattr(self, "processed_target_image") and self.processed_target_image is not None:
                if img.shape[2] != self.comparison_height or img.shape[3] != self.comparison_width:
                    if verbose:
                        print(
                            f"Resizing input image from ({img.shape[2]}, {img.shape[3]}) to "
                            f"({self.comparison_height}, {self.comparison_width}) for comparison."
                        )
                    img = resize_then_crop(
                        img, target_height=self.comparison_height, target_width=self.comparison_width
                    )
            else:
                img = resize_then_crop(img, target_height=self.comparison_height, target_width=self.comparison_width)
        return img

    @override
    def get_prompt_info(self) -> dict:
        return {
            "image_image_loss_type": self.__class__.__name__,
            "reference_image_resolution": list(self.reference_image.shape),
            "processed_target_image_resolution": list(self.processed_target_image.shape),
            "reference_image_path": self.image_path,
        }

    @override
    def forward(self, image) -> torch.Tensor:
        incoming_image = self.preprocess(image)
        assert incoming_image.shape[1:] == self.processed_target_image.shape[1:], (
            f"Image shape mismatch: Incoming image shape: {incoming_image.shape}, "
            f"Target Image Shape: {self.processed_target_image.shape}"
        )
        if incoming_image.shape[0] != self.processed_target_image.shape[0]:
            self.processed_target_image = self.processed_target_image.expand(incoming_image.shape[0], -1, -1, -1)
        return self._loss_implementation(incoming_image)

    @abstractmethod
    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
        """Subclass implementation of the specific distance metric between the incoming_image and the reference image.

        Args:
            incoming_image: Preprocessed incoming image tensor.

        Returns:
            Loss tensor value.
        """

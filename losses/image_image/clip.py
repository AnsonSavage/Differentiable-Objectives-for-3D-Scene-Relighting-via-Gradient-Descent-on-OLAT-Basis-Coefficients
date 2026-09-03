"""Image-to-Image CLIP feature cosine distance loss."""
from typing import Any, override

import torch
from PIL import Image

from losses.image_image.base import ImageImageLoss
from losses.loss_utils import compute_cosine_distance
from utils.image.preprocess_utils import preprocess_image_input


class ImageImageCLIPLoss(ImageImageLoss):
    """Image-to-Image CLIP feature cosine distance loss."""

    def __init__(
        self,
        reference_image: torch.Tensor | Image.Image | str,
        clip_model: Any,
        preprocess: Any,
        comparison_height: int | None = None,
        comparison_width: int | None = None,
        device: str = "cuda",
    ):
        """Initialize CLIP feature image-to-image loss.

        Args:
            reference_image: Reference target image.
            clip_model: Pretrained CLIP model.
            preprocess: CLIP preprocessing transform.
            comparison_height: Optional comparison height.
            comparison_width: Optional comparison width.
            device: PyTorch device.
        """
        self.clip_preprocess = preprocess
        self.device = device
        super().__init__(reference_image, comparison_height, comparison_width, device)
        self.clip_model = clip_model
        with torch.no_grad():
            self.processed_target_image_features = self.clip_model.encode_image(self.processed_target_image)
            self.processed_target_image_features = (
                self.processed_target_image_features / self.processed_target_image_features.norm(dim=1, keepdim=True)
            )
        self.single_image_comparison_mode = self.processed_target_image_features.shape[0] == 1

    @override
    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
        image_features = self.clip_model.encode_image(incoming_image)
        return compute_cosine_distance(
            self.processed_target_image_features, image_features, is_static_embedding_prenormalized=True
        )

    @override
    def preprocess(self, img: torch.Tensor | Image.Image) -> torch.Tensor:
        original = super().preprocess(img)
        return preprocess_image_input(original, preprocess=self.clip_preprocess, device=self.device)

"""Image-to-image losses for image comparison and relighting optimization."""
from losses.image_image.base import ImageImageLoss
from losses.image_image.clip import ImageImageCLIPLoss
from losses.image_image.embedding_similarity import ImageEmbeddingSimilarityLoss
from losses.image_image.lpips import LPIPSLoss
from losses.image_image.standard_error import (
    L1LossWithReferenceImage,
    MSELossWithReferenceImage,
)
from losses.image_image.ssim import SSIMLoss
from losses.image_image.vgg_style import VGGStyleTransferLoss

__all__ = [
    "ImageImageLoss",
    "MSELossWithReferenceImage",
    "L1LossWithReferenceImage",
    "SSIMLoss",
    "LPIPSLoss",
    "ImageImageCLIPLoss",
    "VGGStyleTransferLoss",
    "ImageEmbeddingSimilarityLoss",
]

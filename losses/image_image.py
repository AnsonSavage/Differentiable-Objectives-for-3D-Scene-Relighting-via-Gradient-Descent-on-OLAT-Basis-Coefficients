"""Image-to-image losses for image optimization (MSE, SSIM, LPIPS, Style, Embedding Cosine Distance).

Note:
    All color space conversions (e.g., linear to sRGB) must be handled before passing tensors.
"""
from abc import ABC, abstractmethod
from typing import Any

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import nn

from losses.base import BaseLoss
from losses.loss_utils import compute_cosine_distance
from utils.image.image import resize_then_crop
from utils.image.preprocess_utils import preprocess_image_input


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
            self.comparison_height, self.comparison_width = self.processed_target_image.shape[-2], self.processed_target_image.shape[-1]
            print(f"Initialized {self.__class__.__name__} with comparison resolution: {self.comparison_height}x{self.comparison_width}")

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
                        print(f"Resizing input image from ({img.shape[2]}, {img.shape[3]}) to ({self.comparison_height}, {self.comparison_width}) for comparison.")
                    img = resize_then_crop(img, target_height=self.comparison_height, target_width=self.comparison_width)
            else:
                img = resize_then_crop(img, target_height=self.comparison_height, target_width=self.comparison_width)
        return img

    def get_prompt_info(self) -> dict:
        """Get prompt and configuration information for logging.

        Returns:
            Dictionary containing reference image resolution and metadata.
        """
        return {
            "image_image_loss_type": self.__class__.__name__,
            "reference_image_resolution": list(self.reference_image.shape),
            "processed_target_image_resolution": list(self.processed_target_image.shape),
            "reference_image_path": self.image_path,
        }

    def forward(self, image) -> torch.Tensor:
        """Compute image-to-image loss between input image and reference.

        Args:
            image: Input image tensor or PIL Image.

        Returns:
            Computed scalar or batch loss tensor.
        """
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
        """Subclass implementation of the specific distance metric.

        Args:
            incoming_image: Preprocessed incoming image tensor.

        Returns:
            Loss tensor value.
        """


class MSELossWithReferenceImage(ImageImageLoss):
    """Mean Squared Error (MSE) loss with reference image target."""

    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
        """Calculate MSE loss between input and reference image."""
        return nn.functional.mse_loss(incoming_image, self.processed_target_image)


class L1LossWithReferenceImage(ImageImageLoss):
    """L1 Loss (Mean Absolute Error) with reference image target."""

    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
        """Calculate L1 loss between input and reference image."""
        return nn.functional.l1_loss(incoming_image, self.processed_target_image)


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

    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
        target = self.processed_target_image
        if incoming_image.dim() == 3:
            incoming_image = incoming_image.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        return torch.tensor(1.0) - self.ssim_metric(incoming_image, target)


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

    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
        """Calculate LPIPS perceptual distance."""
        target = self.processed_target_image
        if incoming_image.dim() == 3:
            incoming_image = incoming_image.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        # normalize=True internally normalizes [0, 1] to [-1, 1] range
        return self.lpips(incoming_image, target, normalize=True).squeeze()


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

    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
        """Calculate cosine distance between CLIP feature embeddings."""
        image_features = self.clip_model.encode_image(incoming_image)
        return compute_cosine_distance(self.processed_target_image_features, image_features, is_static_embedding_prenormalized=True)

    def preprocess(self, img: torch.Tensor | Image.Image) -> torch.Tensor:
        """Apply base preprocessing followed by CLIP preprocessing."""
        original = super().preprocess(img)
        return preprocess_image_input(original, preprocess=self.clip_preprocess, device=self.device)


class VGGStyleTransferLoss(ImageImageLoss):
    """Gram matrix style reconstruction loss based on intermediate VGG feature activations."""

    class VGGIntermediate(nn.Module):
        """VGG feature extractor capturing intermediate layer activations."""

        def __init__(self, requested: list[int] | None = None, backbone: str = "vgg16"):
            """Initialize VGG intermediate extractor.

            Args:
                requested: List of child layer indices to record activations for.
                backbone: VGG architecture ('vgg16' or 'vgg19').

            Raises:
                ValueError: If backbone is not 'vgg16' or 'vgg19'.
            """
            super().__init__()
            if requested is None:
                requested = []

            # register_buffer means that these will move to device automatically with the model
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            import torchvision.models as models
            self.intermediates: dict[int, torch.Tensor] = {}
            self.backbone = backbone
            if backbone == "vgg16":
                self.vgg = models.vgg16(pretrained=True).features.eval()
            elif backbone == "vgg19":
                self.vgg = models.vgg19(pretrained=True).features.eval()
            else:
                raise ValueError(f"Unsupported backbone: {backbone}. Choose 'vgg16' or 'vgg19'.")

            for i, m in enumerate(self.vgg.children()):
                if isinstance(m, nn.ReLU):
                    m.inplace = False
                if isinstance(m, nn.MaxPool2d):
                    self.vgg[i] = nn.AvgPool2d(2, 2)
                if i in requested:
                    def curry(idx):
                        def hook(module, input, output):
                            self.intermediates[idx] = output
                        return hook
                    m.register_forward_hook(curry(i))

        def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
            """Extract intermediate activations for normalized input.

            Args:
                x: Input tensor [N, C, H, W] in [0, 1].

            Returns:
                Dictionary mapping requested layer indices to activation tensors.
            """
            self.intermediates = {}
            self.vgg(self._normalize(x))
            return self.intermediates

        def _normalize(self, image: torch.Tensor) -> torch.Tensor:
            """Normalize image tensor by ImageNet mean and std."""
            return (image - self.mean) / self.std

    def __init__(
        self,
        reference_image: torch.Tensor | Image.Image | str,
        requested_names: list[str] = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
        comparison_height: int = 224,
        comparison_width: int = 224,
        device: str = "cuda",
        backbone: str = "vgg16",
    ):
        """Initialize VGG style transfer loss.

        Args:
            reference_image: Reference style image.
            requested_names: Layer names to compute Gram matrices for.
            comparison_height: Comparison image height (default 224).
            comparison_width: Comparison image width (default 224).
            device: PyTorch device.
            backbone: VGG network architecture ('vgg16' or 'vgg19').
        """
        super().__init__(reference_image, comparison_height, comparison_width, device)
        self.backbone = backbone
        self.requested_indices = self._get_requested_indices(requested_names, backbone)
        self.model = self.VGGIntermediate(requested=self.requested_indices, backbone=backbone).eval().to(self.device)

        with torch.no_grad():
            self.processed_target_image = (
                self.processed_target_image.unsqueeze(0)
                if self.processed_target_image.dim() == 3
                else self.processed_target_image
            )
            activations = self.model(self.processed_target_image)
            self.style_image_activations = [activations[i] for i in self.requested_indices]
            style_image_activation_feature_matrices = [
                self._construct_feature_matrix(activation) for activation in self.style_image_activations
            ]
            self.style_gram_matrices = [
                self._compute_gram_matrix(feature_matrix) for feature_matrix in style_image_activation_feature_matrices
            ]

    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
        """Calculate style transfer loss based on Gram matrix MSE."""
        activations = self.model(incoming_image)
        generated_image_style_activations = [activations[i] for i in self.requested_indices]
        generated_image_feature_matrices = [
            self._construct_feature_matrix(activation) for activation in generated_image_style_activations
        ]
        generated_image_gram_matrices = [
            self._compute_gram_matrix(feature_matrix) for feature_matrix in generated_image_feature_matrices
        ]
        total_style_loss = 0.0
        for i in range(len(self.style_gram_matrices)):
            b, c, h_w = generated_image_feature_matrices[i].size()
            # The division by c**2 is handled by using MSE, which already divides by the number of elements (channels) in the gram matrices.
            weight = 1.0 / (4.0 * (h_w ** 2))
            total_style_loss += weight * nn.functional.mse_loss(
                generated_image_gram_matrices[i], self.style_gram_matrices[i]
            )
        return total_style_loss

    def _compute_gram_matrix(self, activation_feature_matrix: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix (C x C) for a batch of flattened feature maps."""
        transposed = activation_feature_matrix.transpose(1, 2) # [B, H*W, C]
        return torch.bmm(activation_feature_matrix, transposed) # [B, C, C]

    def _construct_feature_matrix(self, activations: torch.Tensor) -> torch.Tensor:
        """Flatten spatial dimensions of activations to (Batch, Channels, Height * Width)."""
        size = activations.size() # [B, C, H, W]
        batch, channels, height, width = size[0], size[1], size[2], size[3]
        return activations.view(batch, channels, height * width)

    def _get_requested_indices(self, requested: list[str], backbone: str = "vgg16") -> list[int]:
        """Convert layer names to integer feature layer indices.

        Args:
            requested: List of VGG layer name strings.
            backbone: 'vgg16' or 'vgg19'.

        Returns:
            List of corresponding integer indices.

        Raises:
            ValueError: If backbone is unrecognized.
        """
        if backbone == "vgg16":
            vgg_names = [
                "conv1_1", "relu1_1", "conv1_2", "relu1_2", "maxpool1",
                "conv2_1", "relu2_1", "conv2_2", "relu2_2", "maxpool2",
                "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "maxpool3",
                "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "maxpool4",
                "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "maxpool5",
            ]
        elif backbone == "vgg19":
            vgg_names = [
                "conv1_1", "relu1_1", "conv1_2", "relu1_2", "maxpool1",
                "conv2_1", "relu2_1", "conv2_2", "relu2_2", "maxpool2",
                "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "conv3_4", "relu3_4", "maxpool3",
                "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "conv4_4", "relu4_4", "maxpool4",
                "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "conv5_4", "relu5_4", "maxpool5",
            ]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'vgg16' or 'vgg19'.")
        return [vgg_names.index(name) for name in requested]


class ImageEmbeddingSimilarityLoss(ImageImageLoss):
    """Loss measuring similarity between image representations in a learned feature space."""

    def __init__(
        self,
        reference_image: torch.Tensor | Image.Image | str,
        embedder_checkpoint: str,
        comparison_height: int = 224,
        comparison_width: int = 224,
        device: str = "cuda",
        model_name: str = "vit_b_32",
        mode: str = "cosine",
    ):
        """Initialize image embedding similarity loss.

        Args:
            reference_image: Reference target image.
            embedder_checkpoint: Path to fine-tuned embedder weights checkpoint.
            comparison_height: Comparison height.
            comparison_width: Comparison width.
            device: PyTorch device.
            model_name: Vision backbone architecture name.
            mode: Similarity mode ('cosine' or 'l2').
        """
        self._validate_embedding_similarity_mode(mode)
        self.mode = mode

        super().__init__(reference_image, comparison_height, comparison_width, device)
        self.embedder_checkpoint = embedder_checkpoint
        self.model_name = model_name
        self.image_embedder, _ = self._load_image_embedder(embedder_checkpoint, device, model_name)
        with torch.no_grad():
            self.processed_target_embedding = self.image_embedder.encode_image(self.processed_target_image)  # type: ignore
            if self.mode == "cosine":
                self.processed_target_embedding = (
                    self.processed_target_embedding / self.processed_target_embedding.norm(dim=1, keepdim=True)
                )

    def _load_image_embedder(self, checkpoint_path: str, device: str, model_name: str = "vit_b_32"):
        """Load vision embedder model from checkpoint."""
        from utils.model.model_utils import create_vision_only_model

        return create_vision_only_model(
            model_name=model_name,
            device=device,
            pretrained=False,
            fine_tune=checkpoint_path,
        )

    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
        """Compute embedding similarity loss against reference embedding."""
        incoming_embedding = self.image_embedder.encode_image(incoming_image)  # type: ignore
        return self._compute_embedding_similarity_loss(
            static_embedding=self.processed_target_embedding,
            dynamic_embedding=incoming_embedding,
            mode=self.mode,
            is_static_embedding_prenormalized=True,
        )

    def _validate_embedding_similarity_mode(self, mode: str) -> None:
        """Validate that the similarity mode is supported."""
        if mode not in ("cosine", "l2"):
            raise ValueError(f"Unsupported mode: {mode}. Choose 'cosine' or 'l2'.")

    def _compute_embedding_similarity_loss(
        self,
        static_embedding: torch.Tensor,
        dynamic_embedding: torch.Tensor,
        mode: str = "cosine",
        is_static_embedding_prenormalized: bool = True,
    ) -> torch.Tensor:
        """Compute cosine distance or L2 (MSE) distance between embeddings."""
        self._validate_embedding_similarity_mode(mode)
        if mode == "cosine":
            return compute_cosine_distance(
                static_embedding,
                dynamic_embedding,
                is_static_embedding_prenormalized=is_static_embedding_prenormalized,
            )
        return torch.nn.functional.mse_loss(dynamic_embedding, static_embedding)
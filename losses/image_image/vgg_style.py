"""VGG Gram matrix style transfer loss."""
from typing import override

import torch
from PIL import Image
from torch import nn
from torchvision import models

from losses.image_image.base import ImageImageLoss


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
        """Initialize VGG style transfer loss. Loss computed as described by Gatys et al., 2015.

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
                self._compute_gram_matrix(feature_matrix)
                for feature_matrix in style_image_activation_feature_matrices
            ]

    @override
    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
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
            weight = 1.0 / (4.0 * (h_w**2))
            total_style_loss += weight * nn.functional.mse_loss(
                generated_image_gram_matrices[i], self.style_gram_matrices[i]
            )
        return total_style_loss

    def _compute_gram_matrix(self, activation_feature_matrix: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix (C x C) for a batch of flattened feature maps."""
        transposed = activation_feature_matrix.transpose(1, 2)  # [B, H*W, C]
        return torch.bmm(activation_feature_matrix, transposed)  # [B, C, C]

    def _construct_feature_matrix(self, activations: torch.Tensor) -> torch.Tensor:
        """Flatten spatial dimensions of activations to (Batch, Channels, Height * Width)."""
        size = activations.size()  # [B, C, H, W]
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

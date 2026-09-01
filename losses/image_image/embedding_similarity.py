"""Image embedding similarity loss using fine-tuned vision models."""
from typing import override

import torch
from PIL import Image

from losses.image_image.base import ImageImageLoss
from losses.loss_utils import compute_cosine_distance


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

    @override
    def _loss_implementation(self, incoming_image: torch.Tensor) -> torch.Tensor:
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

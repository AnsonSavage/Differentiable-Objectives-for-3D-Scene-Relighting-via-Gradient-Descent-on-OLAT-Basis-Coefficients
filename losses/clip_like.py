"""CLIP-style loss functions for image relighting with text guidance. Classes are compatible with CLIP-style models that return images and text in a shared embedding space."""
from __future__ import annotations

from typing import Any, override

import torch

from losses.base import BaseLoss
from losses.loss_utils import compute_cosine_distance
from utils.image.preprocess_utils import preprocess_image_input


class CLIPCosineSimilarity(BaseLoss):
    """Loss based on cosine similarity between image and text CLIP embeddings."""

    def __init__(self, text: str, model: Any, tokenizer: Any, device: str | torch.device, preprocess: Any):
        """Initialize CLIP cosine similarity loss.

        Args:
            text: Target text prompt.
            model: Pretrained CLIP model with encode_text and encode_image.
            tokenizer: Tokenizer corresponding to the CLIP model.
            device: PyTorch device to run evaluation on.
            preprocess: CLIP preprocessing transform.
        """
        super().__init__()
        self.model = model
        self.text = text
        self.device = device
        self.preprocess = preprocess
        tokens = tokenizer([text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            self.text_features = text_features / text_features.norm(dim=1, keepdim=True)

    @override
    def forward(self, image) -> torch.Tensor:
        image = preprocess_image_input(image, preprocess=self.preprocess, device=self.device)
        image_features = self.model.encode_image(image)

        return compute_cosine_distance(self.text_features, image_features)

    @override
    def get_prompt_info(self) -> dict[str, str]:
        return {"clip_text_prompt": self.text}


class CLIPDirectionalCosineSimilarity(BaseLoss):
    """Directional CLIP loss aligning image edit vector with text edit vector.

    Implemented as described in equation 9 of DiffusionCLIP (https://arxiv.org/pdf/2110.02711).
    """

    def __init__(
        self,
        initial_text: str,
        target_text: str,
        initial_image: torch.Tensor,
        model: Any,
        tokenizer: Any,
        device: str | torch.device,
        preprocess: Any,
        always_prenormalize_vectors: bool = False,
    ):
        """Initialize directional CLIP loss.

        Args:
            initial_text: Text description of the initial image state.
            target_text: Text description of the desired target state.
            initial_image: Initial image tensor before relighting/editing.
            model: Pretrained CLIP model.
            tokenizer: Tokenizer corresponding to the CLIP model.
            device: PyTorch device.
            preprocess: CLIP preprocessing transform.
            always_prenormalize_vectors: Whether to normalize embedding vectors before computing difference directions.
        """
        super().__init__()
        self.model = model
        self.initial_text = initial_text
        self.target_text = target_text
        self.device = device
        self.preprocess = preprocess
        self.always_prenormalize_vectors = always_prenormalize_vectors

        # Precompute text features for both texts
        tokens_initial = tokenizer([initial_text]).to(device)
        tokens_target = tokenizer([target_text]).to(device)

        with torch.no_grad():
            text_features_initial = model.encode_text(tokens_initial)
            text_features_target = model.encode_text(tokens_target)

            self.initial_image_features = self._get_image_features(
                initial_image, normalize=self.always_prenormalize_vectors
            )

            if self.always_prenormalize_vectors:
                text_features_initial = text_features_initial / text_features_initial.norm(dim=1, keepdim=True)
                text_features_target = text_features_target / text_features_target.norm(dim=1, keepdim=True)

            self.text_direction = text_features_target - text_features_initial
            self.text_direction = self.text_direction / self.text_direction.norm(dim=1, keepdim=True)

    @override
    def forward(self, image) -> torch.Tensor:
        image_features = self._get_image_features(image, normalize=self.always_prenormalize_vectors)
        image_direction = image_features - self.initial_image_features
        image_direction = image_direction / image_direction.norm(dim=1, keepdim=True)

        cosine_similarity = (image_direction @ self.text_direction.T).squeeze()

        return 1 - cosine_similarity # TODO: this could probably delegate to compute_cosine_distance

    def _get_image_features(self, image, normalize: bool = False) -> torch.Tensor:
        """Embed an image with the CLIP vision encoder.

        Args:
            image: PIL Image or torch.Tensor.
            normalize: Whether to L2-normalize the output features.

        Returns:
            Image feature embedding tensor.
        """
        image = preprocess_image_input(image, preprocess=self.preprocess, device=self.device)
        image_features = self.model.encode_image(image)

        if normalize:
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return image_features

    @override
    def get_prompt_info(self) -> dict[str, str]:
        info = {
            "clip_initial_text_prompt": self.initial_text,
            "clip_target_text_prompt": self.target_text,
        }
        if hasattr(self, "always_prenormalize_vectors"):
            info["clip_always_prenormalize_vectors"] = "True" if self.always_prenormalize_vectors else "False"
        return info
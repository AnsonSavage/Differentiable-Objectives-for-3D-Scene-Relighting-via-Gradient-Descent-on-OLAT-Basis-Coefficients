"""Aesthetic-based losses for image lighting optimization."""
import os
from abc import abstractmethod
from urllib.request import urlretrieve

import clip
import torch
from PIL import Image
from torch import nn

from losses.base import BaseLoss
from utils.image.preprocess_utils import preprocess_image_tensor


class BaseAestheticScorer:
    """Abstract interface for aesthetic scoring models."""

    @abstractmethod
    def score(self, image) -> torch.Tensor:
        """Calculate aesthetic score for an image.

        Args:
            image: PIL Image or image tensor.

        Returns:
            Aesthetic score as a scalar tensor in [0, 1].
        """


class LAIONAestheticScorer(BaseAestheticScorer):
    """Aesthetic scorer using LAION's pretrained aesthetic predictor."""

    def __init__(self, device: str):
        """Initialize LAION aesthetic scorer.

        Args:
            device: PyTorch device to run the scorer on.
        """
        clip_model_name = "vit_b_32"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.to(device)
        self.aesthetic_model = self._get_aesthetic_model(clip_model_name)
        self.preprocess = preprocess
        for param in self.aesthetic_model.parameters():
            param.requires_grad = False
        self.aesthetic_model.to(device)
        self.device = device
        self.model = model

    def _get_aesthetic_model(self, clip_model: str) -> nn.Module:
        """Load or download the aesthetic linear head weights.

        Args:
            clip_model: Name of the CLIP architecture ("vit_b_32" or "vit_l_14").

        Returns:
            Loaded PyTorch linear model in eval mode.

        Raises:
            ValueError: If the specified clip_model is unsupported.
        """
        from config import DEFAULT_MODEL_WEIGHTS_DIR
        cache_folder = os.path.join(DEFAULT_MODEL_WEIGHTS_DIR, "aesthetic")
        path_to_model = os.path.join(cache_folder, f"sa_0_4_{clip_model}_linear.pth")
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = (
                "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_" + clip_model + "_linear.pth?raw=true"
            )
            urlretrieve(url_model, path_to_model)

        if clip_model == "vit_l_14":
            m = nn.Linear(768, 1)
        elif clip_model == "vit_b_32":
            m = nn.Linear(512, 1)
        else:
            raise ValueError(f"Unsupported CLIP model: {clip_model}")

        s = torch.load(path_to_model)
        m.load_state_dict(s)
        m.eval()
        return m

    def _preprocess_image(self, image) -> torch.Tensor:
        """Preprocess an input image tensor or PIL image for scoring.

        Args:
            image: PIL Image or torch.Tensor.

        Returns:
            Preprocessed 4D image tensor on the target device.

        Raises:
            TypeError: If image is neither a PIL Image nor a torch.Tensor.
        """
        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            image = preprocess_image_tensor(image)
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if image.device != self.device:
                image = image.to(self.device)
        else:
            raise TypeError(f"Expected PIL Image or torch Tensor, got {type(image)}")
        return image

    def score(self, image) -> torch.Tensor:
        """Calculate the normalized aesthetic score of an image.

        Args:
            image: PIL Image or torch.Tensor to score.

        Returns:
            Predicted aesthetic score scaled to [0, 1]. (Note that the score is not gauranteed to be in [0, 1] for all inputs, but is typically in that range for natural images.)
        """
        image = self._preprocess_image(image)
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # Ensure dtype matches the linear head (avoids Half vs Float mismatch under AMP)
        if isinstance(self.aesthetic_model, nn.Module):
            target_dtype = next(self.aesthetic_model.parameters()).dtype
            if image_features.dtype != target_dtype:
                image_features = image_features.to(dtype=target_dtype)
        predicted_score = self.aesthetic_model(image_features).squeeze()
        return predicted_score / 10.0  # Scorer was originally trained to predict values between 0 and 10 (though there is nothing in the model architecture that requires this range). We scale down to [0, 1] for convenience.


class BaseAestheticLoss(BaseLoss):
    """Base class for aesthetic losses using an aesthetic scorer."""

    def __init__(self, scorer: BaseAestheticScorer):
        """Initialize aesthetic loss.

        Args:
            scorer: Aesthetic scorer instance.
        """
        super().__init__()
        self.scorer = scorer

    def get_prompt_info(self) -> dict:
        """Get prompt and configuration information for logging.

        Returns:
            Dictionary containing aesthetic scorer metadata.
        """
        return {
            "aesthetic_scorer_type": type(self.scorer).__name__
        }


class AestheticLossWithTarget(BaseAestheticLoss):
    """Loss measuring absolute distance to a target aesthetic score."""

    def __init__(self, scorer: BaseAestheticScorer, target_score: float):
        """Initialize target-based aesthetic loss.

        Args:
            scorer: Aesthetic scorer instance.
            target_score: Desired target aesthetic score in [0, 1].
        """
        super().__init__(scorer)
        self.target_score = target_score

    def forward(self, image) -> torch.Tensor:
        """Calculate absolute difference between predicted and target scores.

        Args:
            image: Image tensor or PIL image.

        Returns:
            Absolute error loss scalar tensor.
        """
        predicted_score = self.scorer.score(image)
        return torch.abs(self.target_score - predicted_score)

    def get_prompt_info(self) -> dict:
        """Get prompt and configuration information for logging.

        Returns:
            Dictionary containing target score metadata.
        """
        info = super().get_prompt_info()
        info.update({
            "aesthetic_target_score": float(self.target_score),
        })
        return info


class AestheticLossMaximize(BaseAestheticLoss):
    """Loss maximizing aesthetic score (by minimizing negative aesthetic score)."""

    def forward(self, image) -> torch.Tensor:
        """Calculate negative aesthetic score.

        Args:
            image: Image tensor or PIL image.

        Returns:
            Negative predicted aesthetic score.
        """
        predicted_score = self.scorer.score(image)
        return -predicted_score

    def get_prompt_info(self) -> dict:
        """Get prompt and configuration information for logging.

        Returns:
            Dictionary indicating maximization mode.
        """
        info = super().get_prompt_info()
        info.update({
            "aesthetic_maximize": True,
        })
        return info
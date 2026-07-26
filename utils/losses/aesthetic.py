"""
Aesthetic-based losses for image optimization.
"""
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import expanduser
from urllib.request import urlretrieve
from abc import abstractmethod

from utils.losses.base import BaseLoss
from utils.image.preprocess_utils import preprocess_image_tensor

class BaseAestheticScorer:
    """Abstract scorer interface."""

    @abstractmethod
    def score(self, image):
        """Calculate aesthetic score for an image.
        
        Args:
            image: Image tensor or PIL image
            
        Returns:
            Aesthetic score as a scalar tensor
        """

class LAIONAestheticScorer(BaseAestheticScorer):
    """Scorer using LAION's aesthetic predictor."""
    
    def __init__(self, device):
        """Initialize LAION aesthetic scorer.
        
        Args:
            model: CLIP model to extract features
            device: Device to run the scorer on
            preprocess: CLIP's preprocess function
            clip_model: CLIP model version
        """
        import clip
        clip_model_name="vit_b_32"
        model, preprocess = clip.load("ViT-B/32", device=device) # TODO: It'd be cool if you could choose which CLIP model to use
        model.to(device)
        self.aesthetic_model = self._get_aesthetic_model(clip_model_name)
        self.preprocess = preprocess
        for param in self.aesthetic_model.parameters():
            param.requires_grad = False
        self.aesthetic_model.to(device)
        self.device = device
        self.model = model

    def _get_aesthetic_model(self, clip_model):
        """Load the aesthetic model.
        
        Args:
            clip_model: CLIP model version
            
        Returns:
            Loaded aesthetic model
        """
        home = expanduser("~")
        cache_folder = os.path.join(home, ".cache", "emb_reader")  # TODO: PATH_UPDATE aesthetic model cache directory
        path_to_model = os.path.join(cache_folder, f"sa_0_4_{clip_model}_linear.pth")
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = (
                "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
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

    def _preprocess_image(self, image):
        """Preprocess image for aesthetic scoring."""
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

    def score(self, image):
        """Calculate the aesthetic score of an image."""
        image = self._preprocess_image(image)
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # Ensure dtype matches the linear head (avoids Half vs Float mismatch under AMP)
        if isinstance(self.aesthetic_model, nn.Module):
            target_dtype = next(self.aesthetic_model.parameters()).dtype
            if image_features.dtype != target_dtype:
                image_features = image_features.to(dtype=target_dtype)
        predicted_score = self.aesthetic_model(image_features).squeeze()
        return predicted_score / 10.0 # Originally predicted score is in [0,10], we scale to [0,1]

class BaseAestheticLoss(BaseLoss):
    """Base class for aesthetic losses using a scorer."""
    
    def __init__(self, scorer: BaseAestheticScorer):
        """Initialize aesthetic loss.
        
        Args:
            scorer: Aesthetic scorer instance
        """
        super().__init__()
        self.scorer = scorer
    
    def get_prompt_info(self):
        """Get prompt information for aesthetic loss."""
        return {
            "aesthetic_scorer_type": type(self.scorer).__name__
        }

class AestheticLossWithTarget(BaseAestheticLoss):
    """Loss: |target_score - scorer.score(image)|"""
    
    def __init__(self, scorer: BaseAestheticScorer, target_score: float):
        """Initialize target-based aesthetic loss.
        
        Args:
            scorer: Aesthetic scorer instance
            target_score: Target aesthetic score
        """
        super().__init__(scorer)
        self.target_score = target_score

    def forward(self, image):
        """Calculate absolute difference from target score."""
        predicted_score = self.scorer.score(image)
        return torch.abs(self.target_score - predicted_score)
    
    def get_prompt_info(self):
        """Get prompt information for target-based aesthetic loss."""
        info = super().get_prompt_info()
        info.update({
            "aesthetic_target_score": float(self.target_score),
        })
        return info

class AestheticLossMaximize(BaseAestheticLoss):
    """Loss: -scorer.score(image) (maximize score)"""
    
    def forward(self, image):
        """Calculate negative score to maximize aesthetic score."""
        predicted_score = self.scorer.score(image)
        return -predicted_score
    
    def get_prompt_info(self):
        """Get prompt information for maximize aesthetic loss."""
        info = super().get_prompt_info()
        info.update({
            "aesthetic_maximize": True,
        })
        return info
"""
CLIP-based loss functions for image optimization.
"""
from __future__ import annotations
import torch
from PIL import Image
from utils.losses.base import BaseLoss
from utils.image.preprocess_utils import preprocess_image_input

class CLIPCosineSimilarity(BaseLoss):
    """Loss based on CLIP similarity between image and text."""
    
    def __init__(self, text: str, model, tokenizer, device, preprocess):
        super(CLIPCosineSimilarity, self).__init__()
        self.model = model
        self.text = text
        self.device = device
        self.preprocess = preprocess
        tokens = tokenizer([text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            self.text_features = text_features / text_features.norm(dim=1, keepdim=True)

    def forward(self, image): # TODO: could reduce code duplication by relying on compute_cosine_distance in #loss_utils.py
        """
        Process both PIL Images and torch Tensors.
        
        Args:
            image: Either a PIL Image or a pre-processed torch Tensor
        
        Returns:
            torch.Tensor: 1 - cosine_similarity between image and text features
        """
        image = preprocess_image_input(image, preprocess=self.preprocess, device=self.device)
            
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        cosine_similarity = image_features @ self.text_features.T

        # The more similar they are, the closer cosine_similarity will be to 1,
        # and the closer the loss will be to 0
        return 1 - cosine_similarity
    
    def get_prompt_info(self):
        """Get prompt information for this loss."""
        return {"clip_text_prompt": self.text}

class CLIPDirectionalCosineSimilarity(BaseLoss):
    """Directional loss based on CLIP embeddings.
    
    Implemented as described in equation 9 of https://arxiv.org/pdf/2110.02711
    """
    def __init__(self, initial_text: str, target_text: str, initial_image: torch.Tensor, 
                 model, tokenizer, device, preprocess, always_prenormalize_vectors=False): # I'm not actually sure how much it matters if you normalize the vectors or not, because we only measure cosine similarity anyway. Oh wait, it actually does matter, because the difference vector will have a different direction depending on whether the inputs were normalized or not.
        super(CLIPDirectionalCosineSimilarity, self).__init__()
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
    
    def forward(self, image):
        """
        Process both PIL Images and torch Tensors.
        
        Args:
            image: Either a PIL Image or a pre-processed torch Tensor
        
        Returns:
            torch.Tensor: Loss value based on cosine similarity between image and text direction
        """
        image_features = self._get_image_features(image, normalize=self.always_prenormalize_vectors)
        image_direction = image_features - self.initial_image_features
        image_direction = image_direction / image_direction.norm(dim=1, keepdim=True)

        cosine_similarity = (image_direction @ self.text_direction.T).squeeze() # (N, D) @ (D, 1) -> (N, 1) -> squeeze to (N,)

        # The more similar they are, the closer cosine_similarity will be to 1,
        # and the closer the loss will be to 0
        return 1 - cosine_similarity
        
    def _get_image_features(self, image, normalize=False):
        # TODO: you need to apply tonemapping and color space conversions. :)
        # Welp, I hope that the images in your thesis proposal aren't too wrong :)
        """Embed an image to get its feature vector."""
        image = preprocess_image_input(image, preprocess=self.preprocess, device=self.device)
        
        image_features = self.model.encode_image(image)

        if normalize:
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        return image_features
    
    def get_prompt_info(self):
        """Get prompt information for this directional loss."""
        info = {
            "clip_initial_text_prompt": self.initial_text,
            "clip_target_text_prompt": self.target_text,
        }
        if hasattr(self, "always_prenormalize_vectors"):
            info["clip_always_prenormalize_vectors"] = bool(self.always_prenormalize_vectors)
        return info
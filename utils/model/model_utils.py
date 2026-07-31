import os
import time
from collections.abc import Sequence
from typing import Any

import open_clip
import torch
from torchvision import transforms
from torchvision.models import (
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
    vit_b_16,
    vit_b_32,
    vit_l_16,
    vit_l_32,
)

from config import DEFAULT_MODEL_WEIGHTS_DIR

# Directory for cached fine-tuned weights
MODEL_WEIGHTS_DIR = DEFAULT_MODEL_WEIGHTS_DIR
HF_FINE_TUNED_REPO = "AnsonSavage/FineTunedOpenCLIPModelsForRelightingLossEvaluation"


def get_model_weights_path(filename_or_path: str) -> str:
    """Get the path to a model weights file.

    If the file exists locally (either as a direct path or inside MODEL_WEIGHTS_DIR),
    returns it immediately. Otherwise, downloads it automatically from Hugging Face
    (AnsonSavage/FineTunedOpenCLIPModelsForRelightingLossEvaluation) and caches it.
    """
    if os.path.exists(filename_or_path):
        return filename_or_path

    local_path = os.path.join(MODEL_WEIGHTS_DIR, filename_or_path)
    if os.path.exists(local_path):
        return local_path

    print(f"Model weights '{filename_or_path}' not found locally. Downloading from Hugging Face ({HF_FINE_TUNED_REPO})...")
    os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=HF_FINE_TUNED_REPO,
        filename=filename_or_path,
        local_dir=MODEL_WEIGHTS_DIR,
    )
def infer_head_layers(state_dict: dict, prefix: str) -> list[int] | None:
    """Infer MLP projection head layer dimensions from a saved state_dict."""
    proj_keys = [k for k in state_dict if k.startswith(f"{prefix}.")]
    if not proj_keys:
        return None
    layer_weights = {}
    for k in proj_keys:
        if "mlp." in k and "weight" in k:
            parts = k.split(".")
            layer_idx = int(parts[2])
            layer_weights[layer_idx] = state_dict[k].shape

    if layer_weights:
        sorted_layers = sorted(layer_weights.items())
        layers = [sorted_layers[0][1][1]]
        for _, shape in sorted_layers:
            layers.append(shape[0])
        return layers
    return None


def create_vision_only_model(
    model_name: str = "vit_b_16",
    device: str = "cuda",
    pretrained: bool = False,
    image_head_layers: Sequence[int] | None = None,
    projection_activation: type[torch.nn.Module] = torch.nn.ReLU,
    fine_tune: str | None = None,
) -> tuple[torch.nn.Module, Any]:
    """Create a vision-only ViT model (no text encoder or tokenizer).

    Args:
        model_name: Architecture name from _VIT_REGISTRY (e.g., "vit_b_16").
        device: PyTorch device.
        pretrained: If True, load ImageNet pretrained weights.
        image_head_layers: Optional projection head layer sizes (e.g., [768, 256, 64]).
        projection_activation: Activation function for projection head.
        fine_tune: Optional fine-tuned weights file or checkpoint path.

    Returns:
        (model, preprocess_transform)
    """
    loading_time_start = time.time()
    print(f"Loading vision-only model {model_name} (pretrained={pretrained})...")

    if model_name not in _VIT_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(_VIT_REGISTRY.keys())}")

    factory_fn, weights_enum, embed_dim, image_size = _VIT_REGISTRY[model_name]

    state_dict = None
    if fine_tune is not None:
        weights_path = get_model_weights_path(fine_tune)
        state_dict = torch.load(weights_path, map_location="cpu")
        if image_head_layers is None:
            image_head_layers = infer_head_layers(state_dict, "image_projection")

    if pretrained:
        weights_path = os.path.join(MODEL_WEIGHTS_DIR, f"{model_name}_imagenet.pt")
        if os.path.exists(weights_path):
            print(f"Loading {model_name} weights from {weights_path}...")
            vit = factory_fn(weights=None)
            vit.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            print(f"Local weights not found at {weights_path}, downloading from PyTorch hub...")
            vit = factory_fn(weights=weights_enum)
        preprocess = weights_enum.transforms()
    else:
        vit = factory_fn(weights=None)
        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    image_head = None
    if image_head_layers:
        if image_head_layers[0] != embed_dim:
            image_head_layers = [embed_dim] + list(image_head_layers)
        image_head = _ProjectionHead(image_head_layers, activation=projection_activation)

    model = _VisionOnlyModel(vit, embed_dim, image_projection=image_head)

    if state_dict is not None:
        print(f"Loading fine-tuned weights from {fine_tune}...")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    print(f"Vision-only model loaded in {time.time() - loading_time_start:.2f} seconds")
    return model, preprocess


def create_clip_model_and_tokenizer(
    model_name: str = "ViT-B-16-SigLIP-512",
    device: str = "cuda",
    pretrained: str | None = "webli",
    image_head_layers: Sequence[int] | None = None,
    text_head_layers: Sequence[int] | None = None,
    projection_activation: type[torch.nn.Module] = torch.nn.ReLU,
    fine_tune: str | None = None,
) -> tuple[torch.nn.Module, Any, Any]:
    """Create a combined vision and text OpenCLIP model with tokenizer.

    Args:
        model_name: Architecture name (e.g., "ViT-B-16-SigLIP-512").
        device: PyTorch device.
        pretrained: OpenCLIP pretrained dataset name (e.g., "webli").
        image_head_layers: Optional image projection head layer sizes (e.g., [768, 256, 64]).
        text_head_layers: Optional text projection head layer sizes.
        projection_activation: Activation function for projection heads.
        fine_tune: Optional fine-tuned weights file (e.g., "siglip_blend-training-data_64-output-dim.pt").
            If not found locally, automatically downloads it from Hugging Face.

    Returns:
        (model, tokenizer, preprocess_transform)
    """
    loading_time_start = time.time()
    print(f"Loading base CLIP model {model_name}...")

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)

    if fine_tune is not None:
        weights_path = get_model_weights_path(fine_tune)
        state_dict = torch.load(weights_path, map_location="cpu")

        if image_head_layers is None:
            image_head_layers = infer_head_layers(state_dict, "image_projection")
        if text_head_layers is None:
            text_head_layers = infer_head_layers(state_dict, "text_projection")

    model.to(device)
    model = _wrap_model_with_projection_heads(
        model,
        image_head_layers=image_head_layers,
        text_head_layers=text_head_layers,
        activation=projection_activation,
    )

    if fine_tune is not None:
        print(f"Loading fine-tuned weights from {weights_path}...")
        model.load_state_dict(state_dict)

    model.to(device)
    print(f"CLIP model loaded in {time.time() - loading_time_start:.2f} seconds")
    return model, tokenizer, preprocess


class _ProjectionHead(torch.nn.Module):
    """An MLP projection head for contrastive learning with len(layer_sizes) - 1 layers."""

    def __init__(self, layer_sizes: Sequence[int], activation: type[torch.nn.Module] = torch.nn.ReLU):
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("ProjectionHead requires at least two layer sizes (input and output).")
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # don't apply activation after last layer
                layers.append(activation())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _CLIPWithProjectionHeads(torch.nn.Module):
    """Wraps an OpenCLIP model with optional image/text projection heads."""

    def __init__(
        self,
        clip_model: torch.nn.Module,
        image_projection: torch.nn.Module | None = None,
        text_projection: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.clip_model: Any = clip_model
        self.image_projection = image_projection
        self.text_projection = text_projection

    def forward(self, *args, **kwargs):
        return self.clip_model(*args, **kwargs)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_image(images)
        if self.image_projection is not None:
            features = self.image_projection(features)
        return features

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_text(tokens)
        if self.text_projection is not None:
            features = self.text_projection(features)
        return features

    def __getattr__(self, name: str):
        # Delegate attribute access to the wrapped CLIP model for compatibility.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.clip_model, name)


# Registry mapping model names to (factory_fn, weights_enum, embed_dim, image_size)
_VIT_REGISTRY = {
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1, 768, 224),
    "vit_b_32": (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1, 768, 224),
    "vit_l_16": (vit_l_16, ViT_L_16_Weights.IMAGENET1K_V1, 1024, 224),
    "vit_l_32": (vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1, 1024, 224),
}


class _VisionOnlyModel(torch.nn.Module):
    """A vision-only model wrapper using PyTorch's ViT."""

    def __init__(
        self,
        vit_model: torch.nn.Module,
        embed_dim: int,
        image_projection: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.vit_model = vit_model
        self.embed_dim = embed_dim
        self.image_projection = image_projection
        self.vit_model.heads = torch.nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image(images)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.vit_model(images)
        if self.image_projection is not None:
            features = self.image_projection(features)
        return features

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("VisionOnlyModel does not support text encoding.")


def _wrap_model_with_projection_heads(
    model: torch.nn.Module,
    image_head_layers: Sequence[int] | None = None,
    text_head_layers: Sequence[int] | None = None,
    activation: type[torch.nn.Module] = torch.nn.ReLU,
) -> _CLIPWithProjectionHeads:
    if image_head_layers is not None and text_head_layers is not None:
        assert image_head_layers[-1] == text_head_layers[-1], "Image and text projection heads must have the same output dimension."

    image_head = (
        _ProjectionHead(image_head_layers, activation=activation)
        if image_head_layers
        else None
    )
    text_head = (
        _ProjectionHead(text_head_layers, activation=activation)
        if text_head_layers
        else None
    )

    return _CLIPWithProjectionHeads(model, image_projection=image_head, text_projection=text_head)

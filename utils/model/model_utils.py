import os
import time
from collections.abc import Sequence
from typing import Any

import open_clip
import torch
from peft import get_peft_model
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

# Path to pre-downloaded model weights
MODEL_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "original_model_weights")


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


def _create_vision_only_model(
    model_name: str = "vit_b_16",
    pretrained: bool = False,
    image_head_layers: Sequence[int] | None = None,
    projection_activation: type[torch.nn.Module] = torch.nn.ReLU,
):
    """Create a vision-only ViT model with optional projection head."""
    if model_name not in _VIT_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(_VIT_REGISTRY.keys())}")

    factory_fn, weights_enum, embed_dim, image_size = _VIT_REGISTRY[model_name]

    if pretrained:
        weights_path = os.path.join(MODEL_WEIGHTS_DIR, f"{model_name}_imagenet.pt")
        if os.path.exists(weights_path):
            print(f"Loading {model_name} weights from {weights_path}...")
            vit = factory_fn(weights=None)
            vit.load_state_dict(torch.load(weights_path, map_location='cpu'))
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
    return model, preprocess


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


def create_model_and_tokenizer(
    model_name: str,
    device,
    pretrained: str | None = None,
    vision_only: bool = False,
    image_head_layers: Sequence[int] | None = None,
    text_head_layers: Sequence[int] | None = None,
    lora_config: Any | None = None,
    projection_activation: type[torch.nn.Module] = torch.nn.ReLU,
):
    """Load a model, optional tokenizer, and preprocessing transforms.

    Args:
        model_name: Model architecture name.
            - For OpenCLIP: e.g., "ViT-B-16-SigLIP-512"
            - For vision-only: e.g., "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"
        device: Device to load the model on.
        pretrained: Pretrained weights to load.
            - For OpenCLIP: dataset name like "webli", "laion2b_s34b_b79k", etc.
            - For vision-only: "imagenet" to use ImageNet weights, or None for random init.
        vision_only: If True, use a PyTorch ViT (no text encoder).
            Returns (VisionOnlyModel, None, preprocess).
        image_head_layers: Optional projection head layer sizes (e.g., [768, 256, 64]).
        text_head_layers: Optional text projection head (ignored for vision_only=True).
        lora_config: Optional LoRA configuration (only for OpenCLIP models).
        projection_activation: Activation function for projection heads.

    Returns:
        (model, tokenizer, preprocess) - tokenizer is None for vision_only=True
    """
    loading_time_start = time.time()

    if vision_only:
        use_pretrained = pretrained is not None and pretrained.lower() == "imagenet"
        print(f"Loading vision-only model {model_name} (pretrained={use_pretrained})...")

        model, preprocess = _create_vision_only_model(
            model_name=model_name,
            pretrained=use_pretrained,
            image_head_layers=image_head_layers,
            projection_activation=projection_activation,
        )
        model.to(device)

        print(f"Model loaded in {time.time() - loading_time_start:.2f} seconds")
        return model, None, preprocess

    # OpenCLIP model path
    print(f"Loading model {model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

    if lora_config is not None:
        print("Applying LoRA configuration to model...")
        model = get_peft_model(model, lora_config)  # type: ignore

    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    model = _wrap_model_with_projection_heads(
        model,
        image_head_layers=image_head_layers,
        text_head_layers=text_head_layers,
        activation=projection_activation,
    )

    model.to(device)
    print(f"Model loaded in {time.time() - loading_time_start:.2f} seconds")
    return model, tokenizer, preprocess

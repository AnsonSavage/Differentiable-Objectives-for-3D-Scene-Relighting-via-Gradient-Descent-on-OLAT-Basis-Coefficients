import sys
import os
import torch
import torch.nn.functional as F

def load_image_embedder(checkpoint_path, device, model_name='vit_b_32'):
    """Load the image embedder model from checkpoint."""
    
    # Add the fine_tuning codebase to the path for model loading utilities
    # TODO: PATH_UPDATE fine-tuning codebase path
    FINE_TUNING_PATH = ""
    if FINE_TUNING_PATH not in sys.path:
        sys.path.insert(0, FINE_TUNING_PATH)
    
    from utils.model.model_utils import create_model_and_tokenizer
    
    # Infer model configuration from checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Detect projection head configuration
    image_head_layers = None
    proj_keys = [k for k in state_dict.keys() if k.startswith('image_projection.')]
    if proj_keys:
        layer_weights = {}
        for k in proj_keys:
            if 'mlp.' in k and 'weight' in k:
                parts = k.split('.')
                layer_idx = int(parts[2])
                layer_weights[layer_idx] = state_dict[k].shape
        
        if layer_weights:
            sorted_layers = sorted(layer_weights.items())
            image_head_layers = [sorted_layers[0][1][1]]  # Input dim
            for _, shape in sorted_layers:
                image_head_layers.append(shape[0])  # Output dim
    
    # Create the model
    model, _, preprocess = create_model_and_tokenizer(
        model_name=model_name,
        device=device,
        pretrained=None,
        vision_only=True, # TODO: you would have to change this if you try to load up a fine-tuned CLIP-like model
        image_head_layers=image_head_layers,
    )
    
    # Load weights
    model.load_state_dict(state_dict, strict=False) # strict=False to allow loading models with different keys (e.g. missing text encoder)
    model.to(device)
    model.eval()
    
    return model, preprocess

def compute_cosine_distance(static_embedding, dynamic_embedding, is_static_embedding_prenormalized=True):
    if not is_static_embedding_prenormalized:
        static_embedding = static_embedding / static_embedding.norm(dim=-1, keepdim=True)
    single_reference_embedding = static_embedding.shape[0] == 1

    dynamic_embedding = dynamic_embedding / dynamic_embedding.norm(dim=-1, keepdim=True)
    cosine_similarity = (static_embedding @ dynamic_embedding.T).squeeze()

    if cosine_similarity.dim() <= 1:
        assert single_reference_embedding, "Cosine similarity output is 1D but static embedding has more than one reference."
        if cosine_similarity.dim() == 0:
            cosine_similarity = cosine_similarity.unsqueeze(0)
        return 1 - cosine_similarity
    assert cosine_similarity.dim() == 2, "Cosine similarity output should be 2D when there are multiple reference embeddings."
    return 1 - cosine_similarity.diagonal()

def validate_embedding_similarity_mode(mode: str):
    if mode not in ('cosine', 'l2'):
        raise ValueError(f"Unsupported mode: {mode}. Choose 'cosine' or 'l2'.")

def compute_embedding_similarity_loss(
    static_embedding,
    dynamic_embedding,
    mode: str = 'cosine',
    is_static_embedding_prenormalized: bool = True,
):
    validate_embedding_similarity_mode(mode)
    if mode == 'cosine':
        return compute_cosine_distance(
            static_embedding,
            dynamic_embedding,
            is_static_embedding_prenormalized=is_static_embedding_prenormalized,
        )
    return F.mse_loss(dynamic_embedding, static_embedding)
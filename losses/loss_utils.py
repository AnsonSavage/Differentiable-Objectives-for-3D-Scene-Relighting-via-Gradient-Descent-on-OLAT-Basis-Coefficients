"""Utility functions for computing distances and metrics in loss functions."""
import torch


def compute_cosine_distance(
    static_embedding: torch.Tensor,
    dynamic_embedding: torch.Tensor,
    is_static_embedding_prenormalized: bool = True,
) -> torch.Tensor:
    """Compute the cosine distance (1 - cosine similarity) between sets of embeddings.

    Args:
        static_embedding: Reference embeddings tensor of shape [num_references, embedding_dim].
        dynamic_embedding: Evaluated embeddings tensor of shape [num_dynamics, embedding_dim].
        is_static_embedding_prenormalized: Whether static_embedding is already unit-normalized.

    Returns:
        Tensor of shape [num_dynamics] representing cosine distances.
    """
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

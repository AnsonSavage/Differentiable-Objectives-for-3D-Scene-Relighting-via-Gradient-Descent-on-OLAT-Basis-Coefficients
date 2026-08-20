
def compute_cosine_distance(static_embedding, dynamic_embedding, is_static_embedding_prenormalized=True):
    """ Compute the cosine distance between two sets of embeddings.
    
    Args:
        static_embedding: A tensor of shape [num_references, embedding_dim]
        dynamic_embedding: A tensor of shape [num_dynamics, embedding_dim]
        is_static_embedding_prenormalized: Whether the static embedding is already normalized

    Returns:
        A tensor of shape [num_dynamics] representing the cosine distances
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

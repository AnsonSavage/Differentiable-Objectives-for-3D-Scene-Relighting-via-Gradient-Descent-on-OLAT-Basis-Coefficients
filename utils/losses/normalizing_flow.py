"""
Normalizing flow-based loss for image optimization using learned lighting distributions.
"""
from abc import abstractmethod, ABC
import os
import torch
import torch.nn.functional as F
from utils.losses.base import BaseLoss
from utils.image.preprocess_utils import preprocess_image_tensor
from utils.losses.loss_utils import load_image_embedder, compute_embedding_similarity_loss, validate_embedding_similarity_mode

class FlowLoader(ABC):
    """Abstract base class for loading normalizing flow models."""
    def __init__(self, checkpoint_path: str, device: str, dimensionality: int, num_layers: int, use_batch_norm: bool = False):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dimensionality = dimensionality
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm

    @abstractmethod
    def load_base_distribution(self):
        pass

    @abstractmethod
    def load_transform(self):
        pass
    
    def load(self):
        """Load the normalizing flow model from checkpoint."""
        from nflows.flows.base import Flow
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.permutations import ReversePermutation
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Recreate the flow architecture
        base_dist = self.load_base_distribution()
        
        transforms = []
        for _ in range(self.num_layers):
            transforms.append(ReversePermutation(features=self.dimensionality))
            transforms.append(self.load_transform())
        
        transform = CompositeTransform(transforms)
        flow = Flow(transform, base_dist)
        
        # Load weights
        flow.load_state_dict(checkpoint['model_state_dict'])
        flow.to(self.device)
        flow.eval()
        
        return flow

class UnconditionalFlowLoader(FlowLoader):
    """Loader for unconditional normalizing flow models."""
    def load_base_distribution(self):
        from nflows.distributions.normal import StandardNormal
        return StandardNormal(shape=[self.dimensionality])
    
    def load_transform(self):
        from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
        return MaskedAffineAutoregressiveTransform(
            features=self.dimensionality,
            hidden_features=self.dimensionality * 2,
            num_blocks=2,
            use_batch_norm=self.use_batch_norm
        )
class ConditionalFlowLoader(FlowLoader):
    """Loader for conditional normalizing flow models."""
    def __init__(self, checkpoint_path: str, device: str, dimensionality: int, num_layers: int, condition_dim: int, use_batch_norm: bool = False):
        super().__init__(checkpoint_path, device, dimensionality, num_layers, use_batch_norm)
        self.condition_dim = condition_dim

    def load_base_distribution(self):
        from nflows.distributions.normal import ConditionalDiagonalNormal
        return ConditionalDiagonalNormal(
            shape=[self.dimensionality],
            context_encoder=torch.nn.Linear(self.condition_dim, self.dimensionality * 2)
        )
    
    def load_transform(self):
        from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
        return MaskedAffineAutoregressiveTransform(
            features=self.dimensionality,
            hidden_features=self.dimensionality * 2,
            context_features=self.condition_dim,
            num_blocks=2,
            use_batch_norm=self.use_batch_norm
        )


class NormalizingFlowLossBase(BaseLoss):
    """Shared utilities for normalizing-flow-based image embedding losses."""
    
    def __init__(
        self,
        image_embedder,
        flow_model,
        device='cuda',
        preprocess=None,
        embedder_checkpoint=None, # Only used for reporting
        flow_checkpoint=None,
        conditional_embeddings: torch.Tensor | None = None,
    ):
        super().__init__()
        self.device = device
        self.image_embedder = image_embedder
        self.flow_model = flow_model
        self.preprocess = preprocess
        self.embedder_checkpoint = embedder_checkpoint
        self.flow_checkpoint = flow_checkpoint
        self.conditional_embeddings = conditional_embeddings
        
        self.image_embedder.eval()
        for param in self.image_embedder.parameters():
            param.requires_grad = False

        self.flow_model.eval()
        for param in self.flow_model.parameters():
            param.requires_grad = False

    def _encode_image_embedding(self, image):
        image = preprocess_image_tensor(image, preprocess=self.preprocess)
        embedding = self.image_embedder.encode_image(image)
        # TODO: Someday you could add some assertions that ensure that the embedding dimensionality is correct for the two models
        embedding = embedding.to(dtype=torch.float32)
        return embedding # NOTE: Returns an unnormalized embedding. It's up to the caller to decide whether to normalize it or not.

    def _build_context(self, batch_size: int, device, normalize_context=True):
        if self.conditional_embeddings is None:
            return None
        context = self.conditional_embeddings.expand(batch_size, -1)
        context = context.to(device=device, dtype=torch.float32)
        if normalize_context:
            context = F.normalize(context, dim=-1)
        return context

    def get_prompt_info(self):
        return {
            "loss_type": self.__class__.__name__,
            "embedder_checkpoint": self.embedder_checkpoint,
            "flow_checkpoint": self.flow_checkpoint,
        }


class MaxLogLikelihoodNormalizingFlowLoss(NormalizingFlowLossBase):
    """Loss based on normalizing flow likelihood of image embeddings.

    This loss measures how likely an image's embedding is under a learned
    distribution of natural image lighting. Images with more natural lighting
    will have higher likelihood (lower loss).
    """

    def forward(self, image):
        """Calculate normalizing flow loss for the given image.

        Args:
            image: Image tensor [C, H, W] or PIL Image

        Returns:
            Loss value (negative log likelihood)
        """
        embedding = self._encode_image_embedding(image)

        context = self._build_context(batch_size=embedding.shape[0], device=embedding.device)
        
        # Calculate log probability under the flow
        log_prob = self.flow_model.log_prob(embedding, context=context)
        
        # Return negative log likelihood as loss
        # (Higher likelihood = more natural lighting [hopefully]= lower loss)
        return -log_prob
    
class SampleNormalizingFlowLoss(NormalizingFlowLossBase):
    """Rather than maximizing the log likelihood as in the MaxLogLikelihoodNormalizingFlowLoss, this loss samples an embedding from the trained flow and then tries to minimize the distance between the image embedding and the sampled embedding."""
    def __init__(
        self,
        image_embedder,
        flow_model,
        device='cuda',
        preprocess=None,
        embedder_checkpoint=None, # Only used for reporting
        flow_checkpoint=None,
        conditional_embeddings: torch.Tensor | None = None,
        mode: str = 'cosine',
        num_distinct_samples=1,
        normalize_sampled_embedding=True,
        normalize_context=True
    ):
        super().__init__(
            image_embedder=image_embedder,
            flow_model=flow_model,
            device=device,
            preprocess=preprocess,
            embedder_checkpoint=embedder_checkpoint,
            flow_checkpoint=flow_checkpoint,
            conditional_embeddings=conditional_embeddings,
        )
        validate_embedding_similarity_mode(mode)
        self.mode = mode
        self.normalize_sampled_embedding = normalize_sampled_embedding

        # Sample the vectors that we will optimize towards
        context = self._build_context(batch_size=num_distinct_samples, device=device, normalize_context=normalize_context) # NOTE: you'd need to refactor this if you wanted to pass a separate embedding for each image in the batch; right now it duplicates the context for each unique sample
        self.sampled_embedding = self._sample_flow_embedding(batch_size=num_distinct_samples, context=context)

    def _sample_flow_embedding(self, batch_size: int, context):
        sampled_embedding = self.flow_model.sample(batch_size, context=context) # Sample a separate embedding for each image in the batch

        # What comes back is of shape [1, num_samples, embedding_dim]. We should be able to squeeze the first dimension
        sampled_embedding = sampled_embedding.squeeze(0)

        assert sampled_embedding.dim() == 2, f"Expected sampled embedding to be 2D after squeezing, but got shape {sampled_embedding.shape}"
        assert sampled_embedding.shape[0] == batch_size, f"Expected sampled embedding batch size ({sampled_embedding.shape[0]}) to match image batch size ({batch_size})"

        sampled_embedding = sampled_embedding.to(dtype=torch.float32)
        return F.normalize(sampled_embedding, dim=-1) if self.normalize_sampled_embedding else sampled_embedding

    def forward(self, image):
        image_embedding = self._encode_image_embedding(image)

        return compute_embedding_similarity_loss(
            static_embedding=self.sampled_embedding,
            dynamic_embedding=image_embedding,
            mode=self.mode,
            is_static_embedding_prenormalized=True,
        )

    def get_prompt_info(self):
        info = super().get_prompt_info()
        info["mode"] = self.mode
        return info


def create_normalizing_flow_loss(
    embedder_checkpoint=None,
    flow_checkpoint=None,
    device='cuda',
    model_name='vit_b_32',
    flow_dimensionality=None, # If None, inferred from embedder
    flow_num_layers=15,
    conditional_embeddings: torch.Tensor | None = None,
    use_batch_norm: bool = False,
    loss_variant: str = 'max_log_likelihood',
    sample_mode: str = 'cosine',
    num_distinct_samples: int = 1,
    normalize_sampled_embedding: bool = True,
    normalize_context: bool = True,
) -> NormalizingFlowLossBase:
    """Helper function to create a normalizing-flow-based loss with default or provided checkpoints.

    Args:
        loss_variant: Either 'max_log_likelihood' or 'sample'.
        sample_mode: Similarity mode for sample loss ('cosine' or 'l2'). Only applies if loss_variant is 'sample'.
        num_distinct_samples: Number of sampled target embeddings for sample loss.
        normalize_sampled_embedding: Whether to normalize sampled embeddings for sample loss.
        normalize_context: Whether to normalize context embeddings when sampling.
    """
    if loss_variant not in ('max_log_likelihood', 'sample'):
        raise ValueError("loss_variant must be either 'max_log_likelihood' or 'sample'.")
    
    # Set default checkpoint paths
    if embedder_checkpoint is None:
        embedder_checkpoint = ""  # TODO: PATH_UPDATE image embedder checkpoint
    
    if flow_checkpoint is None:
        flow_checkpoint = ""  # TODO: PATH_UPDATE normalizing flow checkpoint

    if not embedder_checkpoint:
        raise ValueError("embedder_checkpoint must be provided. TODO: PATH_UPDATE image embedder checkpoint")

    if not flow_checkpoint:
        raise ValueError("flow_checkpoint must be provided. TODO: PATH_UPDATE normalizing flow checkpoint")
        
    print(f"Loading image embedder from {embedder_checkpoint}")
    image_embedder, preprocess = load_image_embedder(embedder_checkpoint, device, model_name)
    
    # Infer dimensionality if not provided
    if flow_dimensionality is None:
        # Run a dummy image to get output dimension
        dummy_image = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            dummy_out = image_embedder.encode_image(dummy_image)
        flow_dimensionality = dummy_out.shape[1]
        print(f"Inferred flow dimensionality: {flow_dimensionality}")

    print(f"Loading normalizing flow model from {flow_checkpoint}")

    if conditional_embeddings is not None:
        flow_model_loader = ConditionalFlowLoader(
            checkpoint_path=flow_checkpoint,
            device=device,
            dimensionality=flow_dimensionality,
            num_layers=flow_num_layers,
            condition_dim=conditional_embeddings.shape[1],
            use_batch_norm=use_batch_norm
        )
    else:
        flow_model_loader = UnconditionalFlowLoader(
            checkpoint_path=flow_checkpoint,
            device=device,
            dimensionality=flow_dimensionality,
            num_layers=flow_num_layers,
            use_batch_norm=use_batch_norm
        )
    flow_model = flow_model_loader.load()
    
    if loss_variant == 'sample':
        return SampleNormalizingFlowLoss(
            image_embedder=image_embedder,
            flow_model=flow_model,
            device=device,
            preprocess=preprocess,
            embedder_checkpoint=embedder_checkpoint,
            flow_checkpoint=flow_checkpoint,
            conditional_embeddings=conditional_embeddings,
            mode=sample_mode,
            num_distinct_samples=num_distinct_samples,
            normalize_sampled_embedding=normalize_sampled_embedding,
            normalize_context=normalize_context,
        )

    return MaxLogLikelihoodNormalizingFlowLoss(
        image_embedder=image_embedder,
        flow_model=flow_model,
        device=device,
        preprocess=preprocess,
        embedder_checkpoint=embedder_checkpoint,
        flow_checkpoint=flow_checkpoint,
        conditional_embeddings=conditional_embeddings,
    )

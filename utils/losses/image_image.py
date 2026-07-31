"""
Image-to-image losses for image optimization (MSE, SSIM, etc).

Note that all color space conversions (e.g., linear->sRGB) must be handled by the caller.
"""
import sys
from abc import ABC, abstractmethod

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import nn

from utils.image.image import resize_then_crop
from utils.image.preprocess_utils import preprocess_image_input
from utils.losses.base import BaseLoss
from utils.losses.loss_utils import compute_cosine_distance


# CLASSES
class ImageImageLoss(BaseLoss, ABC):
    def __init__(self, reference_image, comparison_height=None, comparison_width=None, device='cuda'):
        """Initialize ImageImageLoss with reference image.
        
        Args:
            reference_image: Reference image as a torch.Tensor or PIL.Image
            comparison_height, comparison_width: resize both images to this size for comparison. Uses the height and width from the reference image if not specified.
        """
        super().__init__()
        device = torch.device(device)
        self.device = device

        # Convert PIL to tensor if needed
        self.image_path = None
        if isinstance(reference_image, str):
            self.image_path = reference_image
            reference_image = Image.open(reference_image).convert('RGB')

        if isinstance(reference_image, Image.Image):
            reference_image = F.to_tensor(reference_image)

        self.reference_image = reference_image.float()
        self.comparison_height, self.comparison_width = comparison_height, comparison_width
        self.processed_target_image = self.preprocess(self.reference_image)
        if self.comparison_height is None or self.comparison_width is None:
            self.comparison_height, self.comparison_width = self.processed_target_image.shape[-2], self.processed_target_image.shape[-1]
            print(f"Initialized {self.__class__.__name__} with comparison resolution: {self.comparison_height}x{self.comparison_width}")

    def preprocess(self, img, verbose=False) -> torch.Tensor:
        """Preprocess image for comparison.
        
        Args:
            img: Image tensor or PIL image
            device: Device to move tensor to
            
        Returns:
            Preprocessed tensor
        """
        # img: torch.Tensor [C,H,W] or [H,W] or PIL.Image
        if isinstance(img, Image.Image):
            img = F.to_tensor(img)
        if img.dim() == 2: # In the case of black and white images, it is possible to have a 2 dimensional image, so unsequeeze so it's [1, H, W]
            img = img.unsqueeze(0)
        assert img.shape[-3] in (1, 3), "Image must have 1 or 3 channels"

        img = img.float().to(self.device)
        img = img.unsqueeze(0) if img.dim() == 3 else img  # [1,C,H,W]

        if self.comparison_height is not None and self.comparison_width is not None:
            if hasattr(self, 'processed_target_image') and self.processed_target_image is not None:
                # Check and see if sizes for the incoming image
                if img.shape[2] != self.comparison_height or img.shape[3] != self.comparison_width:
                    if verbose:
                        print(f"Resizing input image from ({img.shape[2]}, {img.shape[3]}) to ({self.comparison_height}, {self.comparison_width}) for comparison.")
                    img = resize_then_crop(img, target_height=self.comparison_height, target_width=self.comparison_width)
            else:
                img = resize_then_crop(img, target_height=self.comparison_height, target_width=self.comparison_width)
        return img

    def get_prompt_info(self):
        """Get prompt information for Image-Image loss with reference image."""
        return {
            "image_image_loss_type": self.__class__.__name__,
            "reference_image_resolution": list(self.reference_image.shape),
            "processed_target_image_resolution": list(self.processed_target_image.shape),
            "reference_image_path": self.image_path
        }

    def forward(self, image):
        """
        Compute an image-to-image loss between input and reference image.
        Args:
            image: torch.Tensor or PIL.Image, shape [C,H,W]
        Returns:
            Loss value between preprocessed input and reference image
        """
        incoming_image = self.preprocess(image)
        # Ensure same number of channels
        assert incoming_image.shape[1:] == self.processed_target_image.shape[1:], f"Image shape mismatch: Incoming image shape: {incoming_image.shape}, Target Image Shape: {self.processed_target_image.shape}"
        if incoming_image.shape[0] != self.processed_target_image.shape[0]:
            self.processed_target_image = self.processed_target_image.expand(incoming_image.shape[0], -1, -1, -1) # TODO: NOTE: If you ever need to restore the target image to the original batch size, you'll have to do that at some point :)
        return self._loss_implementation(incoming_image)
    
    @abstractmethod
    def _loss_implementation(self, incoming_image):
        pass

class MSELossWithReferenceImage(ImageImageLoss):
    """Mean Squared Error (MSE) loss with a reference image for comparison, with preprocessing and differentiable resizing."""
    def _loss_implementation(self, incoming_image):
        """Calculate MSE loss between input and reference image."""
        return nn.functional.mse_loss(incoming_image, self.processed_target_image)

class L1LossWithReferenceImage(ImageImageLoss):
    """L1 Loss (Mean Absolute Error) with a reference image for comparison, with preprocessing and differentiable resizing."""
    def _loss_implementation(self, incoming_image):
        """Calculate L1 loss between input and reference image."""
        return nn.functional.l1_loss(incoming_image, self.processed_target_image)

class SSIMLoss(ImageImageLoss):
    """SSIM Loss (Structural Similarity Index)."""
    def __init__(self, reference_image, comparison_height=None, comparison_width=None, device='cuda'):
        super().__init__(reference_image, comparison_height, comparison_width, device)
        from pytorch_msssim import SSIM
        self.ssim_metric = SSIM(data_range=1.0, size_average=False, channel=3)

    def _loss_implementation(self, incoming_image):
        target = self.processed_target_image
        if incoming_image.dim() == 3:
            incoming_image = incoming_image.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        return torch.tensor(1.0) - self.ssim_metric(incoming_image, target)

class LPIPSLoss(ImageImageLoss):
    """LPIPS Loss (Learned Perceptual Image Patch Similarity)."""
    def __init__(self, reference_image, comparison_height=None, comparison_width=None, device='cuda', backbone: str='vgg'):
        super().__init__(reference_image, comparison_height, comparison_width, device)
        from lpips import LPIPS
        self.lpips = LPIPS(net=backbone).to(device)
    
    def _loss_implementation(self, incoming_image):
        target = self.processed_target_image
        if incoming_image.dim() == 3:
            incoming_image = incoming_image.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        return self.lpips(incoming_image, target, normalize=True).squeeze() # Normalize [0, 1] to [-1, 1] range internally

class ImageImageCLIPLoss(ImageImageLoss):
    """Image-to-Image CLIP Loss."""
    def __init__(self, reference_image, clip_model, preprocess, comparison_height=None, comparison_width=None, device='cuda'):
        self.clip_preprocess = preprocess
        self.device=device
        super().__init__(reference_image, comparison_height, comparison_width, device) # Note that even though the comparison_height/width are provided, the preprocess will still resize the images to what is needed for the particular clip model.
        self.clip_model = clip_model
        with torch.no_grad():
            self.processed_target_image_features = self.clip_model.encode_image(self.processed_target_image)
            self.processed_target_image_features = self.processed_target_image_features / self.processed_target_image_features.norm(dim=1, keepdim=True)
        self.single_image_comparison_mode = self.processed_target_image_features.shape[0] == 1 # True if only one reference image in the batch
    
    def _loss_implementation(self, incoming_image):
        image_features = self.clip_model.encode_image(incoming_image)
        return compute_cosine_distance(self.processed_target_image_features, image_features, is_static_embedding_prenormalized=True)
    
    def preprocess(self, img) -> torch.Tensor:
        original = super().preprocess(img)
        return preprocess_image_input(original, preprocess=self.clip_preprocess, device=self.device)

class VGGStyleTransferLoss(ImageImageLoss):
    class VGGIntermediate(nn.Module):
        """VGG feature extractor that captures intermediate layer activations.
        
        Supports VGG16 and VGG19 architectures. Used for style transfer and 
        perceptual losses that compare features at multiple layers.
        """
        def __init__(self, requested=None, backbone='vgg16'):
            super().__init__()
            if requested is None:
                requested = []
            # Use register_buffer so they move to device automatically with the model
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            import torchvision.models as models
            self.intermediates = {}
            self.backbone = backbone
            if backbone == 'vgg16':
                self.vgg = models.vgg16(pretrained=True).features.eval()
            elif backbone == 'vgg19':
                self.vgg = models.vgg19(pretrained=True).features.eval()
            else:
                raise ValueError(f"Unsupported backbone: {backbone}. Choose 'vgg16' or 'vgg19'.")
            
            for i, m in enumerate(self.vgg.children()):
                if isinstance(m, nn.ReLU):   # set relu layers to NOT do relu in place
                    m.inplace = False
                if isinstance(m, nn.MaxPool2d):
                    self.vgg[i] = nn.AvgPool2d(2, 2)
                if i in requested:
                    def curry(idx):
                        def hook(module, input, output):
                            self.intermediates[idx] = output
                        return hook
                    m.register_forward_hook(curry(i))

        def forward(self, x):
            self.intermediates = {} # Clear previous activations
            self.vgg(self._normalize(x))
            return self.intermediates

        def _normalize(self, image):
            return (image - self.mean) / self.std

    def __init__(self, reference_image, requested_names=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'], comparison_height=224, comparison_width=224, device='cuda', backbone='vgg16'): # TODO: it might be required to do comparison at 224x224 for VGG
        super().__init__(reference_image, comparison_height, comparison_width, device)
        self.backbone = backbone
        self.requested_indices = self._get_requested_indices(requested_names, backbone)
        self.model = self.VGGIntermediate(requested=self.requested_indices, backbone=backbone).eval().to(self.device)
        
        with torch.no_grad():
            self.processed_target_image = self.processed_target_image.unsqueeze(0) if self.processed_target_image.dim() == 3 else self.processed_target_image # [1,C,H,W]
            activations = self.model(self.processed_target_image) 
            self.style_image_activations = [activations[i] for i in self.requested_indices]
            style_image_activation_feature_matrices = [self._construct_feature_matrix(activation) for activation in self.style_image_activations]
            self.style_gram_matrices = [self._compute_gram_matrix(feature_matrix) for feature_matrix in style_image_activation_feature_matrices]

    def _loss_implementation(self, incoming_image):
        activations = self.model(incoming_image) # returns the intermediate activations as a dict that maps from the layer index to the activation tensor
        generated_image_style_activations = [activations[i] for i in self.requested_indices]
        generated_image_feature_matrices = [self._construct_feature_matrix(activation) for activation in generated_image_style_activations]
        generated_image_gram_matrices = [self._compute_gram_matrix(feature_matrix) for feature_matrix in generated_image_feature_matrices]
        total_style_loss = 0.0
        for i in range(len(self.style_gram_matrices)):
            b, c, h_w = generated_image_feature_matrices[i].size()
            weight = 1.0 / (4.0 * (h_w ** 2)) # The division by c**2 is handled by using MSE, which already divides by the number of elements (channels) in the gram matrices.
            total_style_loss += weight * nn.functional.mse_loss(generated_image_gram_matrices[i], self.style_gram_matrices[i])
        return total_style_loss

    def _compute_gram_matrix(self, activation_feature_matrix):
        # activation_feature_matrix is (Batch, Channels, width * height) matrix
        transposed = activation_feature_matrix.transpose(1, 2) # Now it's (Batch, width * height, Channels)
        return torch.bmm(activation_feature_matrix, transposed) # Now it's a feature by feature size matrix
    
    def _construct_feature_matrix(self, activations):
        size = activations.size() # size is the tuple (batch, channels, height, width)
        batch = size[0]
        channels = size[1]
        height = size[2]
        width = size[3]
        return activations.view(batch, channels, height * width)

    def _get_requested_indices(self, requested, backbone='vgg16'):
        if backbone == 'vgg16':
            vgg_names = [
                "conv1_1", "relu1_1", "conv1_2", "relu1_2", "maxpool1",
                "conv2_1", "relu2_1", "conv2_2", "relu2_2", "maxpool2",
                "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "maxpool3",
                "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "maxpool4",
                "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "maxpool5"
            ]
        elif backbone == 'vgg19':
            vgg_names = [
                "conv1_1", "relu1_1", "conv1_2", "relu1_2", "maxpool1",
                "conv2_1", "relu2_1", "conv2_2", "relu2_2", "maxpool2",
                "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "conv3_4", "relu3_4", "maxpool3",
                "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "conv4_4", "relu4_4", "maxpool4",
                "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "conv5_4", "relu5_4", "maxpool5"
            ]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'vgg16' or 'vgg19'.")
        return [vgg_names.index(name) for name in requested]

class ImageEmbeddingSimilarityLoss(ImageImageLoss):
    def __init__(self, reference_image, embedder_checkpoint, comparison_height=224, comparison_width=224, device='cuda', model_name='vit_b_32', mode='cosine'):
        self._validate_embedding_similarity_mode(mode)
        self.mode = mode

        super().__init__(reference_image, comparison_height, comparison_width, device)
        self.embedder_checkpoint = embedder_checkpoint
        self.model_name = model_name
        self.image_embedder, _ = self._load_image_embedder(embedder_checkpoint, device, model_name)
        with torch.no_grad():
            self.processed_target_embedding = self.image_embedder.encode_image(self.processed_target_image) # type: ignore
            if self.mode == 'cosine': # Normalize the embedding so you don't have to do it each time
                self.processed_target_embedding = self.processed_target_embedding / self.processed_target_embedding.norm(dim=1, keepdim=True)

    def _load_image_embedder(self, checkpoint_path, device, model_name='vit_b_32'):
        """Load the image embedder model from checkpoint."""
        
        # Add the fine_tuning codebase to the path for model loading utilities
        # TODO: PATH_UPDATE fine-tuning codebase path
        FINE_TUNING_PATH = ""
        if FINE_TUNING_PATH not in sys.path:
            sys.path.insert(0, FINE_TUNING_PATH)
        
        from utils.model.model_utils import create_vision_only_model
        
        # Infer model configuration from checkpoint
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Detect projection head configuration
        image_head_layers = None
        proj_keys = [k for k in state_dict if k.startswith('image_projection.')]
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
        
        # Create the vision-only model
        model, preprocess = create_vision_only_model(
            model_name=model_name,
            device=device,
            pretrained=False,
            image_head_layers=image_head_layers,
        )
        
        # Load weights
        model.load_state_dict(state_dict, strict=False) # strict=False to allow loading models with different keys (e.g. missing text encoder)
        model.to(device)
        model.eval()
        
        return model, preprocess
    
    def _loss_implementation(self, incoming_image):
        incoming_embedding = self.image_embedder.encode_image(incoming_image) # type: ignore
        return self._compute_embedding_similarity_loss(
            static_embedding=self.processed_target_embedding,
            dynamic_embedding=incoming_embedding,
            mode=self.mode,
            is_static_embedding_prenormalized=True,
        )
    def _validate_embedding_similarity_mode(self, mode: str):
        if mode not in ('cosine', 'l2'):
            raise ValueError(f"Unsupported mode: {mode}. Choose 'cosine' or 'l2'.")

    def _compute_embedding_similarity_loss(
        self,
        static_embedding,
        dynamic_embedding,
        mode: str = 'cosine',
        is_static_embedding_prenormalized: bool = True,
    ):
        self._validate_embedding_similarity_mode(mode)
        if mode == 'cosine':
            return compute_cosine_distance(
                static_embedding,
                dynamic_embedding,
                is_static_embedding_prenormalized=is_static_embedding_prenormalized,
            )
        return torch.nn.functional.mse_loss(dynamic_embedding, static_embedding)
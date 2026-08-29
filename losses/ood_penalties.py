"""Out-of-distribution (OOD) penalty losses.

These losses are discussed in section 5.2.1 of:
"Differentiable Objectives for 3D Scene Relighting via Gradient Descent on OLAT Basis Coefficients"
(https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=12256&context=etd)
"""

import os

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, StableDiffusionPipeline

from losses.base import BaseLoss


class DiffusionConfusionLoss(BaseLoss):
    """Loss measuring noise prediction error of a diffusion UNet on rendered images."""

    def __init__(self, device: str, cache_dir: str | None = None):
        """Initialize DiffusionConfusionLoss with Stable Diffusion 2.1.

        Args:
            device: PyTorch computation device.
            cache_dir: Optional directory path for caching diffusion weights.
        """
        super().__init__()
        if cache_dir is None:
            from config import DEFAULT_MODEL_WEIGHTS_DIR
            cache_dir = os.path.join(DEFAULT_MODEL_WEIGHTS_DIR, "sd_cache")
        model_id = "stabilityai/stable-diffusion-2-1-base"
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir=cache_dir).to(device)
        num_timesteps = 50
        self.pipe.scheduler.set_timesteps(num_timesteps)

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between added noise and UNet-predicted noise.

        Args:
            input_image: Image tensor of shape [N, C, H, W] in [0, 1].

        Returns:
            Scalar noise prediction error loss.
        """
        latent = self.pipe.vae.encode(input_image).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        timestep = self.pipe.scheduler.timesteps[10]
        noise = torch.randn_like(latent)
        noisy_latent = self.pipe.scheduler.add_noise(latent, noise, timestep)

        prompt = ""
        text_embeddings = self.pipe.encode_prompt(prompt, self.device, 1, False, negative_prompt=None)

        predicted_noise = self.pipe.unet(noisy_latent, timestep, encoder_hidden_states=text_embeddings[0]).sample
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def get_prompt_info(self) -> dict[str, str]:
        """Get prompt and configuration information for logging.

        Returns:
            Dictionary describing the loss.
        """
        return {"loss_description": "Diffusion Confusion Loss using Stable Diffusion 2.1"}


class VAEReconstructionLoss(BaseLoss):
    """Loss penalizing reconstruction error through a Stable Diffusion VAE."""

    def __init__(self, device: str, cache_dir: str | None = None):
        """Initialize VAEReconstructionLoss.

        Args:
            device: PyTorch device.
            cache_dir: Optional cache directory for VAE weights.
        """
        super().__init__()
        if cache_dir is None:
            from config import DEFAULT_MODEL_WEIGHTS_DIR
            cache_dir = os.path.join(DEFAULT_MODEL_WEIGHTS_DIR, "sd_cache")
        print(f"Using cache directory for VAE: {cache_dir}")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            subfolder="vae",
            cache_dir=cache_dir,
        ).to(device)
        self.vae.eval()

    def get_reconstructed_image(self, input_image: torch.Tensor) -> torch.Tensor:
        """Pass image through VAE encode/decode.

        Args:
            input_image: Image tensor [N, C, H, W].

        Returns:
            Reconstructed image tensor sample.
        """
        result = self.vae(input_image)
        return result.sample

    def encode(self, input_image: torch.Tensor) -> torch.Tensor:
        """Encode image into VAE latent space mode.

        Args:
            input_image: Image tensor [N, C, H, W].

        Returns:
            Latent mode tensor.
        """
        latent_dist = self.vae.encode(input_image).latent_dist
        return latent_dist.mode()

    def get_reconstruction_loss(self, input_image: torch.Tensor) -> torch.Tensor:
        """Compute VAE reconstruction MSE loss with downscaled (to reduce memory usage) image.

        Args:
            input_image: Image tensor [N, C, H, W].

        Returns:
            Reconstruction MSE scalar tensor.
        """
        input_image = F.interpolate(input_image, scale_factor=0.5, mode="bilinear", align_corners=False)
        reconstructed_image = self.get_reconstructed_image(input_image)
        reconstructed_image = F.interpolate(reconstructed_image, size=input_image.shape[2:], mode="bilinear", align_corners=False)
        loss = F.mse_loss(reconstructed_image, input_image)
        return loss

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        """Evaluate VAE reconstruction loss on input image.

        Args:
            input_image: Input image tensor.

        Returns:
            Scalar reconstruction loss.
        """
        return self.get_reconstruction_loss(input_image)

    def get_prompt_info(self) -> dict[str, str]:
        """Get prompt and configuration information for logging.

        Returns:
            Dictionary describing the loss.
        """
        return {"loss_description": "VAE Reconstruction Loss using Stable Diffusion 2.1 VAE"}


class AverageImageLuminanceLoss(BaseLoss):
    """Loss that penalizes deviation of average linear luminance from a target value."""

    def __init__(self, device: str, goal_average_luminance: float = 1.0):
        """Initialize AverageImageLuminanceLoss.

        Args:
            device: PyTorch device.
            goal_average_luminance: Target average linear luminance value.
        """
        super().__init__()
        self.image_luminance_converter_tensor = torch.Tensor([0.2126, 0.7152, 0.0722]).view(3, 1, 1).to(device)
        self.goal_average_luminance = torch.tensor(goal_average_luminance).to(device)

    def srgb_to_linear(self, image: torch.Tensor) -> torch.Tensor:
        """Convert sRGB image tensor to linear space.

        Args:
            image: sRGB image tensor in [0, 1].

        Returns:
            Linear RGB tensor.
        """
        linear_mask = image <= 0.04045
        linear_image = torch.zeros_like(image)
        linear_image[linear_mask] = image[linear_mask] / 12.92
        linear_image[~linear_mask] = ((image[~linear_mask] + 0.055) / 1.055) ** 2.4
        return linear_image

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Compute MSE between mean linear luminance and goal luminance.

        Args:
            image: sRGB image tensor [C, H, W] or [N, C, H, W].

        Returns:
            Luminance difference penalty scalar tensor.
        """
        linear_image = self.srgb_to_linear(image)
        luminance = torch.sum(linear_image * self.image_luminance_converter_tensor, dim=0)
        return F.mse_loss(luminance.mean(), self.goal_average_luminance)

    def get_prompt_info(self) -> dict[str, str]:
        """Get prompt and configuration information for logging.

        Returns:
            Dictionary describing the goal luminance.
        """
        return {"loss_description": f"Image Darkness Loss with goal average luminance {self.goal_average_luminance}"}
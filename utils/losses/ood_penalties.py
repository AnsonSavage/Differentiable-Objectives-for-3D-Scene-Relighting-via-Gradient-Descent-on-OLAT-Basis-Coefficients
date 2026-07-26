"""These losses are discussed in  section 5.2.1 of
Differentiable Objectives for 3D Scene Relighting
via Gradient Descent on OLAT Basis Coefficients
(https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=12256&context=etd)

They were not found to be particularly effective at reducing OOD optima.
"""

import torch
import torch.nn.functional as F
from base import BaseLoss
from diffusers import AutoencoderKL, StableDiffusionPipeline


class DiffusionConfusionLoss(BaseLoss):
    def __init__(self, device, cache_dir=None):  # TODO: PATH_UPDATE diffusion model cache directory
        # Load the pipeline
        model_id = "stabilityai/stable-diffusion-2-1-base" # TODO: use a newer model
        self.device = device
        if cache_dir is not None:
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True).to(device)
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
        num_timesteps = 50
        self.pipe.scheduler.set_timesteps(num_timesteps)
    
    def forward(self, input_image):
        latent = self.pipe.vae.encode(input_image).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        timestep = self.pipe.scheduler.timesteps[10]
        noise = torch.randn_like(latent)
        noisy_latent = self.pipe.scheduler.add_noise(latent, noise, timestep)

        prompt = ""  # empty string for unconditional guidance
        text_embeddings = self.pipe.encode_prompt(prompt, self.device, 1, False, negative_prompt=None)

        predicted_noise = self.pipe.unet(noisy_latent, timestep, encoder_hidden_states=text_embeddings[0]).sample
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    def get_prompt_info(self):
        return {"loss_description": "Diffusion Confusion Loss using Stable Diffusion 2.1"}
    
    def __call__(self, input_image: torch.Tensor):
        return self.forward(input_image)

class VAEReconstructionLoss(BaseLoss):
    def __init__(self, device, cache_dir=None):
        # Load the pre-trained VAE from Stable Diffusion 2.1
        if cache_dir is not None:
            print(f"Using cache directory for VAE: {cache_dir}")
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/stable-diffusion-2-1-base", 
                subfolder="vae",
                cache_dir=cache_dir,
                local_files_only=True
            ).to(device)
        else:
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/stable-diffusion-2-1-base", 
                subfolder="vae"
            ).to(device)
        self.vae.eval()
        

    def get_reconstructed_image(self, input_image: torch.Tensor):
        result = self.vae(input_image)
        return result.sample
    
    def encode(self, input_image: torch.Tensor):
        latent_dist = self.vae.encode(input_image).latent_dist
        return latent_dist.mode()
    
    def get_reconstruction_loss(self, input_image: torch.Tensor):
        # Scale image by 1/2 to reduce memory usage
        input_image = F.interpolate(input_image, scale_factor=0.5, mode="bilinear", align_corners=False)
        reconstructed_image = self.get_reconstructed_image(input_image)
        # Scale the reconstructed image to be the same size as the original input
        reconstructed_image = F.interpolate(reconstructed_image, size=input_image.shape[2:], mode="bilinear", align_corners=False)
        loss = F.mse_loss(reconstructed_image, input_image)
        return loss

    def forward(self, input_image: torch.Tensor):
        return self.get_reconstruction_loss(input_image)
    
    def get_prompt_info(self):
        return {"loss_description": "VAE Reconstruction Loss using Stable Diffusion 2.1 VAE"}
    
    def __call__(self, input_image: torch.Tensor):
        return self.get_reconstruction_loss(input_image)

class AverageImageLuminanceLoss(BaseLoss):
    """Loss that penalizes the average luminance of an image being different from a goal average luminance."""
    def __init__(self, device, goal_average_luminance: float = 1.0):
        self.image_luminance_converter_tensor = torch.Tensor([0.2126, 0.7152, 0.0722]).view(3, 1, 1).to(device)
        self.goal_average_luminance = torch.tensor(goal_average_luminance).to(device)
    def srgb_to_linear(self, image):
        linear_mask = image <= 0.04045
        linear_image = torch.zeros_like(image)
        linear_image[linear_mask] = image[linear_mask] / 12.92
        linear_image[~linear_mask] = ((image[~linear_mask] + 0.055) / 1.055) ** 2.4
        return linear_image

    def __call__(self, image):
        linear_image = self.srgb_to_linear(image)
        luminance = torch.sum(linear_image * self.image_luminance_converter_tensor, dim=0)
        return F.mse_loss(luminance.mean(), self.goal_average_luminance)
    def get_prompt_info(self):
        return {'loss_description': f'Image Darkness Loss with goal average luminance {self.goal_average_luminance}'}
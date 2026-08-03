import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from diffusers import FluxKontextPipeline

from utils.image.display import display_image_batch_grid
from utils.image.image import resize_then_crop
from losses.image_image import ImageImageLoss


class FLUXKontextRelighter():
    def __init__(
        self, 
        cache_dir: str | None = None, 
        device: str = 'cuda', 
        seed: int | None = None,
        base_model_repo: str = "black-forest-labs/FLUX.1-Kontext-dev",
        lora_repo: str = "kontext-community/relighting-kontext-dev-lora-v3",
        lora_weight_name: str = "relighting-kontext-dev-lora-v3.safetensors",
    ):
        if not cache_dir:
            from config import DEFAULT_MODEL_WEIGHTS_DIR
            cache_dir = os.path.join(DEFAULT_MODEL_WEIGHTS_DIR, "flux")

        kontext_dir = os.path.join(cache_dir, "kontext-dev")
        lora_dir = os.path.join(cache_dir, "community_relighting_lora")

        # Load or download base FLUX Kontext pipeline
        if os.path.exists(kontext_dir) and any(os.scandir(kontext_dir)):
            print(f"Loading FLUX Kontext pipeline from local directory: {kontext_dir}...")
            self.pipe = FluxKontextPipeline.from_pretrained(
                kontext_dir, 
                torch_dtype=torch.bfloat16, 
                local_files_only=True
            )
        else:
            print(f"FLUX Kontext pipeline not found locally at {kontext_dir}. Downloading from Hugging Face ({base_model_repo})...")
            os.makedirs(cache_dir, exist_ok=True)
            self.pipe = FluxKontextPipeline.from_pretrained(
                base_model_repo, 
                torch_dtype=torch.bfloat16,
                cache_dir=cache_dir
            )

        # Load or download relighting LoRA weights
        local_lora_file = os.path.join(lora_dir, lora_weight_name)
        direct_cache_lora_file = os.path.join(cache_dir, lora_weight_name)

        if os.path.exists(local_lora_file):
            actual_lora_dir = lora_dir
            actual_weight_name = lora_weight_name
            print(f"Loading LoRA weights from local path: {local_lora_file}")
        elif os.path.exists(direct_cache_lora_file):
            actual_lora_dir = cache_dir
            actual_weight_name = lora_weight_name
            print(f"Loading LoRA weights from local path: {direct_cache_lora_file}")
        else:
            print(f"LoRA weights '{lora_weight_name}' not found locally. Downloading from Hugging Face ({lora_repo})...")
            os.makedirs(lora_dir, exist_ok=True)
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(
                repo_id=lora_repo,
                filename=lora_weight_name,
                local_dir=lora_dir,
            )
            actual_lora_dir = lora_dir
            actual_weight_name = os.path.basename(downloaded_path)

        self.pipe.load_lora_weights(actual_lora_dir, weight_name=actual_weight_name)
        self.pipe.to(device)
        print(f"LoRA weights from {actual_lora_dir} loaded successfully.")
        self.device = device
        self.seed = seed

    def _get_adjusted_width_and_height(self, width, height):
        vae_scale_factor = self.pipe.vae_scale_factor
        # original_height, original_width = height, width
        aspect_ratio = width / height

        max_area = 1024 ** 2
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)

        multiple_of = vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of
        return width, height

    def relight_with_prompt(self, image_to_relight: torch.Tensor, prompt: str, num_images_per_prompt: int, display: bool = False, save_dir: str | None = None) -> torch.Tensor:
        pipe_kwargs = dict(
            prompt=prompt,
            height=image_to_relight.shape[-2],
            width=image_to_relight.shape[-1],
            image=image_to_relight,
            num_images_per_prompt=num_images_per_prompt,
            output_type='pt',  # could also be np
        )
        if self.seed is not None:
            pipe_kwargs['generator'] = torch.Generator(self.device).manual_seed(self.seed)

        output = self.pipe(**pipe_kwargs)
        images = output.images

        # If a save directory is provided, save there; otherwise optionally display
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig, _ = display_image_batch_grid(images.float(), show=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"relight_grid_seed{self.seed}_{timestamp}.png"
            out_path = os.path.join(save_dir, filename)
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved relight grid to {out_path}")
        elif display:
            # Only display if not saving
            display_image_batch_grid(images.float())
        return images

class RelightImageCache():
    def __init__(self, relighter: FLUXKontextRelighter, image_to_relight: torch.Tensor):
        self.relighter = relighter
        self.height, self.width, self.image_to_relight = self._initialize_relight_image(image_to_relight)
        self.cache = {} # This cache maps a tuple of (prompt, number) to the resulting tensor

    def _initialize_relight_image(self, image_to_relight: torch.Tensor):
        """ Sets the image to be relit, resizing as needed for the model.

        (Note that resizing is performed using the resize_then_crop function, which is also done in the MSELossWithReferenceImage class, so you likely do not need to store the height and width if you pass the MSELossWithReferenceImage the resulting reference image generated by this class.

        Args:
            image_to_relight: A 4D tensor (B, C, H, W)
        Returns:
            height, width: The adjusted height and width after resizing
        """
        assert image_to_relight.ndim == 4, "Input image must be a 4D tensor (B, C, H, W)"
        width, height = image_to_relight.shape[3], image_to_relight.shape[2]
        width, height = self.relighter._get_adjusted_width_and_height(width, height)
        cropped_image = resize_then_crop(image_to_relight, target_height=height, target_width=width)
        cropped_image = cropped_image.to(device=self.relighter.device, dtype=torch.bfloat16)
        assert cropped_image.shape[2] == height and cropped_image.shape[3] == width, \
            f"Resized image has incorrect dimensions: expected ({height}, {width}), got ({cropped_image.shape[2]}, {cropped_image.shape[3]})"
        # Return new height and width
        return height, width, cropped_image
    
    def clear_cache(self):
        self.cache = {}

    def get_relighted_image(self, prompt: str, num_images_per_prompt: int, save_dir: str | None = None, display: bool = False) -> torch.Tensor:
        key = (prompt, num_images_per_prompt)
        if key not in self.cache:
            print(f"Relighting image with prompt: '{prompt}' for {num_images_per_prompt} images.")
            relit_images = self.relighter.relight_with_prompt(
                image_to_relight=self.image_to_relight,
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                display=display,
                save_dir=save_dir
            )
            self.cache[key] = relit_images
        else:
            print(f"Using cached relighted images for prompt: '{prompt}'")
        return self.cache[key]

class FluxLoss(torch.nn.Module):
    def __init__(self, cache: RelightImageCache, target_text: str, image_comparison_criterion_cls: type[ImageImageLoss], num_relighted_images: int = 1, save_dir: str | None = None, display=False):
        super(FluxLoss, self).__init__()
        # Defer relighting until we can link to the run directory
        self.images_cache = cache
        self.target_text = target_text
        self.num_relighted_images = num_relighted_images
        self._pending_save_dir = save_dir  # If provided, use it; otherwise may be set later by the trainer
        self.relit_image: torch.Tensor | None = None

        if not isinstance(image_comparison_criterion_cls, type):
            raise TypeError("image_comparison_criterion_cls must be a class (subclass of ImageImageLoss), not an instance")
        if not issubclass(image_comparison_criterion_cls, ImageImageLoss):
            raise TypeError("image_comparison_criterion_cls must be a subclass of ImageImageLoss")

        self.image_comparison_criterion_type: type[ImageImageLoss] = image_comparison_criterion_cls
        self.image_comparison_criterion: ImageImageLoss | None = None

        if save_dir is None and display: # As of right now, save_dir and display are mutually exclusive
            self._ensure_initialized()

    def _ensure_initialized(self):
        """Create relit image and underlying image-comparison criterion if not already initialized.

        The caller must provide a loss class (subclass of `ImageImageLoss`). This method will
        instantiate that class with the generated reference image.
        """
        if self.image_comparison_criterion is None:
            self.relit_image = self.images_cache.get_relighted_image(
                prompt=self.target_text,
                num_images_per_prompt=self.num_relighted_images,
                save_dir=self._pending_save_dir,
                display=self._pending_save_dir is None
            )
            # Instantiate the provided loss class with the reference image
            self.image_comparison_criterion = self.image_comparison_criterion_type(reference_image=self.relit_image)

    def on_run_dir_created(self, run_dir: str):
        """Hook called by the optimization loop to link this loss's save dir to the run folder."""
        if self._pending_save_dir is None:
            self._pending_save_dir = run_dir
    
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        self._ensure_initialized() # This is all done to allow the optimize function to set the save directory
        return self.image_comparison_criterion(input_image)

    def get_prompt_info(self):
        return {
            'target_text': self.target_text,
            'flux_seed': self.images_cache.relighter.seed
        }
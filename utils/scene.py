from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from utils.color.linear_to_srgb_converters import (
    LinearRec709ToAgXBase,
    LinearRec709TosRGB,
)
from utils.display import display_tensor
from utils.load_utils import (
    get_images_tensor_from_multi_layer_exr,
    get_images_tensor_from_OLAT_dir,
)


class Scene(ABC):
    def __init__(self, name: str, description: str = "", device: str = 'cuda'):
        self.name = name
        self.description = description
        self.device = device

    @abstractmethod
    def get_optimizable_images(self) -> torch.Tensor:
        """Load and return a tensor of optimizable images.

        Returns:
            Tensor of shape (N, H, W, C) where N is number of images,
            H is height, W is width, and C is number of channels (3).
        """

    @abstractmethod
    def get_light_name_list(self) -> list[str] | None:
        """Get a list of light names corresponding to the optimizable images.

        Returns:
            List of light names or None if not applicable.
        """

    def get_non_optimized_lights(self) -> torch.Tensor | None:
        """Load and return a tensor of non-optimized lights to add to predictions.

        Returns:
            Tensor of shape (H, W, C) or None if no non-optimized lights.
        """
        return None

    def get_alpha_mask(self) -> torch.Tensor | None:
        """Load and return an alpha mask tensor if available.

        Returns:
            Tensor of shape (H, W, 1) with values in the range [0, 1] or None if no mask.
        """
        return None
    
    def get_scene_metadata(self) -> dict:
        """Get metadata about the scene.

        Returns:
            Dictionary of metadata.
        """
        return {"scene_name": self.name}

    def get_combined_image(self, color_space_converter: LinearRec709TosRGB | None, apply_alpha_mask: bool=False) -> torch.Tensor:
        """
        Returns the combined image of optimizable images and non-optimized lights if available. (H, W, C)

        Args:
            color_space_converter: Converter to apply to the final image, or None to skip conversion.
            apply_alpha_mask: Whether to apply the alpha mask if available.
        Returns:
            Combined image tensor of shape (H, W, C).
        """
        images = self.get_optimizable_images()
        non_optimized = self.get_non_optimized_lights()

        total_image = torch.sum(images, dim=0)
        if non_optimized is not None:
            total_image += non_optimized
        if apply_alpha_mask:
            alpha_mask = self.get_alpha_mask()
            if alpha_mask is not None:
                total_image *= alpha_mask
        if color_space_converter is not None:
            total_image = color_space_converter(total_image.permute(2, 0, 1)).permute(1, 2, 0) # TODO: we should refactor the code to stop doing so much permuting :)
        return total_image

    def display_scene(self, display_individual_OLATs=True, color_space_converter: LinearRec709TosRGB = LinearRec709ToAgXBase()) -> None:
        ''' Display the optimizable images plus the non-optimized lights if available '''
        
        images = self.get_optimizable_images()
        total_image = self.get_combined_image(color_space_converter)
        print("Displaying scene:", self.name)
        print("Using color space converter:", color_space_converter.settings_info())
        print("\tTotal images")
        display_tensor(total_image.permute(2, 0, 1))

        if display_individual_OLATs:
            print("\tIndividual optimizable images:")
            light_names = self.get_light_name_list()
            if light_names is None:
                raise ValueError("Scene did not provide light names for optimizable images")
            assert len(light_names) == images.shape[0], "Number of light names must match number of images"
            for i in range(images.shape[0]):
                print(f"\t\tLight: {light_names[i]}")
                display_tensor(color_space_converter(images[i].permute(2, 0, 1)))
        
        alpha_mask = self.get_alpha_mask()
        if alpha_mask is not None:
            print("\tAlpha mask:")
            display_tensor(alpha_mask.permute(2, 0, 1))

    def get_image_resolution(self) -> tuple[int, int]:
        """ Get the resolution (width, height) of the optimizable images.
        """
        images = self.get_optimizable_images()
        if images.shape[0] < 1:
            raise ValueError("No optimizable images found in scene")
        return images.shape[-2], images.shape[-3]  # Return (width, height)

class OLATDirScene(Scene):
    def __init__(
        self,
        name: str,
        description: str,
        path_to_olat_dir: str,
        name_of_non_optimized_lights_file: str | None = None,
        include_alpha_mask: bool = False,
        alpha_mask_path: str | None = None,
        device: str = 'cuda',
    ):
        super().__init__(name, description, device=device)
        self.optimizable_images, self.non_optimized_lights_tensor, self.light_name_list = get_images_tensor_from_OLAT_dir(
            path_to_olat_dir,
            name_of_non_optimized_lights_layer=name_of_non_optimized_lights_file,
            device=device,
        )
        if len(self.optimizable_images) < 1:
            raise ValueError(f"No optimizable images found in directory: {path_to_olat_dir}")

        # Optionally load an alpha mask for this scene
        self._alpha_mask: torch.Tensor | None = None
        if include_alpha_mask:
            try:
                from utils.load_utils import load_alpha_tensor
                mask_path = Path(alpha_mask_path) if alpha_mask_path is not None else Path(path_to_olat_dir) / 'alpha.exr'
                if mask_path.exists():
                    self._alpha_mask = load_alpha_tensor(str(mask_path), device=device)
            except Exception as e:
                print(f"Warning: Failed to load alpha mask: {e}")
    
    def get_optimizable_images(self) -> torch.Tensor:
        return self.optimizable_images
    
    def get_light_name_list(self) -> list[str] | None:
        return self.light_name_list

    def get_non_optimized_lights(self) -> torch.Tensor | None:
        return self.non_optimized_lights_tensor

    def get_alpha_mask(self) -> torch.Tensor | None:
        return self._alpha_mask

class MultiLayerEXRScene(Scene):
    def __init__(self, name: str, description: str, path_to_exr: str, return_non_optimized_lights_layer: bool = False, device: str = 'cuda'):
        super().__init__(name, description, device=device)
        self.optimizable_images, self.non_optimized_lights_tensor, self.light_name_list = get_images_tensor_from_multi_layer_exr(
            path_to_exr, 
            return_non_optimized_lights_layer=return_non_optimized_lights_layer,
            device=device
        )
        if len(self.optimizable_images) < 1:
            raise ValueError(f"No optimizable images found in EXR file: {path_to_exr}")
    
    def get_optimizable_images(self) -> torch.Tensor:
        return self.optimizable_images
    
    def get_light_name_list(self) -> list[str] | None:
        return self.light_name_list

    def get_non_optimized_lights(self) -> torch.Tensor | None:
        return self.non_optimized_lights_tensor
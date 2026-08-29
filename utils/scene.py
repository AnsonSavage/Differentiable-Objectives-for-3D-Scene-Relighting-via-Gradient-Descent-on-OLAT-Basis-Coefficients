"""Scene abstractions for loading and managing OLAT lighting data."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from utils.color.linear_to_srgb_converters import (
    LinearRec709ToAgXBase,
    LinearRec709TosRGB,
)
from utils.image.display import display_tensor
from utils.image.load_utils import (
    get_images_tensor_from_multi_layer_exr,
    get_images_tensor_from_OLAT_dir,
)


class Scene(ABC):
    """Abstract base class representing an optimizable 3D scene with OLAT lighting."""

    def __init__(self, name: str, description: str = "", device: str = "cuda"):
        """Initialize Scene.

        Args:
            name: Identifier name for the scene.
            description: Text description of scene contents.
            device: Computation device for image tensors.
        """
        self.name = name
        self.description = description
        self.device = device

    @abstractmethod
    def get_optimizable_images(self) -> torch.Tensor:
        """Load and return a tensor of optimizable light basis images.

        Returns:
            Tensor of shape (N, H, W, C) where N is number of lights,
            H is height, W is width, and C is number of channels (3).
        """

    @abstractmethod
    def get_light_name_list(self) -> list[str] | None:
        """Get list of light names corresponding to the optimizable images.

        Returns:
            List of light name strings or None if not applicable.
        """

    def get_non_optimized_lights(self) -> torch.Tensor | None:
        """Load and return constant non-optimized ambient/base lights.

        Returns:
            Tensor of shape (H, W, C) or None if absent.
        """
        return None

    def get_alpha_mask(self) -> torch.Tensor | None:
        """Load and return an alpha mask tensor if available.

        Returns:
            Tensor of shape (H, W, 1) in [0, 1] or None if unavailable.
        """
        return None

    def get_scene_metadata(self) -> dict:
        """Get metadata dictionary for the scene.

        Returns:
            Dictionary containing scene metadata.
        """
        return {"scene_name": self.name}

    def get_combined_image(
        self,
        color_space_converter: LinearRec709TosRGB | None,
        apply_alpha_mask: bool = False,
    ) -> torch.Tensor:
        """Return composite sum of all light passes with optional tonemapping and mask.

        Args:
            color_space_converter: Color space converter or None to keep linear.
            apply_alpha_mask: Whether to multiply by alpha mask if available.

        Returns:
            Combined image tensor of shape (H, W, C).
        """
        images = self.get_optimizable_images()
        non_optimized = self.get_non_optimized_lights()

        total_image = torch.sum(images, dim=0)
        if non_optimized is not None:
            total_image += non_optimized
        if color_space_converter is not None:
            total_image = color_space_converter(total_image.permute(2, 0, 1)).permute(1, 2, 0) # TODO: Refactor to remove extra permute calls
        if apply_alpha_mask:
            alpha_mask = self.get_alpha_mask()
            if alpha_mask is not None:
                total_image *= alpha_mask
        return total_image

    def display_scene(
        self,
        display_individual_OLATs: bool = True,
        color_space_converter: LinearRec709TosRGB = LinearRec709ToAgXBase(),
    ) -> None:
        """Display the combined image and individual OLAT light passes.

        Args:
            display_individual_OLATs: If True, displays every light pass individually.
            color_space_converter: Display color transform (defaults to AgX Base).

        Raises:
            ValueError: If individual display is requested but scene lacks light names.
        """
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
        """Get spatial resolution (width, height) of scene images.

        Returns:
            Tuple of (width, height).

        Raises:
            ValueError: If scene has no optimizable images.
        """
        images = self.get_optimizable_images()
        if images.shape[0] < 1:
            raise ValueError("No optimizable images found in scene")
        return images.shape[-2], images.shape[-3]


class OLATDirScene(Scene):
    """Scene loaded from a directory containing per-light EXR files."""

    def __init__(
        self,
        name: str,
        description: str,
        path_to_olat_dir: str,
        name_of_non_optimized_lights_file: str | None = None,
        include_alpha_mask: bool = False,
        alpha_mask_path: str | None = None,
        device: str = "cuda",
    ):
        """Initialize OLAT directory scene.

        Args:
            name: Scene name.
            description: Description of the scene.
            path_to_olat_dir: Directory containing optimizable_lights subfolder.
            name_of_non_optimized_lights_file: Optional filename of base light pass.
            include_alpha_mask: Whether to look for and load alpha.exr mask.
            alpha_mask_path: Explicit custom path to alpha mask EXR.
            device: PyTorch device.

        Raises:
            ValueError: If no images are found in path_to_olat_dir.
        """
        super().__init__(name, description, device=device)
        self.optimizable_images, self.non_optimized_lights_tensor, self.light_name_list = get_images_tensor_from_OLAT_dir(
            path_to_olat_dir,
            name_of_non_optimized_lights_layer=name_of_non_optimized_lights_file,
            device=device,
        )
        if len(self.optimizable_images) < 1:
            raise ValueError(f"No optimizable images found in directory: {path_to_olat_dir}")

        self._alpha_mask: torch.Tensor | None = None
        if include_alpha_mask:
            try:
                from utils.image.load_utils import load_alpha_tensor
                mask_path = Path(alpha_mask_path) if alpha_mask_path is not None else Path(path_to_olat_dir) / "alpha.exr"
                if mask_path.exists():
                    self._alpha_mask = load_alpha_tensor(str(mask_path), device=device)
            except Exception as e:
                print(f"Warning: Failed to load alpha mask: {e}")

    def get_optimizable_images(self) -> torch.Tensor:
        """Return stack of optimizable light tensors (N, H, W, C)."""
        return self.optimizable_images

    def get_light_name_list(self) -> list[str] | None:
        """Return list of light filenames."""
        return self.light_name_list

    def get_non_optimized_lights(self) -> torch.Tensor | None:
        """Return non-optimized base lighting tensor if loaded."""
        return self.non_optimized_lights_tensor

    def get_alpha_mask(self) -> torch.Tensor | None:
        """Return alpha mask tensor if loaded."""
        return self._alpha_mask


class MultiLayerEXRScene(Scene):
    """Scene loaded from a single multi-layer OpenEXR file."""

    def __init__(
        self,
        name: str,
        description: str,
        path_to_exr: str,
        return_non_optimized_lights_layer: bool = False,
        device: str = "cuda",
    ):
        """Initialize MultiLayerEXRScene.

        Args:
            name: Scene name.
            description: Description of the scene.
            path_to_exr: Filepath to multilayer EXR.
            return_non_optimized_lights_layer: If True, computes residual non-optimized pass.
            device: PyTorch device.

        Raises:
            ValueError: If no optimizable light layers are found.
        """
        super().__init__(name, description, device=device)
        self.optimizable_images, self.non_optimized_lights_tensor, self.light_name_list = get_images_tensor_from_multi_layer_exr(
            path_to_exr,
            return_non_optimized_lights_layer=return_non_optimized_lights_layer,
            device=device,
        )
        if len(self.optimizable_images) < 1:
            raise ValueError(f"No optimizable images found in EXR file: {path_to_exr}")

    def get_optimizable_images(self) -> torch.Tensor:
        """Return stack of optimizable light layer tensors (N, H, W, C)."""
        return self.optimizable_images

    def get_light_name_list(self) -> list[str] | None:
        """Return list of layer names in the EXR."""
        return self.light_name_list

    def get_non_optimized_lights(self) -> torch.Tensor | None:
        """Return non-optimized residual layer tensor if computed."""
        return self.non_optimized_lights_tensor
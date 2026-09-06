"""Pre-configured example OLAT scenes downloaded from Hugging Face.

You can test this framework with your own 3D scenes. The framework supports OLATs stored in two structures, but could easily be extended to support other formats.

### 1. Per-Light EXR Directory (used by `OLATDirScene`)
Per-light scenes must contain an `optimizable_lights/` subdirectory with individual `.exr` passes for each light to be optimized. Any non-optimizable background or base light pass (e.g., `base_lighting.exr`) must be placed directly in the root scene directory, not inside `optimizable_lights/`.

```text
my_scene/
├── base_lighting.exr       (optional non-optimized light pass)
├── alpha.exr               (optional alpha mask)
└── optimizable_lights/     (directory containing ONLY optimizable light passes)
    ├── light_pass_001.exr
    ├── light_pass_002.exr
    └── ...
```

### 2. Multi-Layer EXR File (used by `MultiLayerEXRScene`)
Multi-layer EXR scenes store all light passes within a single `.exr` file, where per-light channel names each contain the substring `'LGT'`.

"""

from pathlib import Path

from utils.scene import MultiLayerEXRScene, OLATDirScene

BASE_DIR = Path(__file__).resolve().parent
LOCAL_EXAMPLE_OLATS_DIR = BASE_DIR / "EXAMPLE_OLATS"
HF_BUCKET_ID = "AnsonSavage/DemoOLATScenes"
REMOTE_PREFIX = "example_olats"


class OlatCacheManager:
    """Manages downloading and caching demo OLAT scenes from Hugging Face."""

    _download_attempted = False

    @staticmethod
    def _has_any_files(directory: Path) -> bool:
        """Check if a directory exists and contains files."""
        return directory.exists() and any(directory.iterdir())

    @staticmethod
    def _download_example_olats_from_hf() -> None:
        """Download all example OLAT scene files from Hugging Face bucket."""
        if OlatCacheManager._download_attempted or OlatCacheManager._has_any_files(LOCAL_EXAMPLE_OLATS_DIR):
            return
        OlatCacheManager._download_attempted = True

        from huggingface_hub import download_bucket_files, list_bucket_tree

        file_pairs = []
        items = list_bucket_tree(HF_BUCKET_ID, prefix=REMOTE_PREFIX)
        for item in items:
            if item.type == "file":
                remote_path = item.path
                # Map remote prefix (e.g. 'example_olats/...') to uppercase LOCAL_EXAMPLE_OLATS_DIR
                if remote_path.startswith(REMOTE_PREFIX + "/") or remote_path.startswith(REMOTE_PREFIX + "\\"):
                    rel_subpath = remote_path[len(REMOTE_PREFIX) + 1:]
                else:
                    rel_subpath = remote_path
                local_path = LOCAL_EXAMPLE_OLATS_DIR / rel_subpath
                file_pairs.append((remote_path, local_path))

        if not file_pairs:
            raise FileNotFoundError(
                f"No files found in Hugging Face bucket '{HF_BUCKET_ID}' under prefix '{REMOTE_PREFIX}'."
            )

        download_bucket_files(HF_BUCKET_ID, files=file_pairs)

    @staticmethod
    def _ensure_example_olats_available() -> None:
        """Ensure example OLAT assets exist locally, downloading if necessary.

        Raises:
            FileNotFoundError: If assets are missing and downloading fails.
        """
        if OlatCacheManager._has_any_files(LOCAL_EXAMPLE_OLATS_DIR):
            return
        try:
            OlatCacheManager._download_example_olats_from_hf()
        except Exception as e:
            raise FileNotFoundError(
                f"Could not find example OLATs locally in {LOCAL_EXAMPLE_OLATS_DIR} "
                f"and failed to download them from Hugging Face: {e}"
            ) from e

        if not OlatCacheManager._has_any_files(LOCAL_EXAMPLE_OLATS_DIR):
            raise FileNotFoundError(
                f"Download completed but no files were found in {LOCAL_EXAMPLE_OLATS_DIR}."
            )


def _scene_config_dir(scene_kind: str, scene_name: str, configuration: str = "standard") -> str:
    """Get absolute directory path for a per-light directory scene configuration.

    Args:
        scene_kind: Kind of scene ('per_light' or 'multilayer').
        scene_name: Name of the scene folder.
        configuration: Subfolder configuration name.

    Returns:
        String path to the scene directory.
    """
    OlatCacheManager._ensure_example_olats_available()
    return str(LOCAL_EXAMPLE_OLATS_DIR / scene_kind / scene_name / configuration)


def _scene_multilayer_file(scene_name: str, configuration: str, filename: str) -> str:
    """Get absolute path to a multilayer EXR scene file.

    Args:
        scene_name: Name of the multilayer scene.
        configuration: Configuration subfolder.
        filename: Name of the EXR file.

    Returns:
        String path to the multilayer EXR file.
    """
    OlatCacheManager._ensure_example_olats_available()
    return str(LOCAL_EXAMPLE_OLATS_DIR / "multilayer" / scene_name / configuration / filename)


LOCAL_EXAMPLE_REFERENCE_IMAGES_DIR = BASE_DIR / "EXAMPLE_REFERENCE_IMAGES"


def get_example_reference_image_path(filename: str) -> str:
    """Get absolute path to an example reference image.

    Args:
        filename: Name of the reference image file in EXAMPLE_REFERENCE_IMAGES.

    Returns:
        String path to the reference image file.

    Raises:
        FileNotFoundError: If the reference image file does not exist.
    """
    image_path = LOCAL_EXAMPLE_REFERENCE_IMAGES_DIR / filename
    if not image_path.exists():
        raise FileNotFoundError(
            f"Example reference image '{filename}' not found at: {image_path}"
        )
    return str(image_path)


class SpringScene(MultiLayerEXRScene):
    """Spring scene: A girl and dog running in the mountains (multilayer EXR)."""

    def __init__(self, device: str = "cuda"):
        """Initialize SpringScene.

        Args:
            device: PyTorch device.
        """
        path_to_exr = _scene_multilayer_file("spring", "standard", "spring_all_lights.exr")
        super().__init__("spring", "A girl and dog running in the mountains", path_to_exr, device=device)


class SciFiRobotScene(OLATDirScene):
    """SciFi Robot scene: A sci-fi robot against a brick wall."""

    def __init__(self, include_alpha_mask: bool = False, device: str = "cuda"):
        """Initialize SciFiRobotScene.

        Args:
            include_alpha_mask: Whether to load and apply alpha mask.
            device: PyTorch device.
        """
        path_to_olat_dir = _scene_config_dir("per_light", "scifi_armor", "cube_sphere_with_base_lighting")
        super().__init__(
            "scifiRobot",
            "A sci-fi robot against a brick wall",
            path_to_olat_dir,
            "base_lighting.exr",
            include_alpha_mask=include_alpha_mask,
            device=device,
        )


class BlenderManScene(OLATDirScene):
    """BlenderMan scene: A man in a robot suit."""

    def __init__(self, device: str = "cuda"):
        """Initialize BlenderManScene.

        Args:
            device: PyTorch device.
        """
        path_to_olat_dir = _scene_config_dir("per_light", "blenderman", "cube_sphere_with_base_lighting")
        super().__init__("blenderman", "A man in a robot suit", path_to_olat_dir, "base_lighting.exr", device=device)


class CarScene(OLATDirScene):
    """Car scene: A Volkswagen beetle car."""

    def __init__(self, device: str = "cuda"):
        """Initialize CarScene.

        Args:
            device: PyTorch device.
        """
        path_to_olat_dir = _scene_config_dir("per_light", "rendered_lights_car", "dome_area_lights")
        super().__init__("car", "a Volkswagen beetle car", path_to_olat_dir, device=device)


class RedCarScene(OLATDirScene):
    """Red Car scene: A red sports car."""

    def __init__(self, include_alpha_mask: bool = False, device: str = "cuda"):
        """Initialize RedCarScene.

        Args:
            include_alpha_mask: Whether to load and apply alpha mask.
            device: PyTorch device.
        """
        path_to_olat_dir = _scene_config_dir("per_light", "red_car")
        super().__init__("redCar", "a red sports car", path_to_olat_dir, include_alpha_mask=include_alpha_mask, device=device)


class CandleScene(OLATDirScene):
    """Candle scene: A candle on a table."""

    def __init__(self, include_alpha_mask: bool = False, device: str = "cuda"):
        """Initialize CandleScene.

        Args:
            include_alpha_mask: Whether to load and apply alpha mask.
            device: PyTorch device.
        """
        path_to_olat_dir = _scene_config_dir("per_light", "candle")
        super().__init__("candle", "a candle on a table", path_to_olat_dir, include_alpha_mask=include_alpha_mask, device=device)


class HouseScene(OLATDirScene):
    """House scene: An interior living room."""

    def __init__(self, device: str = "cuda"):
        """Initialize HouseScene.

        Args:
            device: PyTorch device.
        """
        path_to_olat_dir = _scene_config_dir("per_light", "house")
        super().__init__("house", "an interior of a living room", path_to_olat_dir, device=device)


class DinoScene(OLATDirScene):
    """Dino scene: A dinosaur in the woods."""

    def __init__(self, device: str = "cuda"):
        """Initialize DinoScene.

        Args:
            device: PyTorch device.
        """
        path_to_olat_dir = _scene_config_dir("per_light", "dino")
        super().__init__("dino", "a dinosaur in the woods", path_to_olat_dir, device=device)


class FlowerPotScene(OLATDirScene):
    """Flower Pot scene: A flower pot on the street."""

    def __init__(self, device: str = "cuda"):
        """Initialize FlowerPotScene.

        Args:
            device: PyTorch device.
        """
        path_to_olat_dir = _scene_config_dir("per_light", "flower_pot")
        super().__init__("flowerPot", "a flower pot on the street", path_to_olat_dir, device=device)


class CarStudioScene(OLATDirScene):
    """Car Studio scene: A car in a studio lighting setup."""

    def __init__(self, configuration: str = "dome_lights", device: str = "cuda"):
        """Initialize CarStudioScene.

        Args:
            configuration: Configuration name ('dome_lights', 'four_small_area_lights', or 'single_sun_light').
            device: PyTorch device.
        """
        assert configuration in ("dome_lights", "four_small_area_lights", "single_sun_light"), "Invalid configuration for CarStudioScene"
        path_to_olat_dir = _scene_config_dir("per_light", "car_studio", configuration)
        super().__init__("carStudio", "a car in a studio setup", path_to_olat_dir, "base_lighting.exr", device=device)


class EinarScene(OLATDirScene):
    """Einar scene: An old man with a beard against a mountain background."""

    def __init__(self, device: str = "cuda"):
        """Initialize EinarScene.

        Args:
            device: PyTorch device.
        """
        path_to_olat_dir = _scene_config_dir("per_light", "einar")
        super().__init__("einar", "an old man with a beard against a mountain background", path_to_olat_dir, device=device)


class EinarSmallDomeScene(OLATDirScene):
    """Einar Small Dome scene: An old man lit with a smaller dome setup."""

    def __init__(self, device: str = "cuda"):
        """Initialize EinarSmallDomeScene.

        Args:
            device: PyTorch device.
        """
        path_to_olat_dir = _scene_config_dir("per_light", "einar_small_dome")
        super().__init__("einarSmallDome", "an old man with a beard against a mountain background", path_to_olat_dir, device=device)


class SpringPortraitScene(OLATDirScene):
    """Spring Portrait scene: A 3D stylized portrait of a girl."""

    def __init__(self, device: str = "cuda"):
        """Initialize SpringPortraitScene.

        Args:
            device: PyTorch device.
        """
        path_to_exr = _scene_config_dir("per_light", "spring_portrait")
        super().__init__("springPortrait", "A 3D stylized portrait of a girl", path_to_exr, device=device)


class SpringPortraitSmallDomeScene(OLATDirScene):
    """Spring Portrait Small Dome scene: A 3D stylized portrait lit by a smaller dome."""

    def __init__(self, device: str = "cuda"):
        """Initialize SpringPortraitSmallDomeScene.

        Args:
            device: PyTorch device.
        """
        path_to_exr = _scene_config_dir("per_light", "spring_portrait_small_dome")
        super().__init__("springPortraitSmallDome", "A 3D stylized portrait of a girl lit by a set of planes configured in a smaller dome", path_to_exr, device=device)


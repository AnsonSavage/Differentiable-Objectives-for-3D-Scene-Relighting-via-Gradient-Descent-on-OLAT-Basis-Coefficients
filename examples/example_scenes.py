"""Pre-configured example OLAT scenes downloaded from Hugging Face."""
from pathlib import Path

from utils.scene import MultiLayerEXRScene, OLATDirScene

BASE_DIR = Path(__file__).resolve().parent
LOCAL_EXAMPLE_OLATS_DIR = BASE_DIR / "EXAMPLE_OLATS"
HF_BUCKET_ID = "AnsonSavage/DemoOLATScenes"


class OlatCacheManager:
    """Manages downloading and caching demo OLAT scenes from Hugging Face."""

    @staticmethod
    def _is_available(path: Path) -> bool:
        """Check if target path exists and is populated."""
        if not path.exists():
            return False
        if path.is_file():
            return path.stat().st_size > 0
        return any(path.iterdir())

    @staticmethod
    def _download_scene_from_hf(pattern: str) -> None:
        """Download scene files matching a pattern from Hugging Face.

        Args:
            pattern: Glob pattern relative to example_olats directory on HF.

        Raises:
            FileNotFoundError: If download fails or files cannot be retrieved.
        """
        import shutil

        from huggingface_hub import snapshot_download

        try:
            snapshot_download(
                repo_id=HF_BUCKET_ID,
                repo_type="dataset",
                allow_patterns=[f"example_olats/{pattern}"],
                local_dir=str(BASE_DIR),
            )
            lower_dir = BASE_DIR / "example_olats"
            if lower_dir.exists():
                for item in lower_dir.rglob("*"):
                    if item.is_file():
                        rel = item.relative_to(lower_dir)
                        dest = LOCAL_EXAMPLE_OLATS_DIR / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(item), str(dest))
                shutil.rmtree(str(lower_dir), ignore_errors=True)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not download scene files for '{pattern}' from Hugging Face ({HF_BUCKET_ID}): {e}"
            ) from e

    @staticmethod
    def resolve_scene_dir(scene_kind: str, scene_name: str, configuration: str | None = None) -> str:
        """Resolve and ensure local availability of a per-light directory scene.

        Args:
            scene_kind: Kind of scene ('per_light' or 'multilayer').
            scene_name: Name of the scene folder.
            configuration: Optional subfolder configuration name.

        Returns:
            String path to the local scene directory.
        """
        subpath = f"{scene_kind}/{scene_name}/{configuration}" if configuration else f"{scene_kind}/{scene_name}"
        target_path = LOCAL_EXAMPLE_OLATS_DIR / subpath

        if not OlatCacheManager._is_available(target_path):
            OlatCacheManager._download_scene_from_hf(f"{subpath}/*")

        return str(target_path)

    @staticmethod
    def resolve_scene_file(scene_kind: str, scene_name: str, configuration: str, filename: str) -> str:
        """Resolve and ensure local availability of a specific scene file.

        Args:
            scene_kind: Kind of scene ('per_light' or 'multilayer').
            scene_name: Name of the scene folder.
            configuration: Configuration subfolder name.
            filename: Target file name.

        Returns:
            String path to the local scene file.
        """
        subpath = f"{scene_kind}/{scene_name}/{configuration}/{filename}"
        target_path = LOCAL_EXAMPLE_OLATS_DIR / subpath

        if not OlatCacheManager._is_available(target_path):
            OlatCacheManager._download_scene_from_hf(subpath)

        return str(target_path)


def _scene_config_dir(scene_kind: str, scene_name: str, configuration: str | None = None) -> str:
    """Get absolute directory path for a scene configuration, downloading on demand.

    Args:
        scene_kind: Kind of scene ('per_light' or 'multilayer').
        scene_name: Name of the scene folder.
        configuration: Optional subfolder configuration name.

    Returns:
        String path to the scene directory.
    """
    return OlatCacheManager.resolve_scene_dir(scene_kind, scene_name, configuration)


def _scene_multilayer_file(scene_name: str, configuration: str, filename: str) -> str:
    """Get absolute path to a multilayer EXR scene file, downloading on demand.

    Args:
        scene_name: Name of the multilayer scene.
        configuration: Configuration subfolder.
        filename: Name of the EXR file.

    Returns:
        String path to the multilayer EXR file.
    """
    return OlatCacheManager.resolve_scene_file("multilayer", scene_name, configuration, filename)


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


from __future__ import annotations

from pathlib import Path

from utils.scene import MultiLayerEXRScene, OLATDirScene

BASE_DIR = Path(__file__).resolve().parent
LOCAL_EXAMPLE_OLATS_DIR = BASE_DIR / "EXAMPLE_OLATS"
HF_BUCKET_ID = "AnsonSavage/DemoOLATScenes"


class OlatCacheManager:
    @staticmethod
    def _is_available(path: Path) -> bool:
        if not path.exists():
            return False
        if path.is_file():
            return path.stat().st_size > 0
        return any(path.iterdir())

    @staticmethod
    def _download_scene_from_hf(pattern: str) -> None:
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
        subpath = f"{scene_kind}/{scene_name}/{configuration}" if configuration else f"{scene_kind}/{scene_name}"
        target_path = LOCAL_EXAMPLE_OLATS_DIR / subpath

        if not OlatCacheManager._is_available(target_path):
            OlatCacheManager._download_scene_from_hf(f"{subpath}/*")

        return str(target_path)

    @staticmethod
    def resolve_scene_file(scene_kind: str, scene_name: str, configuration: str, filename: str) -> str:
        subpath = f"{scene_kind}/{scene_name}/{configuration}/{filename}"
        target_path = LOCAL_EXAMPLE_OLATS_DIR / subpath

        if not OlatCacheManager._is_available(target_path):
            OlatCacheManager._download_scene_from_hf(subpath)

        return str(target_path)


def _scene_config_dir(scene_kind: str, scene_name: str, configuration: str | None = None) -> str:
    return OlatCacheManager.resolve_scene_dir(scene_kind, scene_name, configuration)


def _scene_multilayer_file(scene_name: str, configuration: str, filename: str) -> str:
    return OlatCacheManager.resolve_scene_file("multilayer", scene_name, configuration, filename)


class SpringScene(MultiLayerEXRScene):
    def __init__(self, device: str = "cuda"):
        path_to_exr = _scene_multilayer_file("spring", "standard", "spring_all_lights.exr")
        super().__init__("spring", "A girl and dog running in the mountains", path_to_exr, device=device)


class SciFiRobotScene(OLATDirScene):
    def __init__(self, include_alpha_mask: bool = False, device: str = "cuda"):
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
    def __init__(self, device: str = "cuda"):
        path_to_olat_dir = _scene_config_dir("per_light", "blenderman", "cube_sphere_with_base_lighting")
        super().__init__("blenderman", "A man in a robot suit", path_to_olat_dir, "base_lighting.exr", device=device)


class CarScene(OLATDirScene):
    def __init__(self, device: str = "cuda"):
        path_to_olat_dir = _scene_config_dir("per_light", "rendered_lights_car", "dome_area_lights")
        super().__init__("car", "a Volkswagen beetle car", path_to_olat_dir, device=device)


class RedCarScene(OLATDirScene):
    def __init__(self, include_alpha_mask: bool = False, device: str = "cuda"):
        path_to_olat_dir = _scene_config_dir("per_light", "red_car")
        super().__init__("redCar", "a red sports car", path_to_olat_dir, include_alpha_mask=include_alpha_mask, device=device)


class CandleScene(OLATDirScene):
    def __init__(self, include_alpha_mask: bool = False, device: str = "cuda"):
        path_to_olat_dir = _scene_config_dir("per_light", "candle")
        super().__init__("candle", "a candle on a table", path_to_olat_dir, include_alpha_mask=include_alpha_mask, device=device)


class HouseScene(OLATDirScene):
    def __init__(self, device: str = "cuda"):
        path_to_olat_dir = _scene_config_dir("per_light", "house")
        super().__init__("house", "an interior of a living room", path_to_olat_dir, device=device)


class DinoScene(OLATDirScene):
    def __init__(self, device: str = "cuda"):
        path_to_olat_dir = _scene_config_dir("per_light", "dino")
        super().__init__("dino", "a dinosaur in the woods", path_to_olat_dir, device=device)


class FlowerPotScene(OLATDirScene):
    def __init__(self, device: str = "cuda"):
        path_to_olat_dir = _scene_config_dir("per_light", "flower_pot")
        super().__init__("flowerPot", "a flower pot on the street", path_to_olat_dir, device=device)


class CarStudioScene(OLATDirScene):
    def __init__(self, configuration: str = "dome_lights", device: str = "cuda"):
        assert configuration in ("dome_lights", "four_small_area_lights", "single_sun_light"), "Invalid configuration for CarStudioScene"
        path_to_olat_dir = _scene_config_dir("per_light", "car_studio", configuration)
        super().__init__("carStudio", "a car in a studio setup", path_to_olat_dir, "base_lighting.exr", device=device)


class EinarScene(OLATDirScene):
    def __init__(self, device: str = "cuda"):
        path_to_olat_dir = _scene_config_dir("per_light", "einar")
        super().__init__("einar", "an old man with a beard against a mountain background", path_to_olat_dir, device=device)


class EinarSmallDomeScene(OLATDirScene):
    def __init__(self, device: str = "cuda"):
        path_to_olat_dir = _scene_config_dir("per_light", "einar_small_dome")
        super().__init__("einarSmallDome", "an old man with a beard against a mountain background", path_to_olat_dir, device=device)


class SpringPortraitScene(OLATDirScene):
    def __init__(self, device: str = "cuda"):
        path_to_exr = _scene_config_dir("per_light", "spring_portrait")
        super().__init__("springPortrait", "A 3D stylized portrait of a girl", path_to_exr, device=device)


class SpringPortraitSmallDomeScene(OLATDirScene):
    def __init__(self, device: str = "cuda"):
        path_to_exr = _scene_config_dir("per_light", "spring_portrait_small_dome")
        super().__init__("springPortraitSmallDome", "A 3D stylized portrait of a girl lit by a set of planes configured in a smaller dome", path_to_exr, device=device)

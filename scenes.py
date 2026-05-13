from utils.scene import MultiLayerEXRScene, OLATDirScene
from pathlib import Path

# Directory containing this scenes.py file. Use this so paths are resolved relative
# to the module location (works when importing from a notebook) instead of the
# current working directory.
BASE_DIR = Path(__file__).resolve().parent

def get_full_path(*parts) -> str:
    """Return a path string relative to the `scenes.py` file.
    """
    return str(BASE_DIR.joinpath(*parts))

class SpringScene(MultiLayerEXRScene):
    def __init__(self, device: str = 'cuda'):
        path_to_exr = get_full_path('render', 'multilayer', 'spring', 'spring_all_lights.exr')
        super().__init__("spring", "A girl and dog running in the mountains", path_to_exr, device=device)

class SciFiRobotScene(OLATDirScene):
    def __init__(self, include_alpha_mask: bool = False, device: str = 'cuda'):
        path_to_olat_dir = get_full_path('render', 'per_light', 'scifi_armor', 'cube_sphere_with_base_lighting')
        base_lighting_file = "base_lighting.exr"
        # For this specific scene, the alpha.exr is one directory above the OLAT lights directory
        alpha_mask_path = get_full_path('render', 'per_light', 'scifi_armor', 'alpha.exr') if include_alpha_mask else None
        super().__init__(
            "scifiRobot",
            "A sci-fi robot against a brick wall",
            path_to_olat_dir,
            base_lighting_file,
            include_alpha_mask=include_alpha_mask,
            alpha_mask_path=alpha_mask_path,
            device=device,
        )

class BlenderManScene(OLATDirScene):
    def __init__(self, device: str = 'cuda'):
        path_to_olat_dir = get_full_path('render', 'per_light', 'blenderman', 'cube_sphere_with_base_lighting')
        base_lighting_file = "base_lighting.exr"
        super().__init__("blenderman", "A man in a robot suit", path_to_olat_dir, base_lighting_file, device=device)

class CarScene(OLATDirScene):
    def __init__(self, device: str = 'cuda'):
        path_to_olat_dir = get_full_path('render', 'per_light', 'rendered_lights_car', 'dome_area_lights')
        super().__init__("car", "a Volkswagen beetle car", path_to_olat_dir, device=device)

class RedCarScene(OLATDirScene):
    def __init__(self, include_alpha_mask: bool = False, device: str = 'cuda'):
        path_to_olat_dir = get_full_path('render', 'per_light', 'red_car', 'optimizable_lights')
        alpha_mask_path = get_full_path('render', 'per_light', 'red_car', 'alpha.exr') if include_alpha_mask else None
        super().__init__("redCar", "a red sports car", path_to_olat_dir, alpha_mask_path=alpha_mask_path, device=device)

class CandleScene(OLATDirScene):
    def __init__(self, include_alpha_mask: bool = False, device: str = 'cuda'):
        path_to_olat_dir = get_full_path('render', 'per_light', 'candle', 'optimizable_lights')
        alpha_mask_path = get_full_path('render', 'per_light', 'candle', 'alpha.exr') if include_alpha_mask else None
        super().__init__("candle", "a candle on a table", path_to_olat_dir, alpha_mask_path=alpha_mask_path, device=device)

class HouseScene(OLATDirScene):
    def __init__(self, device: str = 'cuda'):
        path_to_olat_dir = get_full_path('render', 'per_light', 'house')
        super().__init__("house", "an interior of a living room", path_to_olat_dir, device=device)

class DinoScene(OLATDirScene):
    def __init__(self, device: str = 'cuda'):
        path_to_olat_dir = get_full_path('render', 'per_light', 'dino')
        super().__init__("dino", "a dinosaur in the woods", path_to_olat_dir, device=device)

class FlowerPotScene(OLATDirScene):
    def __init__(self, device: str = 'cuda'):
        path_to_olat_dir = get_full_path('render', 'per_light', 'flower_pot')
        super().__init__("flowerPot", "a flower pot on the street", path_to_olat_dir, device=device)

class CarStudioScene(OLATDirScene):
    def __init__(self, configuration: str = 'dome_lights', device: str = 'cuda'):
        assert configuration in ('dome_lights', 'four_small_area_lights', 'single_sun_light'), "Invalid configuration for CarStudioScene"
        path_to_olat_dir = get_full_path('render', 'per_light', 'car_studio', configuration)
        super().__init__("carStudio", "a car in a studio setup", path_to_olat_dir, 'non_optimized_lights.exr', device=device)

class EinarScene(OLATDirScene):
    def __init__(self, device: str = 'cuda'):
        path_to_olat_dir = get_full_path('render', 'per_light', 'einar')
        super().__init__("einar", "an old man with a beard against a mountain background", path_to_olat_dir, device=device)

class EinarSmallDomeScene(OLATDirScene):
    def __init__(self, device: str = 'cuda'):
        path_to_olat_dir = get_full_path('render', 'per_light', 'einar_small_dome')
        super().__init__("einarSmallDome", "an old man with a beard against a mountain background", path_to_olat_dir, device=device)

class SpringPortraitScene(OLATDirScene):
    def __init__(self, device: str = 'cuda'):
        path_to_exr = get_full_path('render', 'per_light', 'spring_portrait')
        super().__init__("springPortrait", "A 3D stylized portrait of a girl", path_to_exr, device=device)

class SpringPortraitSmallDomeScene(OLATDirScene):
    def __init__(self, device: str = 'cuda'):
        path_to_exr = get_full_path('render', 'per_light', 'spring_portrait_small_dome')
        super().__init__("springPortraitSmallDome", "A 3D stylized portrait of a girl lit by a set of planes configured in a smaller dome", path_to_exr, device=device)
"""
Utilities for loading OLAT scenes from EXRs into PyTorch tensors with pluggable backends.

Provides:
- get_images_tensor_from_OLAT_dir: load per-light EXR images from a directory.
- get_images_tensor_from_multi_layer_exr: load per-light layers from a multilayer EXR.
- load_alpha_tensor: load only the alpha channel as a tensor.

Returned image tensors use channel-last layout (H, W, C) and stacks of lights
use (N, H, W, C).
"""

import os
from typing import Optional, Protocol
try:
    import OpenEXR  # type: ignore  # Lightweight EXR I/O
    import Imath  # type: ignore
except Exception:
    OpenEXR = None
    Imath = None
import numpy as np
import torch
import glob

# Backend strategy interfaces and implementations
class EXRReader(Protocol):
    def read_rgb(self, path: str, numpy_precision=np.float32) -> np.ndarray: ...
    def read_alpha(self, path: str, numpy_precision=np.float32) -> np.ndarray: ...
    def read_multilayer(self, path: str, light_layer_keyword: str, numpy_precision=np.float32) -> tuple[list[str], np.ndarray]: ...

class OpenEXRReader:
    def __init__(self):
        if OpenEXR is None or Imath is None:
            raise ImportError("OpenEXR/Imath not available. Install 'OpenEXR' and 'Imath'.")

    def read_rgb(self, path: str, numpy_precision=np.float32) -> np.ndarray:
        """Read RGB from an EXR file using OpenEXR.

        Returns an array shaped (H, W, 3) in the requested numpy_precision.
        """
        if OpenEXR is None or Imath is None:
            raise ImportError("OpenEXR/Imath not available. Install 'OpenEXR' and 'Imath' to use the OpenEXR backend.")
        f = OpenEXR.InputFile(path)
        header = f.header()
        dw = header['dataWindow']
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        chs = header['channels'].keys()

        def find_chan(name: str) -> Optional[str]:
            if name in chs:
                return name
            # layered naming like 'layer.R'
            for k in chs:
                if k.endswith('.' + name):
                    return k
            # lowercase fallback
            for k in chs:
                if k.lower() == name.lower():
                    return k
            return None

        r_name, g_name, b_name = find_chan('R'), find_chan('G'), find_chan('B')
        if r_name is None or g_name is None or b_name is None:
            raise ValueError(f"EXR at {path} is missing RGB channels")
        r, g, b = f.channels([r_name, g_name, b_name], pt)
        r_arr = np.frombuffer(r, dtype=np.float32).reshape(h, w)
        g_arr = np.frombuffer(g, dtype=np.float32).reshape(h, w)
        b_arr = np.frombuffer(b, dtype=np.float32).reshape(h, w)
        rgb = np.stack([r_arr, g_arr, b_arr], axis=-1).astype(numpy_precision, copy=False)
        return rgb

    def read_alpha(self, path: str, numpy_precision=np.float32) -> np.ndarray:
        """Read alpha channel from an EXR as (H, W, 1) using OpenEXR.
        Raises ValueError if alpha not present.
        """
        if OpenEXR is None or Imath is None:
            raise ImportError("OpenEXR/Imath not available. Install 'OpenEXR' and 'Imath' to use the OpenEXR backend.")
        f = OpenEXR.InputFile(path)
        header = f.header()
        dw = header['dataWindow']
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        chs = header['channels'].keys()

        def find_alpha() -> Optional[str]:
            if 'A' in chs:
                return 'A'
            for k in chs:
                if k.endswith('.A'):
                    return k
            return None

        a_name = find_alpha()
        if a_name is None:
            raise ValueError(f"Alpha channel not found in EXR: {path}")
        a = f.channels([a_name], pt)[0]
        a_arr = np.frombuffer(a, dtype=np.float32).reshape(h, w).astype(numpy_precision, copy=False)
        return a_arr[..., None]

    def read_multilayer(self, path: str, light_layer_keyword: str, numpy_precision=np.float32):
        raise NotImplementedError("OpenEXR backend does not support multi-layer grouping; use MitsubaBackend.")

class MitsubaEXRReader:
    def __init__(self):
        import mitsuba as mi  # type: ignore  # Heavy but robust multi-layer support
        self.mi = mi
        if self.mi is None:
            raise ImportError("Mitsuba not available. Install 'mitsuba'.")

    def read_rgb(self, path: str, numpy_precision=np.float32) -> np.ndarray:
        bmp = self.mi.Bitmap(path)
        return np.array(bmp).astype(numpy_precision)[:, :, 0:3]

    def read_alpha(self, path: str, numpy_precision=np.float32) -> np.ndarray:
        bmp = self.mi.Bitmap(path)
        alpha_np = np.array(bmp).astype(numpy_precision)
        if alpha_np.shape[2] < 4:
            raise ValueError(f"Alpha EXR at {path} does not have a valid alpha channel.")
        return alpha_np[:, :, 3:4]

    def read_multilayer(self, path: str, light_layer_keyword: str, numpy_precision=np.float32):
        multi_exr = self.mi.Bitmap(path)
        layers = [str(layer.name) for layer in multi_exr.struct_()]
        dict_light_name_to_start_index = {}
        for layer in layers:
            if light_layer_keyword in layer and layer.endswith('.R'):
                light_name = layer[:-2]
                dict_light_name_to_start_index[light_name] = layers.index(layer)
        image_np = np.array(multi_exr).astype(numpy_precision)
        optimized_layer_names = list(dict_light_name_to_start_index.keys())
        images_list = [image_np[:, :, dict_light_name_to_start_index[name]:dict_light_name_to_start_index[name] + 3]
                       for name in optimized_layer_names]
        if len(images_list) > 0:
            images_np = np.stack(images_list, axis=0)  # (N, H, W, C)
        else:
            h, w, _ = image_np.shape
            images_np = np.empty((0, h, w, 3), dtype=image_np.dtype)
        return optimized_layer_names, images_np

def select_exr_reader_implementation(preferred: str) -> EXRReader:
    preferred = (preferred or '').lower()
    if preferred == 'openexr':
        try:
            return OpenEXRReader()
        except Exception:
            return MitsubaEXRReader()
    if preferred == 'mitsuba':
        try:
            return MitsubaEXRReader()
        except Exception:
            return OpenEXRReader()
    # Default preference for simple reads
    try:
        return OpenEXRReader()
    except Exception:
        return MitsubaEXRReader()

def get_images_tensor_from_OLAT_dir(path_to_olat_dir, name_of_non_optimized_lights_layer = None, file_pattern="*.exr", device='cuda', numpy_precision=np.float32, torch_precision=torch.float32, backend: str = 'openexr'):
    """
    Load a stack of per-light images from a directory of EXR files.

    - Files are globbed using file_pattern and sorted lexicographically.
    - RGB channels are kept (alpha discarded).
    - Optionally extracts one image (by filename) as a "non-optimized lights" layer
      and removes it from the optimized stack.

    Args:
        path_to_olat_dir: Directory containing per-light EXR images.
        name_of_non_optimized_lights_layer: Optional base filename to be separated
            out as the non-optimized lights layer (e.g., 'env.exr'). If found,
            it will be returned separately and excluded from the stack.
        file_pattern: Glob pattern for files (default '*.exr').
        device: Torch device for returned tensors (e.g., 'cuda' or 'cpu').
        numpy_precision: NumPy dtype used when reading.
        torch_precision: Torch dtype for output tensors.
        backend: 'openexr' (default) for lightweight I/O, or 'mitsuba'. Falls back
            to the available one if the preferred backend is not available.

    Returns:
        images_tensor: Tensor of shape (N, H, W, 3) with per-light images.
        non_optimized_lights_tensor: Tensor of shape (H, W, 3) if extracted,
            otherwise None.
        sorted_files: List of file paths corresponding to images_tensor order
            (with the non-optimized file removed if extracted).

    Notes:
        - Sorting determines light order; keep track of 'sorted_files' to map
          multipliers back to filenames.
        - Channels are in linear space as provided by Mitsuba's Bitmap loader.
    """
    file_pattern = os.path.join(path_to_olat_dir, file_pattern)
    file_paths = glob.glob(file_pattern)
    images_list = []
    sorted_files = sorted(file_paths)
    index_of_non_optimized_lights_layer = -1
    exr_reader = select_exr_reader_implementation(backend)
    for i, file_path in enumerate(sorted_files):
        # Check and see if the base name of the file path is the name of the non-optimized lights layer
        image_np: np.ndarray = exr_reader.read_rgb(file_path, numpy_precision=numpy_precision)
        images_list.append(image_np)
        if name_of_non_optimized_lights_layer is not None:
            base_name = os.path.basename(file_path)
            if base_name == name_of_non_optimized_lights_layer:
                index_of_non_optimized_lights_layer = i
    
    non_optimized_lights_tensor = None
    if index_of_non_optimized_lights_layer != -1:
        # Extract the non-optimized layer efficiently from NumPy
        non_opt_np = images_list[index_of_non_optimized_lights_layer]
        non_optimized_lights_tensor = torch.from_numpy(non_opt_np).to(device=device, dtype=torch_precision)
        # Remove the non-optimized lights layer from the images list
        images_list.pop(index_of_non_optimized_lights_layer)
        sorted_files.pop(index_of_non_optimized_lights_layer)
    # Stack list of NumPy arrays before converting to Torch (avoids slow path & warning)
    assert len(images_list) > 0, "No images found in the specified directory with the given pattern."
    images_np = np.stack(images_list, axis=0)  # (N, H, W, C)
    images_tensor = torch.from_numpy(images_np).to(device=device, dtype=torch_precision)

    return images_tensor, non_optimized_lights_tensor, sorted_files

def get_images_tensor_from_multi_layer_exr(path_to_exr, return_non_optimized_lights_layer=False, device='cuda', light_layer_keyword = 'LGT', numpy_precision=np.float32, torch_precision=torch.float32, backend: str = 'mitsuba'):
    """
    Load per-light layers from a multilayer EXR into a stack.

    Detects light layers by name using a keyword (default 'LGT') and groups
    consecutive .R/.G/.B channels into one RGB image per light. The order of
    lights follows the EXR's layer listing.

    Args:
        path_to_exr: Path to the multilayer EXR.
        return_non_optimized_lights_layer: If True, computes a residual by
            subtracting the sum of optimized lights from the first 3 channels
            of the EXR (assumed to be the full image).
        device: Torch device for returned tensors.
        light_layer_keyword: Substring that identifies light layers (e.g., 'LGT').
        numpy_precision: NumPy dtype used when reading with Mitsuba.
        torch_precision: Torch dtype for output tensors.

    Returns:
        images_tensor: Tensor of shape (N, H, W, 3) with per-light images.
        non_optimized_lights_tensor: Tensor of shape (H, W, 3) if requested,
            otherwise None.
        optimized_layer_names: List of light names corresponding to images_tensor.

    Notes:
        - Non optimized lights layer computation can be noisy; consider denoising or a more robust
          separation if artifacts appear.
    """
    # Use Mitsuba backend for multilayer reads; enforce capability
    exr_reader = MitsubaEXRReader() # Other backends are untested.
    optimized_layer_names, images_np = exr_reader.read_multilayer(path_to_exr, light_layer_keyword, numpy_precision=numpy_precision)
    if images_np.size > 0:
        images_tensor = torch.from_numpy(images_np).to(device=device, dtype=torch_precision)
    else:
        # No detected light layers; create empty with correct spatial dims
        base_rgb = exr_reader.read_rgb(path_to_exr, numpy_precision=numpy_precision)
        h, w, _ = base_rgb.shape
        images_tensor = torch.empty((0, h, w, 3), device=device, dtype=torch_precision)
    non_optimized_lights_tensor = None
    if return_non_optimized_lights_layer: # Subtract the sum of the optimized lights from the full image to get the non-optimized lights.
        # TODO: This is prone to artifacts from noise and stuff, would be good to refactor
        base_full_np = exr_reader.read_rgb(path_to_exr, numpy_precision=numpy_precision)
        base_full_tensor = torch.from_numpy(base_full_np[:, :, :3]).to(device=device, dtype=torch_precision)
        if images_tensor.shape[0] > 0:
            non_optimized_lights_tensor = base_full_tensor - torch.sum(images_tensor, dim=0)
        else:
            non_optimized_lights_tensor = base_full_tensor
    return images_tensor, non_optimized_lights_tensor, optimized_layer_names # (N, H, W, C), (H, W, C), list of light names corresponding with images_tensor

def load_alpha_tensor(path_to_alpha_exr, device='cuda', numpy_precision=np.float32, torch_precision=torch.float32, backend: str = 'openexr'):
    """
    Load the alpha channel from an EXR as a (H, W, 1) tensor.

    Args:
        path_to_alpha_exr: Path to an EXR containing an alpha channel.
        device: Torch device for the returned tensor.
        numpy_precision: NumPy dtype used when reading with Mitsuba.
        torch_precision: Torch dtype for the output tensor.

    Returns:
        alpha_tensor: Tensor of shape (H, W, 1) with the alpha channel.

    Raises:
        ValueError: If the EXR does not contain a channel at index 3.
    """
    exr_reader = select_exr_reader_implementation(backend)
    alpha_np = exr_reader.read_alpha(path_to_alpha_exr, numpy_precision=numpy_precision)
    alpha_tensor = torch.from_numpy(alpha_np).to(device=device, dtype=torch_precision)
    return alpha_tensor
    
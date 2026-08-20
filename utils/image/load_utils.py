"""
Utilities for loading OLAT scenes from EXRs into PyTorch tensors.

Provides:
- get_images_tensor_from_OLAT_dir: load per-light EXR images from a directory.
- get_images_tensor_from_multi_layer_exr: load per-light layers from a multilayer EXR.
- load_alpha_tensor: load only the alpha channel as a tensor.

Returned image tensors use channel-last layout (H, W, C) and stacks of lights
use (N, H, W, C).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import Imath
import numpy as np
import OpenEXR
import torch


class OpenEXRReader:
    @staticmethod
    def _read_channels(f, channel_names: list[str], h: int, w: int, numpy_precision=np.float32) -> list[np.ndarray]:
        """Decode raw EXR channel buffers into 2D NumPy arrays of shape (H, W)."""
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        raw_buffers = f.channels(channel_names, pt)
        return [
            np.frombuffer(buf, dtype=np.float32).reshape(h, w).astype(numpy_precision, copy=False)
            for buf in raw_buffers
        ]

    def read_rgb(self, path: str, numpy_precision=np.float32) -> np.ndarray:
        """Read RGB from an EXR file using OpenEXR.

        Returns an array shaped (H, W, 3) in the requested numpy_precision.
        """
        f = OpenEXR.InputFile(path)
        header = f.header()
        dw = header['dataWindow']
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        chs = header['channels'].keys()

        def find_channel(name: str) -> str | None:
            if name in chs:
                return name
            for k in chs:
                if k.endswith('.' + name):
                    return k
            return None

        r_name, g_name, b_name = find_channel('R'), find_channel('G'), find_channel('B')
        if r_name is None or g_name is None or b_name is None:
            raise ValueError(f"EXR at {path} is missing RGB channels")
        r_arr, g_arr, b_arr = self._read_channels(f, [r_name, g_name, b_name], h, w, numpy_precision)
        return np.stack([r_arr, g_arr, b_arr], axis=-1)

    def read_alpha(self, path: str, numpy_precision=np.float32) -> np.ndarray:
        """Read alpha channel from an EXR as (H, W, 1) using OpenEXR.
        Raises ValueError if alpha not present.
        """
        f = OpenEXR.InputFile(path)
        header = f.header()
        dw = header['dataWindow']
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        chs = header['channels'].keys()

        def find_alpha() -> str | None:
            if 'A' in chs:
                return 'A'
            for k in chs:
                if k.endswith('.A'):
                    return k
            return None

        a_name = find_alpha()
        if a_name is None:
            raise ValueError(f"Alpha channel not found in EXR: {path}")
        (a_arr,) = self._read_channels(f, [a_name], h, w, numpy_precision)
        return a_arr[..., None]

    def read_multilayer(self, path: str, light_layer_keyword: str, numpy_precision=np.float32) -> tuple[list[str], np.ndarray]:
        """Read per-light layers from a multilayer EXR using OpenEXR."""
        f = OpenEXR.InputFile(path)
        header = f.header()
        dw = header['dataWindow']
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        
        channels = list(header['channels'].keys())
        
        # Group channels by light name
        dict_light_name_to_channels = {}
        for ch in channels:
            if light_layer_keyword in ch and ch.endswith('.R'):
                light_name = ch[:-2]
                dict_light_name_to_channels[light_name] = {'R': ch, 'G': ch[:-2] + '.G', 'B': ch[:-2] + '.B'}
        
        optimized_layer_names = list(dict_light_name_to_channels.keys())
        images_list = []
        for name in optimized_layer_names:
            c_names = dict_light_name_to_channels[name]
            # Ensure all RGB channels exist for this light
            if c_names['G'] not in channels or c_names['B'] not in channels:
                print(f"Warning: Missing G or B channel for light layer {name}. Skipping.")
                continue

            r_arr, g_arr, b_arr = self._read_channels(f, [c_names['R'], c_names['G'], c_names['B']], h, w, numpy_precision)
            rgb = np.stack([r_arr, g_arr, b_arr], axis=-1)
            images_list.append(rgb)
            
        if len(images_list) > 0:
            images_np = np.stack(images_list, axis=0)  # (N, H, W, 3)
        else:
            images_np = np.empty((0, h, w, 3), dtype=numpy_precision)
            
        return optimized_layer_names, images_np


def get_images_tensor_from_OLAT_dir(path_to_olat_dir, name_of_non_optimized_lights_layer = None, file_pattern="*.exr", device='cuda', numpy_precision=np.float32, torch_precision=torch.float32):
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
    base_dir = Path(path_to_olat_dir)
    optimized_dir = base_dir / "optimizable_lights"
    if not optimized_dir.exists() or not optimized_dir.is_dir():
        raise FileNotFoundError(f"Expected optimizable_lights directory at: {optimized_dir}")
    
    file_paths = glob.glob(os.path.join(str(optimized_dir), file_pattern))
    sorted_files = sorted(file_paths)
    
    non_optimized_lights_path: str | None = None
    if name_of_non_optimized_lights_layer is not None:
        # Non-optimized light files must be placed directly in the scene base_dir (not in optimizable_lights/)
        candidate = base_dir / name_of_non_optimized_lights_layer
        if candidate.exists():
            non_optimized_lights_path = str(candidate)
        else:
            raise FileNotFoundError(
                f"Non-optimized light file '{name_of_non_optimized_lights_layer}' was not found in scene base directory '{base_dir}'. "
                f"Non-optimized light EXR files must be placed directly in the scene base directory (not inside 'optimizable_lights/')."
            )

    exr_reader = OpenEXRReader()
    
    images_list = []
    for file_path in sorted_files:
        image_np: np.ndarray = exr_reader.read_rgb(file_path, numpy_precision=numpy_precision)
        images_list.append(image_np)
    
    non_optimized_lights_tensor = None
    if non_optimized_lights_path is not None:
        non_opt_np = exr_reader.read_rgb(non_optimized_lights_path, numpy_precision=numpy_precision)
        non_optimized_lights_tensor = torch.from_numpy(non_opt_np).to(device=device, dtype=torch_precision)
        
    assert len(images_list) > 0, "No images found in the specified directory with the given pattern."
    images_np = np.stack(images_list, axis=0)  # (N, H, W, C)
    images_tensor = torch.from_numpy(images_np).to(device=device, dtype=torch_precision)

    return images_tensor, non_optimized_lights_tensor, sorted_files

def get_images_tensor_from_multi_layer_exr(path_to_exr, return_non_optimized_lights_layer=False, device='cuda', light_layer_keyword = 'LGT', numpy_precision=np.float32, torch_precision=torch.float32):
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
            This assumes that only the optimizable light passes are present in the EXR.
        device: Torch device for returned tensors.
        light_layer_keyword: Substring that identifies light layers (e.g., 'LGT').
        numpy_precision: NumPy dtype used when reading with Mitsuba.
        torch_precision: Torch dtype for output tensors.

    Returns:
        images_tensor: Tensor of shape (N, H, W, 3) with per-light images.
        non_optimized_lights_tensor: Tensor of shape (H, W, 3) if requested,
            which is the full image minus the sum of the optimizable light passes.
        optimized_layer_names: List of light names corresponding to images_tensor.

    Notes:
        - Non optimized lights layer computation can be noisy; consider denoising or a more robust
          separation if artifacts appear.
    """
    exr_reader = OpenEXRReader()
    optimized_layer_names, images_np = exr_reader.read_multilayer(path_to_exr, light_layer_keyword, numpy_precision=numpy_precision)

    if images_np.size > 0:
        images_tensor = torch.from_numpy(images_np).to(device=device, dtype=torch_precision)
    else:
        raise ValueError("No light layers found in the EXR.")

    non_optimized_lights_tensor = None
    if return_non_optimized_lights_layer:
        # Subtract the sum of the optimized lights from the full image to get the non-optimized lights.
        # NOTE: This is prone to artifacts from noise and stuff, would be good to refactor
        base_full_np = exr_reader.read_rgb(path_to_exr, numpy_precision=numpy_precision)
        base_full_tensor = torch.from_numpy(base_full_np[:, :, :3]).to(device=device, dtype=torch_precision)
        if images_tensor.shape[0] > 0:
            non_optimized_lights_tensor = base_full_tensor - torch.sum(images_tensor, dim=0)
        else:
            raise ValueError("No optimized lights to subtract.")
    return images_tensor, non_optimized_lights_tensor, optimized_layer_names # (N, H, W, C), (H, W, C), list of light names corresponding with images_tensor

def load_alpha_tensor(path_to_alpha_exr, device='cuda', numpy_precision=np.float32, torch_precision=torch.float32):
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
    exr_reader = OpenEXRReader()
    alpha_np = exr_reader.read_alpha(path_to_alpha_exr, numpy_precision=numpy_precision)
    alpha_tensor = torch.from_numpy(alpha_np).to(device=device, dtype=torch_precision)
    return alpha_tensor
    
"""Utilities for loading OLAT scenes from EXRs into PyTorch tensors.

Provides:
- get_images_tensor_from_OLAT_dir: Load per-light EXR images from a directory.
- get_images_tensor_from_multi_layer_exr: Load per-light layers from a multilayer EXR.
- load_alpha_tensor: Load only the alpha channel as a tensor.
"""

import glob
import os
from pathlib import Path

import Imath
import numpy as np
import OpenEXR
import torch


class OpenEXRReader:
    """Reader helper for decoding EXR image files via OpenEXR."""

    @staticmethod
    def _read_channels(f: OpenEXR.InputFile, channel_names: list[str], h: int, w: int, numpy_precision=np.float32) -> list[np.ndarray]:
        """Decode raw EXR channel byte buffers into 2D NumPy arrays.

        Args:
            f: OpenEXR InputFile handle.
            channel_names: List of channel names to read.
            h: Image height.
            w: Image width.
            numpy_precision: Target NumPy floating precision.

        Returns:
            List of 2D NumPy arrays of shape (H, W).
        """
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        raw_buffers = f.channels(channel_names, pt)
        return [
            np.frombuffer(buf, dtype=np.float32).reshape(h, w).astype(numpy_precision, copy=False)
            for buf in raw_buffers
        ]

    def read_rgb(self, path: str, numpy_precision=np.float32) -> np.ndarray:
        """Read RGB channels from an EXR file.

        Args:
            path: Path to the EXR file.
            numpy_precision: NumPy dtype (default float32).

        Returns:
            NumPy array of shape (H, W, 3).

        Raises:
            ValueError: If required RGB channels are missing.
        """
        f = OpenEXR.InputFile(path)
        header = f.header()
        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        chs = header["channels"].keys()

        def find_channel(name: str) -> str | None:
            if name in chs:
                return name
            for k in chs:
                if k.endswith("." + name):
                    return k
            return None

        r_name, g_name, b_name = find_channel("R"), find_channel("G"), find_channel("B")
        if r_name is None or g_name is None or b_name is None:
            raise ValueError(f"EXR at {path} is missing RGB channels")
        r_arr, g_arr, b_arr = self._read_channels(f, [r_name, g_name, b_name], h, w, numpy_precision)
        return np.stack([r_arr, g_arr, b_arr], axis=-1)

    def read_alpha(self, path: str, numpy_precision=np.float32) -> np.ndarray:
        """Read alpha channel from an EXR file.

        Args:
            path: Path to the EXR file.
            numpy_precision: NumPy dtype (default float32).

        Returns:
            NumPy array of shape (H, W, 1).

        Raises:
            ValueError: If alpha channel is not present.
        """
        f = OpenEXR.InputFile(path)
        header = f.header()
        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        chs = header["channels"].keys()

        def find_alpha() -> str | None:
            if "A" in chs:
                return "A"
            for k in chs:
                if k.endswith(".A"):
                    return k
            return None

        a_name = find_alpha()
        if a_name is None:
            raise ValueError(f"Alpha channel not found in EXR: {path}")
        (a_arr,) = self._read_channels(f, [a_name], h, w, numpy_precision)
        return a_arr[..., None]

    def read_multilayer(self, path: str, light_layer_keyword: str, numpy_precision=np.float32) -> tuple[list[str], np.ndarray]:
        """Read per-light layers from a multilayer EXR file.

        Args:
            path: Path to multilayer EXR.
            light_layer_keyword: Substring identifying light pass layers.
            numpy_precision: NumPy dtype.

        Returns:
            Tuple of (layer_names_list, images_array_4d) where images_array_4d has shape (N, H, W, 3).
        """
        f = OpenEXR.InputFile(path)
        header = f.header()
        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1

        channels = list(header["channels"].keys())

        # Group channels by light name
        dict_light_name_to_channels = {}
        for ch in channels:
            if light_layer_keyword in ch and ch.endswith(".R"):
                light_name = ch[:-2]
                dict_light_name_to_channels[light_name] = {"R": ch, "G": ch[:-2] + ".G", "B": ch[:-2] + ".B"}

        optimized_layer_names = list(dict_light_name_to_channels.keys())
        images_list = []
        for name in optimized_layer_names:
            c_names = dict_light_name_to_channels[name]
            if c_names["G"] not in channels or c_names["B"] not in channels:
                print(f"Warning: Missing G or B channel for light layer {name}. Skipping.")
                continue

            r_arr, g_arr, b_arr = self._read_channels(f, [c_names["R"], c_names["G"], c_names["B"]], h, w, numpy_precision)
            rgb = np.stack([r_arr, g_arr, b_arr], axis=-1)
            images_list.append(rgb)

        if len(images_list) > 0:
            images_np = np.stack(images_list, axis=0)
        else:
            images_np = np.empty((0, h, w, 3), dtype=numpy_precision)

        return optimized_layer_names, images_np


def get_images_tensor_from_OLAT_dir(
    path_to_olat_dir: str,
    name_of_non_optimized_lights_layer: str | None = None,
    file_pattern: str = "*.exr",
    device: str = "cuda",
    numpy_precision=np.float32,
    torch_precision=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
    """Load a stack of per-light images from an OLAT scene directory.

    Args:
        path_to_olat_dir: Directory containing optimizable_lights/ subfolder.
        name_of_non_optimized_lights_layer: Optional filename for base light pass (e.g. 'base_lighting.exr').
        file_pattern: Glob pattern for matching EXRs (default '*.exr').
        device: PyTorch device for returned tensors.
        numpy_precision: NumPy dtype used during decoding.
        torch_precision: PyTorch tensor dtype.

    Returns:
        Tuple of (images_tensor [N, H, W, 3], non_optimized_tensor [H, W, 3] or None, sorted_files_list).

    Raises:
        FileNotFoundError: If optimizable_lights directory or non-optimized file is missing.
    """
    base_dir = Path(path_to_olat_dir)
    optimized_dir = base_dir / "optimizable_lights"
    if not optimized_dir.exists() or not optimized_dir.is_dir():
        raise FileNotFoundError(f"Expected optimizable_lights directory at: {optimized_dir}")

    file_paths = glob.glob(os.path.join(str(optimized_dir), file_pattern))
    sorted_files = sorted(file_paths)

    non_optimized_lights_path: str | None = None
    if name_of_non_optimized_lights_layer is not None:
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
    images_np = np.stack(images_list, axis=0)
    images_tensor = torch.from_numpy(images_np).to(device=device, dtype=torch_precision)

    return images_tensor, non_optimized_lights_tensor, sorted_files


def get_images_tensor_from_multi_layer_exr(
    path_to_exr: str,
    return_non_optimized_lights_layer: bool = False,
    device: str = "cuda",
    light_layer_keyword: str = "LGT",
    numpy_precision=np.float32,
    torch_precision=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
    """Load per-light layers from a multilayer EXR into a PyTorch stack.

    Args:
        path_to_exr: Path to the multilayer EXR file.
        return_non_optimized_lights_layer: If True, computes residual non-optimized pass.
        device: PyTorch device.
        light_layer_keyword: Substring identifying light layers.
        numpy_precision: NumPy dtype used during decoding.
        torch_precision: PyTorch output tensor dtype.

    Returns:
        Tuple of (images_tensor [N, H, W, 3], non_optimized_tensor [H, W, 3] or None, layer_names_list).

    Raises:
        ValueError: If no light layers are found in the EXR.
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
    return images_tensor, non_optimized_lights_tensor, optimized_layer_names


def load_alpha_tensor(
    path_to_alpha_exr: str,
    device: str = "cuda",
    numpy_precision=np.float32,
    torch_precision=torch.float32,
) -> torch.Tensor:
    """Load the alpha channel from an EXR file as an (H, W, 1) tensor.

    Args:
        path_to_alpha_exr: Path to the alpha channel EXR.
        device: PyTorch device.
        numpy_precision: NumPy dtype used during reading.
        torch_precision: PyTorch output tensor dtype.

    Returns:
        Alpha mask tensor of shape (H, W, 1).
    """
    exr_reader = OpenEXRReader()
    alpha_np = exr_reader.read_alpha(path_to_alpha_exr, numpy_precision=numpy_precision)
    alpha_tensor = torch.from_numpy(alpha_np).to(device=device, dtype=torch_precision)
    return alpha_tensor
    
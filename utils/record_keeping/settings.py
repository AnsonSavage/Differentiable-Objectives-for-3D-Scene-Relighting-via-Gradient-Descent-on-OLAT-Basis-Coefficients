"""Helper for tracking and assembling experiment run settings for JSON serialization."""
from collections.abc import Callable
from datetime import datetime
from typing import Any

import torch
import torchvision as tv

from utils.record_keeping.serialization import to_serializable


def build_settings(
    *,
    title_prefix: str,
    learning_rate: float,
    n_iterations: int,
    save_every: int,
    save_loss_plot_each_iteration: bool,
    color_space_converter: dict[str, Any],
    require_non_negative_multipliers: bool,
    torch_precision: torch.dtype,
    device: str,
    images_tensor_shape: tuple[int, ...] | list[int],
    augmentation: Any,
    augmentation_callback: Any,
    parameters_stored_as_hsv: bool,
    hsv_callback: Callable[[int], str] | None,
    learning_rate_scheduler_creator_callback: Callable[[torch.optim.Optimizer], object] | None = None,
    criterion_type: str,
    prompt_info: dict[str, Any] | None = None,
    init_mean: float | tuple[float, float, float],
    init_std: float | tuple[float, float, float],
    scene_name: str | None = None,
    model_name: str,
    pretrained_source: str,
    patience: int | None = None,
    seed: int,
) -> dict[str, Any]:
    """Assemble a structured experiment configuration dictionary for JSON serialization.

    Args:
        title_prefix: Prefix string for image titles.
        learning_rate: Optimizer learning rate.
        n_iterations: Total optimization iterations.
        save_every: Snapshot frequency interval.
        save_loss_plot_each_iteration: Whether to plot loss on each iteration.
        color_space_converter: Dictionary containing color space converter metadata.
        require_non_negative_multipliers: Whether non-negative multiplier constraint was applied.
        torch_precision: Computation precision dtype.
        device: Device string (e.g. 'cuda:0' or 'cpu').
        images_tensor_shape: Shape tuple/list of the optimizable images tensor.
        augmentation: Static torchvision augmentation transform or composition.
        augmentation_callback: Step-dependent augmentation callback.
        parameters_stored_as_hsv: Whether optimization parameters are stored in HSV space.
        hsv_callback: Callback defining active HSV parameter channels.
        learning_rate_scheduler_creator_callback: Learning rate scheduler callback.
        criterion_type: Loss criterion class name.
        prompt_info: Optional dictionary containing prompt metadata.
        init_mean: Initial multiplier mean value or per-channel tuple.
        init_std: Initial multiplier standard deviation or per-channel tuple.
        scene_name: Optional name of the scene being optimized.
        model_name: Vision/CLIP backbone model name.
        pretrained_source: Pretrained weights source identifier.
        patience: Early stopping patience (iterations without improvement).
        seed: Random seed applied for reproducibility.

    Returns:
        Structured settings dictionary ready for JSON serialization and logging.
    """
    settings: dict[str, Any] = {
        "prompts": prompt_info or {},
        "criterion": {
            "type": criterion_type,
        },
        "clip_model": {
            "name": model_name,
            "pretrained": pretrained_source,
        },
        "run": {
            "timestamp": datetime.now().isoformat(),
            "title_prefix": title_prefix,
            "seed": int(seed),
        },
        "data": {
            "scene": scene_name,
            "num_lights": int(images_tensor_shape[0]),
            "image_shape": list(images_tensor_shape),
        },
        "init": {
            "starting_multiplier_mean": init_mean,
            "starting_multiplier_std": init_std,
            "note": "Defaults: RGB mean=(1.0, 1.0, 1.0), HSV mean=(0.0, 0.0, 1.0)",
        },
        "training": {
            "n_iterations": n_iterations,
            "save_every": save_every,
            "save_loss_plot_each_iteration": save_loss_plot_each_iteration,
            "color_space_converter": color_space_converter,
            "require_non_negative_multipliers": require_non_negative_multipliers,
            "precision": str(torch_precision),
            "device": str(device),
            "augmentation": to_serializable(augmentation) if augmentation is not None else None,
            "augmentation_callback": to_serializable(augmentation_callback) if augmentation_callback is not None else None,
            "parameters_stored_as_hsv": parameters_stored_as_hsv,
            "hsv_callback": to_serializable(hsv_callback) if hsv_callback is not None else None,
            "learning_rate_scheduler_creator_callback": to_serializable(learning_rate_scheduler_creator_callback) if learning_rate_scheduler_creator_callback is not None else None,
            "optimizer": {
                "algorithm": "Adam",
                "learning_rate": learning_rate,
            },
            "early_stopping_patience": int(patience) if patience is not None else None,
        },
        "environment": {
            "torch": torch.__version__,
            "torchvision": getattr(tv, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    return settings

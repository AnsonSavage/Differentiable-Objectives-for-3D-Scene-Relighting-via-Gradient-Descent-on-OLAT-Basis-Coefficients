"""
Optimization loop for learning multipliers of OLATs for a given criterion.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
from torchvision.transforms import v2
from tqdm import tqdm

from losses.base import BaseLoss, UpdatableLoss
from utils.color.linear_to_srgb_converters import (
    LinearRec709ToAgXBase,
    LinearRec709TosRGB,
)
from utils.image.display import display_image_batch_grid
from utils.parameter_strategies import (
    HSVParameterStrategy,
    ParameterStrategy,
    RGBParameterStrategy,
)
from utils.record_keeping.experiment import (
    FolderManager,
    PlotManager,
    ResourceUsageTracker,
)
from utils.record_keeping.settings import build_settings
from utils.scene import Scene
from utils.seed import set_global_seed


def optimize_with_criterion(
    scene: Scene,
    learning_rate: float,
    n_iterations: int,
    criterion: BaseLoss | UpdatableLoss,
    starting_multiplier_std: float | tuple[float, float, float],
    output_subdirectory_name: str,
    n_results: int = 1,
    patience: int | None = None,
    learning_rate_scheduler_creator_callback: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler] | None = None,
    torch_precision=torch.float32,
    device='cuda',
    augmentation: v2.Transform | None = None,
    augmentation_callback: Callable[[int], v2.Transform] | None = None,
    parameters_stored_as_hsv: bool = False,
    hsv_callback: Callable[[int], str] | None = None,
    render_color_space_converter: LinearRec709TosRGB = LinearRec709ToAgXBase(),
    require_physically_plausible_multipliers: bool = True,
    loss_on_multipliers: torch.nn.Module | None = None,
    title_prefix: str = "",
    save_every=50,
    model_name: str | None = None,
    pretrained_source: str | None = None,
    seed: int | None = None,
    show_images: bool = True,
    show_images_before_augmentation: bool = True,
    show_images_after_augmentation: bool = False,
    show_images_after_augmentation_callback: bool = False,
    save_final_display_to_run_dir: bool = True,
    run_name_suffix: str | None = None,
    starting_multiplier_mean: float | tuple[float, float, float] | None = None,
    save_loss_plot_each_iteration: bool = False,
):
    """Train image multipliers to optimize a criterion.
    
    Args:
        scene: Scene object providing images to optimize
        learning_rate: Learning rate for optimization
        n_iterations: Number of iterations to run
        criterion: BaseLoss | UpdatableLoss, Loss function to minimize
        starting_multiplier_std: Standard deviation for multiplier initialization
                                 (single float for RGB, tuple of 3 floats for HSV)
        starting_multiplier_mean: Mean for multiplier initialization
                      (single float or tuple of 3 floats). Defaults are
                      RGB=(1.0, 1.0, 1.0), HSV=(0.0, 0.0, 1.0).
        n_results: Number of images to optimize in parallel
        patience: Number of iterations with no improvement to wait before early stopping
        torch_precision: Data type precision
        device: Computation device
        augmentation: A random augmentation to apply during training
        augmentation_callback: A callable that takes the current epoch and returns a v2.Transform augmentation to apply
        parameters_stored_as_hsv: Whether to store parameters in HSV color space
        hsv_callback: A callable that takes the current epoch and returns a string containing a combination of the character 'h', 's', and 'v', indicating whether it is legal to adjust the hue, saturation, or value
        render_color_space_converter: A LinearRec709TosRGB implementation to convert to display space
        require_physically_plausible_multipliers: Whether to clamp multipliers to be non-negative, etc.
        title_prefix: Prefix for image titles
        show_images_before_augmentation: If True, display images before any augmentations
        show_images_after_augmentation: If True, display images after the augmentation passed in via the augmentation argument
        show_images_after_augmentation_callback: If True, display images after augmentation_callback
        
    Returns:
        Tuple of (multipliers, loss_values)

    Side Effects / Outputs:
        - settings.json: configuration used for this run
        - resources_summary.json: runtime + GPU memory usage (runtime ms, runtime cuda bytes)
        - loss_plot.png, intermediate/final images, multipliers snapshots
        - optionally, loss_plot_iterXXXX.png for every iteration
    """
    # Determine seed and set global RNGs for reproducibility
    seed = set_global_seed(seed)

    # Resource tracking (runtime + CUDA memory) kept minimal and out of main logic
    usage_tracker = ResourceUsageTracker(device)
    usage_tracker.start()

    # Setup experiment tracking
    folder_manager = FolderManager(output_subdirectory_name)
    criterion_name = criterion.__class__.__name__
    run_name_parts = [scene.name, criterion_name]
    if model_name:
        run_name_parts.append(str(model_name))
    if run_name_suffix:
        run_name_parts.append(str(run_name_suffix))
    folder_manager.create_run_folder("_".join(run_name_parts))
    # Inform the criterion about the run directory if it supports such a hook
    on_run_dir = getattr(criterion, "on_run_dir_created", None)
    if callable(on_run_dir):
        try:
            on_run_dir(folder_manager.run_dir)
        except Exception as e:
            print(f"Warning: criterion.on_run_dir_created failed: {e}")
    plot_manager = PlotManager(folder_manager)

    # Extract values from the scene
    optimizable_images = scene.get_optimizable_images()
    non_optimized_lights_tensor = scene.get_non_optimized_lights()
    alpha_mask_tensor = scene.get_alpha_mask()

    if starting_multiplier_mean is None:
        starting_multiplier_mean = (0.0, 0.0, 1.0) if parameters_stored_as_hsv else (1.0, 1.0, 1.0)

    # Save settings (once per run)
    settings = build_settings(
        title_prefix=title_prefix,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        save_every=save_every,
        save_loss_plot_each_iteration=save_loss_plot_each_iteration,
        color_space_converter=render_color_space_converter.settings_info(),
        require_non_negative_multipliers=require_physically_plausible_multipliers,
        torch_precision=torch_precision,
        device=device,
        images_tensor_shape=optimizable_images.shape,
        augmentation=augmentation,
        augmentation_callback=augmentation_callback,
        learning_rate_scheduler_creator_callback=learning_rate_scheduler_creator_callback,
        parameters_stored_as_hsv=parameters_stored_as_hsv,
        hsv_callback=hsv_callback,
        criterion_type=criterion_name,
        prompt_info=criterion.get_prompt_info(),
        init_mean=starting_multiplier_mean,
        init_std=starting_multiplier_std,
        scene_name=scene.name,
        model_name=model_name or "",
        pretrained_source=pretrained_source or "",
        patience=patience,
        seed=seed,
    )
    folder_manager.save_settings(settings)
    
    # Initialize multipliers using strategy pattern
    num_lights = optimizable_images.shape[0]
    
    # Create appropriate parameter strategy
    if parameters_stored_as_hsv:
        param_strategy: ParameterStrategy = HSVParameterStrategy(hsv_callback=hsv_callback, verbose=True)
    else:
        param_strategy: ParameterStrategy = RGBParameterStrategy(verbose=True)
    
    # Initialize parameters via strategy
    param_strategy.initialize_parameters(
        n_results=n_results,
        num_lights=num_lights,
        mean=starting_multiplier_mean,
        std=starting_multiplier_std,
        torch_precision=torch_precision,
        device=device,
        learning_rate=learning_rate,
        learning_rate_scheduler_creator_callback=learning_rate_scheduler_creator_callback,
    )

    if require_physically_plausible_multipliers:
        # If the run requests physically plausible multipliers, enforce them immediately
        # so the very first loss evaluation cannot be poisoned by invalid parameter values.
        param_strategy.apply_physical_constraints()
    
    loss_values = []

    # Early stopping bookkeeping
    best_loss = float("inf")
    n_iterations_with_no_improvement = 0

    def _save_and_display(
        iteration: int,
        loss_val: float,
        images_before,
        images_after=None,
        images_after_callback=None,
        is_final: bool = False,
    ):
        """Helper to save images, display optional views, and save multipliers.

        Matches the behavior used in the training loop's save blocks so we don't
        duplicate code.
        """
        print(f"Iteration {iteration}, Loss: {loss_val:.4f}")
        title = f"{title_prefix}{' ' if title_prefix else ''}Iteration: {iteration}, Loss: {loss_val:.4f}"
        plot_manager.save_image(images_before, iteration, title=title, loss=loss_val, plain_image=True)
        if show_images:
            # Optionally display images before augmentation
            if show_images_before_augmentation:
                assert images_before.shape[0] == n_results
                _save_path = None
                if save_final_display_to_run_dir and is_final:
                    _save_path = folder_manager.run_dir
                display_image_batch_grid(
                    images_before,
                    title=title + " (before augmentation)",
                    max_cols=4,
                    show=True,
                    save_path=_save_path,
                    save_each=False,
                    is_last=is_final,
                    save_index=iteration,
                )
            # Optionally display images after augmentation
            if show_images_after_augmentation and images_after is not None:
                assert images_after.shape[0] == n_results
                display_image_batch_grid(
                    images_after,
                    title=title + " (after augmentation)",
                    max_cols=4,
                    show=True,
                )
            # Optionally display images after augmentation callback
            if show_images_after_augmentation_callback and images_after_callback is not None:
                assert images_after_callback.shape[0] == n_results
                display_image_batch_grid(
                    images_after_callback,
                    title=title + " (after augmentation_callback)",
                    max_cols=4,
                    show=True,
                )
        # Save multipliers snapshot for this iteration
        multipliers_path = folder_manager.get_multipliers_path(iteration, loss_val)
        params_to_save = param_strategy.get_parameters_for_saving()
        torch.save(params_to_save.detach().cpu(), multipliers_path)

    for epoch in tqdm(range(n_iterations)):
        # Zero gradients
        param_strategy.zero_grad()
        
        # Update parameter constraints (e.g., which HSV channels are trainable)
        param_strategy.update_parameter_constraints(epoch)

        if isinstance(criterion, UpdatableLoss):
            criterion.update_parameters(epoch, n_iterations)

        # Get multipliers in RGB space (handles conversion if needed)
        multipliers = param_strategy.get_multipliers()
        mult_reshaped = multipliers.view(n_results, num_lights, 1, 1, 3)

        # Sum across the lights dimension to get [height, width, channels]
        predicted_images = torch.sum(mult_reshaped * optimizable_images.unsqueeze(0), dim=1) # Should be [n_results, H, W, C]

        if non_optimized_lights_tensor is not None:
            predicted_images += non_optimized_lights_tensor.unsqueeze(0)
        
        # Permute from [n_results, height, width, channels] to [n_results, channels, height, width]
        predicted_images = predicted_images.permute(0, 3, 1, 2)
        
        predicted_images = render_color_space_converter(predicted_images)
        images_before_augmentations = predicted_images.clone()
        if augmentation:
            predicted_images = torch.stack([augmentation(individual_img) for individual_img in predicted_images])
        if alpha_mask_tensor is not None: # TODO: Currently only sets non white pixels to black. No support for mixing with another background.
            predicted_images *= alpha_mask_tensor.permute(2, 0, 1).unsqueeze(0)
        if show_images and show_images_after_augmentation:
            images_after_augmentation = predicted_images.clone()
        if augmentation_callback:
            aug = augmentation_callback(epoch)
            predicted_images = torch.stack([aug(individual_img) for individual_img in predicted_images])
        images_after_augmentation_callback = predicted_images # If you edit predicted images further, you'd need to clone :)

        loss = criterion(predicted_images).mean(dim=0) # Note that when using Adam, the magnitude of the gradient cancels

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected at iteration {epoch + 1}: {loss.item()}")

        if loss_on_multipliers is not None:
            loss += loss_on_multipliers(multipliers)

        loss.backward()
        
        # Perform optimizer step via strategy
        param_strategy.step()

        # Apply physical constraints via strategy
        if require_physically_plausible_multipliers:
            param_strategy.apply_physical_constraints()

        loss_values.append(loss.item())  # Store the loss value

        # Early stopping: check for improvement
        cur_loss_val = loss.item()
        if cur_loss_val < best_loss:
            best_loss = cur_loss_val
            n_iterations_with_no_improvement = 0
        else:
            n_iterations_with_no_improvement += 1

        # Save images and multipliers at regular intervals
        should_save = (epoch + 1) % save_every == 0 or epoch == 0 or (epoch + 1) == n_iterations
        if should_save:
            _save_and_display(
                epoch + 1,
                loss.item(),
                images_before_augmentations,
                images_after_augmentation if (show_images and show_images_after_augmentation) else None,
                images_after_augmentation_callback if (show_images and show_images_after_augmentation_callback) else None,
                is_final=(epoch + 1) == n_iterations,
            )

        if save_loss_plot_each_iteration:
            plot_manager.save_loss_plot(loss_values, iteration=epoch + 1, show_plot=False)

        # If patience is set and we've gone too many iterations without improvement,
        # do one last display/save and exit early.
        if patience is not None and patience > 0 and n_iterations_with_no_improvement >= patience:
            if not should_save:
                _save_and_display(
                    epoch + 1,
                    loss.item(),
                    images_before_augmentations,
                    images_after_augmentation if (show_images and show_images_after_augmentation) else None,
                    images_after_augmentation_callback if (show_images and show_images_after_augmentation_callback) else None,
                    is_final=True,
                )

            break

    # Save final results
    plot_manager.save_loss_plot(loss_values)
    
    # Save final multipliers as a torch tensor
    if folder_manager.run_dir:
        final_params = param_strategy.get_parameters_for_saving()
        torch.save(final_params.detach().cpu(), os.path.join(folder_manager.run_dir, "final_multipliers.pt"))

    # Finalize resource usage tracking (only extra info beyond settings.json)
    usage_tracker.finish()
    folder_manager.save_resources_summary(usage_tracker.summary())
    
    return multipliers, loss_values
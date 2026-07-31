"""
Experiment management utilities for tracking and visualizing optimization runs.
"""

import json
import os
import re
import shutil
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import torch

from config import DEFAULT_OPTIMIZATION_RUNS_DIR
from utils.record_keeping.serialization import to_serializable


class FolderManager:
    """Manages experiment folders and file paths."""

    def __init__(self, subdirectory: str | None = None, base_dir: str | None = None):
        """Initialize folder manager.

        Args:
            subdirectory: Subdirectory inside base_dir for specific experiment groups
            base_dir: Base directory for all experiment runs. Defaults to DEFAULT_OPTIMIZATION_RUNS_DIR.
        """
        if not base_dir:
            base_dir = DEFAULT_OPTIMIZATION_RUNS_DIR
        self.base_dir = base_dir
        self.run_dir = ""
        if subdirectory:
            self.base_dir = os.path.join(base_dir, subdirectory)

        # Create base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        # Set when the run directory is actually created.
        self.timestamp: str | None = None

    def _new_timestamp(self) -> str:
        # Microsecond precision to reduce collision probability between concurrent shards.
        return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    
    def create_run_folder(self, prefix="run"):
        """Create a uniquely named folder for this experiment run.
        
        Args:
            prefix: Prefix for the folder name
        Returns:
            Path to created folder
        """
        # Never reuse an existing directory: collisions here cause multiple jobs to
        # write into the same run folder and corrupt results.
        last_err: Exception | None = None
        for _ in range(200):
            timestamp = self._new_timestamp()
            run_name = f"{prefix}_{timestamp}"
            run_path = os.path.join(self.base_dir, run_name)
            try:
                os.makedirs(run_path, exist_ok=False)
            except FileExistsError as e:
                last_err = e
                # Extremely unlikely with microseconds, but possible under heavy concurrency.
                continue
            self.timestamp = timestamp
            self.run_dir = run_path
            return run_path

        raise RuntimeError(
            f"Failed to create a unique run folder in {self.base_dir} after many attempts"
        ) from last_err
    
    def save_settings(self, settings_dict):
        """Save experiment settings to JSON.
        
        Args:
            settings_dict: Dictionary of experiment parameters
        """
        if not self.run_dir:
            self.create_run_folder()
        
        # Make all values JSON-serializable
        serialized_dict = to_serializable(settings_dict)
        with open(os.path.join(self.run_dir, "settings.json"), 'w') as f:
            json.dump(serialized_dict, f, indent=2)

    def get_multipliers_dir(self) -> str:
        """Get (and ensure) the directory path for saving multipliers snapshots."""
        if not self.run_dir:
            self.create_run_folder()
        path = os.path.join(self.run_dir, "multipliers")
        os.makedirs(path, exist_ok=True)
        return path

    def get_multipliers_path(self, iteration: int, loss: float) -> str:
        """Build a filename for multipliers at a given iteration and loss."""
        mdir = self.get_multipliers_dir()
        return os.path.join(mdir, f"multipliers_iter_{iteration:04d}_loss_{loss:.4f}.pt")

    def get_image_path(self, iteration, opt_index=None):
        """Get path for saving an iteration's image, optionally with an optimization index suffix."""
        if opt_index is None:
            return os.path.join(self.run_dir, f"image_iter_{iteration:04d}.png")
        else:
            return os.path.join(self.run_dir, f"image_iter_{iteration:04d}_opt{opt_index}.png")
    
    def get_loss_plot_path(self):
        """Get path for saving the loss plot."""
        return os.path.join(self.run_dir, "loss_plot.png")

    def save_resources_summary(self, summary: dict, filename: str = "resources_summary.json"):
        """Save a JSON file containing resource usage metrics for the run.

        Args:
            summary: Dictionary containing keys like runtime, memory usage, etc.
            filename: Name of the json file to write (default resources_summary.json)
        """
        if not self.run_dir:
            self.create_run_folder()
        # Ensure serializable (re-using to_serializable might be overkill; rely on simple types expected here)
        path = os.path.join(self.run_dir, filename)
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

    def collect_results(self, summary_folder_name="summary"):
        """
        Collects loss plots and final image grids from all runs in the current base directory
        and copies them to a summary folder.
        """
        source_dir = self.base_dir
        summary_dir = os.path.join(source_dir, summary_folder_name)
        
        if not os.path.exists(source_dir):
            print(f"Source directory not found: {source_dir}")
            return

        os.makedirs(summary_dir, exist_ok=True)
        print(f"Collecting results into: {summary_dir}")
        
        # Regex to find display grids and extract iteration number
        # Pattern matches: display_grid_<iteration>_<timestamp>.png
        grid_pattern = re.compile(r"display_grid_(\d+)_(\d+)\.png")

        for run_folder in os.listdir(source_dir):
            run_path = os.path.join(source_dir, run_folder)
            
            # Skip if not a directory or is the summary folder itself
            if not os.path.isdir(run_path) or run_folder == summary_folder_name:
                continue
                
            print(f"Processing: {run_folder}")
            
            # 1. Collect Loss Plot
            loss_plot_src = os.path.join(run_path, "loss_plot.png")
            if os.path.exists(loss_plot_src):
                loss_plot_dst = os.path.join(summary_dir, f"{run_folder}_loss_plot.png")
                shutil.copy2(loss_plot_src, loss_plot_dst)
            else:
                print(f"  - No loss plot found in {run_folder}")

            # 2. Collect Final Image Grid
            # Find all display_grid files
            grid_files = []
            for f in os.listdir(run_path):
                match = grid_pattern.match(f)
                if match:
                    iteration = int(match.group(1))
                    timestamp = int(match.group(2))
                    grid_files.append((iteration, timestamp, f))
            
            if grid_files:
                # Sort by iteration (descending), then timestamp (descending)
                grid_files.sort(key=lambda x: (x[0], x[1]), reverse=True)
                last_grid_file = grid_files[0][2]
                
                grid_src = os.path.join(run_path, last_grid_file)
                grid_dst = os.path.join(summary_dir, f"{run_folder}_final_grid.png")
                shutil.copy2(grid_src, grid_dst)
            else:
                print(f"  - No display grid found in {run_folder}")

        print("Collection complete!")


class ResourceUsageTracker:
    """Lightweight helper to measure runtime and peak CUDA memory.

    Usage:
        tracker = ResourceUsageTracker(device)
        tracker.start()
        ... run training ...
        tracker.finish()
        folder_manager.save_resources_summary(tracker.summary())
    """

    def __init__(self, device: str):
        self.device = device
        self._start_time = None
        self._end_time = None
        self._cuda_tracked = False
        self._max_cuda_mem_bytes = None

    def start(self):
        self._start_time = time.time()
        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            try:
                torch.cuda.reset_peak_memory_stats(device=self.device)
                self._cuda_tracked = True
            except TypeError:
                # Fallback / older versions: skip tracking to avoid noisy errors
                self._cuda_tracked = False

    def finish(self):
        self._end_time = time.time()
        if self._cuda_tracked:
            try:
                self._max_cuda_mem_bytes = torch.cuda.max_memory_allocated(device=self.device)
            except TypeError:
                try:
                    self._max_cuda_mem_bytes = torch.cuda.max_memory_allocated()
                except Exception:
                    self._max_cuda_mem_bytes = None

    def summary(self) -> dict:
        if self._start_time is None or self._end_time is None:
            raise RuntimeError("Tracker must be started and finished before requesting summary.")
        runtime_ms = (self._end_time - self._start_time) * 1000.0
        return {
            "runtime_ms": runtime_ms,
            "max_cuda_memory_bytes": self._max_cuda_mem_bytes,
        }

class PlotManager:
    """Handles visualization and saving of images and plots."""
    
    def __init__(self, folder_manager):
        """Initialize plot manager.
        
        Args:
            folder_manager: FolderManager instance
        """
        self.folder_manager = folder_manager
    
    def save_image(self, image_tensor, iteration, title="", loss=None, plain_image=True):
        """Save an image or a batch of images to disk.

        Args:
            image_tensor: Image tensor either [C,H,W] or [N, C, H, W]
            iteration: Current iteration number
            title: Image title (used for metadata only)
            Note: Callers are responsible for any color space conversion.
        """
        # Support both single image [C,H,W] and batched [N,C,H,W]
        if image_tensor.dim() == 3:
            batch = image_tensor.unsqueeze(0)
        else:
            batch = image_tensor

        for i in range(batch.shape[0]):
            img = batch[i]
            img_np = img.detach().cpu().permute(1, 2, 0).numpy()

            # Build filename with info
            info_parts = [f"iter{iteration:04d}"]
            if batch.shape[0] > 1:
                info_parts.append(f"opt{i}")
            if loss is not None:
                info_parts.append(f"loss_{loss:.4f}")
            if self.folder_manager.timestamp:
                info_parts.append(self.folder_manager.timestamp) # We'll assume that the timestamp acts as a unique identifier in case the images get separated from their run folder
            fname = "image_" + "_".join(info_parts) + ".png"
            save_path = os.path.join(self.folder_manager.run_dir, fname)

            if plain_image:
                plt.imsave(save_path, img_np)
            else:
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(img_np)
                plt.axis('off')
                if title:
                    plt.title(f"{title} (opt {i})")
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
    def save_loss_plot(self, loss_values, iteration: int | None = None, show_plot: bool = True):
        """Save the loss curve plot.
        
        Args:
            loss_values: List of loss values from training
            iteration: Optional iteration number for per-iteration snapshots.
            show_plot: Whether to display the figure after saving.
        """
        if iteration is None:
            save_path = self.folder_manager.get_loss_plot_path()
        else:
            save_path = os.path.join(self.folder_manager.run_dir, f"loss_plot_iter{iteration:04d}.png")

        plt.figure(figsize=(10, 5))
        plt.plot(loss_values, label="Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        title = "Loss over Iterations"
        if iteration is not None:
            title = f"{title} (iter {iteration})"
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        # Also display it if requested
        if show_plot:
            plt.show()
        plt.close()
"""Experiment management utilities for tracking and visualizing optimization runs."""

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
    """Manages experiment folders, file paths, and output collection."""

    def __init__(self, subdirectory: str | None = None, base_dir: str | None = None):
        """Initialize folder manager.

        Args:
            subdirectory: Subdirectory inside base_dir for specific experiment groups.
            base_dir: Base directory for all experiment runs (default DEFAULT_OPTIMIZATION_RUNS_DIR).
        """
        if not base_dir:
            base_dir = DEFAULT_OPTIMIZATION_RUNS_DIR
        self.base_dir = base_dir
        self.run_dir = ""
        if subdirectory:
            self.base_dir = os.path.join(base_dir, subdirectory)

        os.makedirs(self.base_dir, exist_ok=True)
        self.timestamp: str | None = None

    def _new_timestamp(self) -> str:
        """Generate a microsecond-precision UTC timestamp string."""
        return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    def create_run_folder(self, prefix: str = "run") -> str:
        """Create a uniquely named folder for this experiment run.

        Args:
            prefix: Prefix string for the folder name.

        Returns:
            Path string to the created run folder.

        Raises:
            RuntimeError: If unique folder creation fails after multiple attempts.
        """
        last_err: Exception | None = None
        for _ in range(200):
            timestamp = self._new_timestamp()
            run_name = f"{prefix}_{timestamp}"
            run_path = os.path.join(self.base_dir, run_name)
            try:
                os.makedirs(run_path, exist_ok=False)
            except FileExistsError as e:
                last_err = e
                continue
            self.timestamp = timestamp
            self.run_dir = run_path
            return run_path

        raise RuntimeError(
            f"Failed to create a unique run folder in {self.base_dir} after many attempts"
        ) from last_err

    def save_settings(self, settings_dict: dict) -> None:
        """Save experiment settings to settings.json in the run directory.

        Args:
            settings_dict: Dictionary of experiment configuration parameters.
        """
        if not self.run_dir:
            self.create_run_folder()

        serialized_dict = to_serializable(settings_dict)
        with open(os.path.join(self.run_dir, "settings.json"), "w") as f:
            json.dump(serialized_dict, f, indent=2)

    def get_multipliers_dir(self) -> str:
        """Get (and create if needed) the multipliers snapshot directory path.

        Returns:
            String path to multipliers/ directory.
        """
        if not self.run_dir:
            self.create_run_folder()
        path = os.path.join(self.run_dir, "multipliers")
        os.makedirs(path, exist_ok=True)
        return path

    def get_multipliers_path(self, iteration: int, loss: float) -> str:
        """Build path for saving multipliers at a given iteration and loss.

        Args:
            iteration: Optimization iteration index.
            loss: Loss float value at that iteration.

        Returns:
            Filepath string.
        """
        mdir = self.get_multipliers_dir()
        return os.path.join(mdir, f"multipliers_iter_{iteration:04d}_loss_{loss:.4f}.pt")

    def get_image_path(self, iteration: int, opt_index: int | None = None) -> str:
        """Get filepath for saving an iteration's rendered image.

        Args:
            iteration: Current iteration number.
            opt_index: Optional index within parallel batch.

        Returns:
            Image file path string.
        """
        if opt_index is None:
            return os.path.join(self.run_dir, f"image_iter_{iteration:04d}.png")
        return os.path.join(self.run_dir, f"image_iter_{iteration:04d}_opt{opt_index}.png")

    def get_loss_plot_path(self) -> str:
        """Get filepath for the final loss curve plot.

        Returns:
            Path string to loss_plot.png.
        """
        return os.path.join(self.run_dir, "loss_plot.png")

    def save_resources_summary(self, summary: dict, filename: str = "resources_summary.json") -> None:
        """Save resource consumption metrics (runtime, peak GPU memory) to JSON.

        Args:
            summary: Dictionary with resource tracking statistics.
            filename: Output filename (default resources_summary.json).
        """
        if not self.run_dir:
            self.create_run_folder()
        path = os.path.join(self.run_dir, filename)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    def collect_results(self, summary_folder_name: str = "summary") -> None:
        """Collect loss plots and final display grids from all runs into a summary directory.

        Args:
            summary_folder_name: Name of target summary directory inside base_dir.
        """
        source_dir = self.base_dir
        summary_dir = os.path.join(source_dir, summary_folder_name)

        if not os.path.exists(source_dir):
            print(f"Source directory not found: {source_dir}")
            return

        os.makedirs(summary_dir, exist_ok=True)
        print(f"Collecting results into: {summary_dir}")

        grid_pattern = re.compile(r"display_grid_(\d+)_(\d+)\.png")

        for run_folder in os.listdir(source_dir):
            run_path = os.path.join(source_dir, run_folder)

            if not os.path.isdir(run_path) or run_folder == summary_folder_name:
                continue

            print(f"Processing: {run_folder}")

            loss_plot_src = os.path.join(run_path, "loss_plot.png")
            if os.path.exists(loss_plot_src):
                loss_plot_dst = os.path.join(summary_dir, f"{run_folder}_loss_plot.png")
                shutil.copy2(loss_plot_src, loss_plot_dst)
            else:
                print(f"  - No loss plot found in {run_folder}")

            grid_files = []
            for f in os.listdir(run_path):
                match = grid_pattern.match(f)
                if match:
                    iteration = int(match.group(1))
                    timestamp = int(match.group(2))
                    grid_files.append((iteration, timestamp, f))

            if grid_files:
                grid_files.sort(key=lambda x: (x[0], x[1]), reverse=True)
                last_grid_file = grid_files[0][2]

                grid_src = os.path.join(run_path, last_grid_file)
                grid_dst = os.path.join(summary_dir, f"{run_folder}_final_grid.png")
                shutil.copy2(grid_src, grid_dst)
            else:
                print(f"  - No display grid found in {run_folder}")

        print("Collection complete!")


class ResourceUsageTracker:
    """Measures optimization runtime and peak CUDA memory consumption."""

    def __init__(self, device: str):
        """Initialize resource usage tracker.

        Args:
            device: Computation device string.
        """
        self.device = device
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._cuda_tracked = False
        self._max_cuda_mem_bytes: int | None = None

    def start(self) -> None:
        """Start wall-clock timer and reset peak CUDA memory tracking."""
        self._start_time = time.time()
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            try:
                torch.cuda.reset_peak_memory_stats(device=self.device)
                self._cuda_tracked = True
            except TypeError:
                self._cuda_tracked = False

    def finish(self) -> None:
        """Stop wall-clock timer and record peak CUDA memory allocated."""
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
        """Return runtime in ms and peak CUDA memory allocated in bytes.

        Returns:
            Dictionary containing 'runtime_ms' and 'max_cuda_memory_bytes'.

        Raises:
            RuntimeError: If tracker has not been started and finished.
        """
        if self._start_time is None or self._end_time is None:
            raise RuntimeError("Tracker must be started and finished before requesting summary.")
        runtime_ms = (self._end_time - self._start_time) * 1000.0
        return {
            "runtime_ms": runtime_ms,
            "max_cuda_memory_bytes": self._max_cuda_mem_bytes,
        }


class PlotManager:
    """Handles visualization, image exporting, and loss plot saving."""

    def __init__(self, folder_manager: FolderManager):
        """Initialize plot manager.

        Args:
            folder_manager: FolderManager instance for path resolution.
        """
        self.folder_manager = folder_manager

    def save_image(
        self,
        image_tensor: torch.Tensor,
        iteration: int,
        title: str = "",
        loss: float | None = None,
        plain_image: bool = True,
    ) -> None:
        """Save a single image or parallel batch of images to disk.

        Args:
            image_tensor: Image tensor of shape [C, H, W] or [N, C, H, W].
            iteration: Current optimization iteration number.
            title: Title string (used when plain_image is False).
            loss: Optional loss value for filename tagging.
            plain_image: If True, writes direct raw pixels; if False, includes matplotlib axes/title.
        """
        if image_tensor.dim() == 3:
            batch = image_tensor.unsqueeze(0)
        else:
            batch = image_tensor

        for i in range(batch.shape[0]):
            img = batch[i]
            img_np = img.detach().cpu().permute(1, 2, 0).numpy()

            info_parts = [f"iter{iteration:04d}"]
            if batch.shape[0] > 1:
                info_parts.append(f"opt{i}")
            if loss is not None:
                info_parts.append(f"loss_{loss:.4f}")
            if self.folder_manager.timestamp:
                info_parts.append(self.folder_manager.timestamp)
            fname = "image_" + "_".join(info_parts) + ".png"
            save_path = os.path.join(self.folder_manager.run_dir, fname)

            if plain_image:
                plt.imsave(save_path, img_np)
            else:
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(img_np)
                plt.axis("off")
                if title:
                    plt.title(f"{title} (opt {i})")
                plt.savefig(save_path, bbox_inches="tight", dpi=150)
                plt.close(fig)

    def save_loss_plot(self, loss_values: list[float], iteration: int | None = None, show_plot: bool = True) -> None:
        """Save training loss curve plot.

        Args:
            loss_values: List of loss history values.
            iteration: Optional iteration number for intermediate snapshot filenames.
            show_plot: Whether to display figure in interactive environments.
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

        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        if show_plot:
            plt.show()
        plt.close()
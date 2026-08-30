"""Common test fixtures, base test case, and configuration utilities."""
import os
import shutil
import sys
import unittest
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import torch

import config
from tests.dummy_scene import DummyScene
from utils.color.linear_to_srgb_converters import LinearRec709ToAgXBase
from utils.color.tonemapping.agx_looks import AgXPunchyLook


def get_configured_device() -> str:
    """Get the target device for testing (configurable via DEVICE env var)."""
    return os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def should_use_real_olats() -> bool:
    """Whether to use real OLAT scenes (which may trigger HF downloads)."""
    return os.environ.get("USE_REAL_OLATS", "0").lower() in {"1", "true", "yes"}


def should_download_hf() -> bool:
    """Whether to test downloading large weights from Hugging Face."""
    return os.environ.get("DOWNLOAD_HF", "0").lower() in {"1", "true", "yes"}


def should_cleanup_runs() -> bool:
    """Whether to remove output test run directories after completion."""
    return os.environ.get("CLEANUP_RUNS", "0").lower() in {"1", "true", "yes"}


class MockFluxRelighter:
    """Lightweight mock relighter for fast FLUX workflow testing."""

    def __init__(self, device: str = "cpu", seed: int | None = 1):
        self.device = device
        self.seed = seed

    def _get_adjusted_width_and_height(self, width: int, height: int) -> tuple[int, int]:
        return width, height

    def relight_with_prompt(
        self,
        image_to_relight: torch.Tensor,
        prompt: str,
        num_images_per_prompt: int = 1,
        save_dir: str | None = None,
        display: bool = False,
    ) -> torch.Tensor:
        b, c, h, w = image_to_relight.shape
        return torch.rand(
            num_images_per_prompt, c, h, w, device=self.device, dtype=torch.float32
        )


class BaseWorkflowTest(unittest.TestCase):
    """Base class for example workflow tests."""

    @classmethod
    def setUpClass(cls):
        cls.device = get_configured_device()
        cls.use_real_olats = should_use_real_olats()
        cls.download_hf = should_download_hf()
        cls.cleanup = should_cleanup_runs()
        cls.color_converter = LinearRec709ToAgXBase(AgXPunchyLook())
        cls.n_test_iterations = 2

    def _get_scene(self, scene_name: str = "SciFiRobot", num_lights: int = 4, has_alpha: bool = False) -> Any:
        """Helper to get either a real OLAT scene or a fast in-memory DummyScene."""
        if self.use_real_olats:
            from examples import example_scenes

            scene_cls = getattr(example_scenes, f"{scene_name}Scene", None)
            if scene_cls is None:
                raise AttributeError(f"Scene class '{scene_name}Scene' not found in examples.example_scenes")

            if "include_alpha_mask" in scene_cls.__init__.__code__.co_varnames:
                return scene_cls(include_alpha_mask=has_alpha, device=self.device)
            return scene_cls(device=self.device)

        return DummyScene(
            num_lights=num_lights,
            height=64,
            width=64,
            has_non_optimized=True,
            has_alpha_mask=has_alpha,
            device=self.device,
            name=f"dummy_{scene_name.lower()}",
        )

    def _cleanup_dir(self, subdirectory_name: str) -> None:
        """Optional helper to delete test optimization run outputs if cleanup is explicitly enabled."""
        if not self.cleanup:
            return
        target = os.path.join(config.DEFAULT_OPTIMIZATION_RUNS_DIR, subdirectory_name)
        if os.path.exists(target):
            shutil.rmtree(target, ignore_errors=True)

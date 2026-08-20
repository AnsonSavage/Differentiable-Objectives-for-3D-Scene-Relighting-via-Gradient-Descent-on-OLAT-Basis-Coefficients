"""Test 04: Image Perceptual Losses (image_perceptual_example.ipynb)."""
from __future__ import annotations

import unittest
import torch
from losses.image_image import LPIPSLoss, SSIMLoss, VGGStyleTransferLoss
from tests.common import BaseWorkflowTest
from utils.optimize import optimize_with_criterion


class Test04ImagePerceptual(BaseWorkflowTest):
    """Mirror image_perceptual_example.ipynb with SSIM, LPIPS, and VGG-16 losses."""

    def test_image_perceptual(self):
        scene = self._get_scene("SciFiRobot", num_lights=4)
        subdir = "test_image_perceptual_example"
        target_image = torch.rand(3, 64, 64, device=self.device)

        loss_configs = [
            ("SSIM", SSIMLoss(reference_image=target_image, device=self.device)),
            ("LPIPS", LPIPSLoss(reference_image=target_image, device=self.device)),
            ("VGG-16 Style", VGGStyleTransferLoss(reference_image=target_image, backbone="vgg16", device=self.device)),
        ]

        try:
            for loss_name, criterion in loss_configs:
                optimize_with_criterion(
                    scene=scene,
                    learning_rate=0.06,
                    n_iterations=self.n_test_iterations,
                    criterion=criterion,
                    starting_multiplier_std=(0.1, 0.1, 0.1),
                    output_subdirectory_name=subdir,
                    n_results=1,
                    render_color_space_converter=self.color_converter,
                    require_physically_plausible_multipliers=True,
                    title_prefix=f"Perceptual Loss ({loss_name})",
                    device=self.device,
                    save_every=1,
                    model_name=loss_name,
                    seed=2,
                    show_images=False,
                )
        finally:
            self._cleanup_dir(subdir)


if __name__ == "__main__":
    unittest.main()

"""Test 02: Flux Diffusion Relighting (flux_relight_example.ipynb)."""

import unittest

import torch

from losses.flux_relight_loss import FLUXKontextRelighter, FluxLoss, RelightImageCache
from losses.image_image import MSELossWithReferenceImage
from tests.common import BaseWorkflowTest, MockFluxRelighter
from utils.optimize import optimize_with_criterion


class Test02FluxRelight(BaseWorkflowTest):
    """Mirror flux_relight_example.ipynb with alpha masking and FluxLoss."""

    def test_flux_relight(self):
        scene = self._get_scene("SciFiRobot", num_lights=4, has_alpha=True)
        subdir = "test_flux_relight_example"

        try:
            if self.download_hf:
                relighter = FLUXKontextRelighter(device=self.device, seed=1)
            else:
                relighter = MockFluxRelighter(device=self.device)

            image_to_relight = scene.get_combined_image(self.color_converter).permute(2, 0, 1).unsqueeze(0).to(self.device)
            images_cache = RelightImageCache(relighter, image_to_relight=image_to_relight)

            target_text = "aesthetic, golden hour lighting"
            criterion_mse = FluxLoss(
                cache=images_cache,
                image_comparison_criterion_cls=MSELossWithReferenceImage,
                target_text=target_text,
                num_relighted_images=1,
                display=False,
            )

            optimize_with_criterion(
                scene=scene,
                learning_rate=0.06,
                n_iterations=self.n_test_iterations,
                criterion=criterion_mse,
                starting_multiplier_std=0.1,
                output_subdirectory_name=subdir,
                n_results=1,
                render_color_space_converter=self.color_converter,
                require_physically_plausible_multipliers=True,
                title_prefix=f"Flux Relight ({target_text}) with MSE Loss",
                device=self.device,
                save_every=1,
                model_name="flux",
                pretrained_source="flux",
                seed=1,
                show_images=False,
            )
        finally:
            self._cleanup_dir(subdir)


if __name__ == "__main__":
    unittest.main()

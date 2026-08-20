"""Test 08: VGG Varying Light Rigs Ablation (vgg_varying_lights_ablation.ipynb)."""

import unittest

import torch

from losses.image_image import VGGStyleTransferLoss
from tests.common import BaseWorkflowTest
from tests.dummy_scene import DummyScene
from utils.optimize import optimize_with_criterion


class Test08VGGVaryingLightsAblation(BaseWorkflowTest):
    """Mirror vgg_varying_lights_ablation.ipynb benchmarking multiple light rigs."""

    def test_vgg_varying_lights_ablation(self):
        subdir = "test_vgg_varying_lights_ablation"
        target_image = torch.rand(3, 64, 64, device=self.device)

        if self.use_real_olats:
            from examples.example_scenes import CarStudioScene
            configurations = ["dome_lights", "four_small_area_lights"]
            scenes_to_test = [
                CarStudioScene(configuration=c, device=self.device) for c in configurations
            ]
        else:
            configurations = ["config_2_lights", "config_6_lights"]
            scenes_to_test = [
                DummyScene(num_lights=2, height=64, width=64, device=self.device, name="carStudio_2lights"),
                DummyScene(num_lights=6, height=64, width=64, device=self.device, name="carStudio_6lights"),
            ]

        try:
            for scene, configuration in zip(scenes_to_test, configurations):
                criterion = VGGStyleTransferLoss(reference_image=target_image, backbone="vgg16", device=self.device)

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
                    title_prefix=f"VGG Style ({configuration})",
                    device=self.device,
                    save_every=1,
                    model_name="VGG-16",
                    seed=2,
                    save_loss_plot_each_iteration=True,
                    show_images=False,
                )
        finally:
            self._cleanup_dir(subdir)


if __name__ == "__main__":
    unittest.main()

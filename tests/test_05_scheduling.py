"""Test 05: Scheduling & Parameter Constraints (scheduling_example.ipynb)."""
from __future__ import annotations

import unittest
import torch
from torchvision.transforms.v2 import Identity, RandomChoice, RandomResizedCrop

from losses.clip_like import CLIPDirectionalCosineSimilarity
from tests.common import BaseWorkflowTest
from utils.model.model_utils import create_clip_model_and_tokenizer
from utils.optimize import optimize_with_criterion


class Test05Scheduling(BaseWorkflowTest):
    """Mirror scheduling_example.ipynb with LR scheduling and dynamic augmentation."""

    def test_scheduling(self):
        scene = self._get_scene("SciFiRobot", num_lights=4)
        subdir = "test_scheduling_example"

        try:
            clip_model_name = "ViT-B-16-SigLIP-512"
            clip_pretrained = "webli"
            model, tokenizer, preprocess_eval = create_clip_model_and_tokenizer(
                clip_model_name,
                device=self.device,
                pretrained=clip_pretrained,
            )

            initial_text = "flat, unappealing lighting"
            target_text = "dramatic moody lighting"
            criterion = CLIPDirectionalCosineSimilarity(
                initial_text,
                target_text,
                scene.get_combined_image(self.color_converter).permute(2, 1, 0),
                model,
                tokenizer,
                device=self.device,
                preprocess=preprocess_eval,
            )

            def lr_scheduler_creator(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
                return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

            def aug_callback(epoch: int):
                size = model.visual.preprocess_cfg["size"] or (224, 224)
                return RandomChoice([
                    Identity(),
                    RandomResizedCrop(size=size, scale=(0.8, 1.0), antialias=True),
                ])

            def hsv_cb(epoch: int) -> str:
                return "v" if epoch == 0 else "hsv"

            optimize_with_criterion(
                scene=scene,
                learning_rate=0.04,
                n_iterations=self.n_test_iterations,
                criterion=criterion,
                starting_multiplier_std=(0.0, 0.0, 0.1),
                output_subdirectory_name=subdir,
                n_results=1,
                learning_rate_scheduler_creator_callback=lr_scheduler_creator,
                augmentation_callback=aug_callback,
                parameters_stored_as_hsv=True,
                hsv_callback=hsv_cb,
                render_color_space_converter=self.color_converter,
                require_physically_plausible_multipliers=True,
                title_prefix="Scheduling Test",
                device=self.device,
                save_every=1,
                model_name=clip_model_name,
                pretrained_source=clip_pretrained,
                seed=2,
                show_images=False,
            )
        finally:
            self._cleanup_dir(subdir)


if __name__ == "__main__":
    unittest.main()

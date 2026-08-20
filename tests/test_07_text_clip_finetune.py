"""Test 07: Text CLIP Fine-tuned vs Standard Weights (text_clip_finetune_example.ipynb)."""
from __future__ import annotations

import unittest
from torchvision.transforms.v2 import RandomChoice, RandomResizedCrop

from losses.clip_like import CLIPDirectionalCosineSimilarity
from tests.common import BaseWorkflowTest
from utils.model.model_utils import create_clip_model_and_tokenizer
from utils.optimize import optimize_with_criterion


class Test07TextCLIPFineTune(BaseWorkflowTest):
    """Mirror text_clip_finetune_example.ipynb comparing standard vs fine-tuned weights."""

    def test_text_clip_finetune(self):
        scene = self._get_scene("SciFiRobot", num_lights=4)
        subdir = "test_text_clip_finetune_example"

        models_to_test = [None]
        if self.download_hf:
            models_to_test.append("siglip_blend-training-data_64-output-dim.pt")

        try:
            for fine_tune in models_to_test:
                clip_model_name = "ViT-B-16-SigLIP-512"
                clip_pretrained = "webli"
                model, tokenizer, preprocess_eval = create_clip_model_and_tokenizer(
                    clip_model_name,
                    device=self.device,
                    pretrained=clip_pretrained,
                    fine_tune=fine_tune,
                )

                criterion = CLIPDirectionalCosineSimilarity(
                    "flat lighting",
                    "golden hour sunset",
                    scene.get_combined_image(self.color_converter).permute(2, 1, 0),
                    model,
                    tokenizer,
                    device=self.device,
                    preprocess=preprocess_eval,
                )

                size = model.visual.preprocess_cfg["size"] or (224, 224)

                optimize_with_criterion(
                    scene=scene,
                    learning_rate=0.05,
                    n_iterations=self.n_test_iterations,
                    criterion=criterion,
                    starting_multiplier_std=(0.1, 0.1, 0.1),
                    output_subdirectory_name=subdir,
                    n_results=1,
                    augmentation=RandomChoice([RandomResizedCrop(size=size, scale=(0.5, 1.0), antialias=True)]),
                    render_color_space_converter=self.color_converter,
                    require_physically_plausible_multipliers=True,
                    title_prefix="FineTune Test",
                    device=self.device,
                    save_every=1,
                    model_name=clip_model_name,
                    pretrained_source=fine_tune,
                    seed=3,
                    show_images=False,
                )
        finally:
            self._cleanup_dir(subdir)


if __name__ == "__main__":
    unittest.main()

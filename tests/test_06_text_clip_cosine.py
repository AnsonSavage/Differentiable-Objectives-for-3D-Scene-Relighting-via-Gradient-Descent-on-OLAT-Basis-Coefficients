"""Test 06: Text CLIP Cosine & Directional Similarity (text_clip_cosine_similarity_example.ipynb)."""

import unittest

from torchvision.transforms.v2 import RandomChoice, RandomResizedCrop

from losses.clip_like import CLIPCosineSimilarity, CLIPDirectionalCosineSimilarity
from tests.common import BaseWorkflowTest
from utils.model.model_utils import create_clip_model_and_tokenizer
from utils.optimize import optimize_with_criterion


class Test06TextCLIPCosine(BaseWorkflowTest):
    """Mirror text_clip_cosine_similarity_example.ipynb with CLIP cosine similarity."""

    def test_text_clip_cosine(self):
        scene = self._get_scene("SciFiRobot", num_lights=4)
        subdir = "test_text_clip_cosine_similarity_example"

        try:
            clip_model_name = "ViT-B-16-SigLIP-512"
            clip_pretrained = "webli"
            model, tokenizer, preprocess_eval = create_clip_model_and_tokenizer(
                clip_model_name,
                device=self.device,
                pretrained=clip_pretrained,
            )

            # Test Directional CLIP
            criterion_dir = CLIPDirectionalCosineSimilarity(
                "flat lighting",
                "dramatic moody cinematic lighting",
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
                criterion=criterion_dir,
                starting_multiplier_std=(0.1, 0.1, 0.1),
                output_subdirectory_name=subdir,
                n_results=1,
                augmentation=RandomChoice([RandomResizedCrop(size=size, scale=(0.5, 1.0), antialias=True)]),
                render_color_space_converter=self.color_converter,
                require_physically_plausible_multipliers=True,
                title_prefix="Directional CLIP Test",
                device=self.device,
                save_every=1,
                model_name=clip_model_name,
                pretrained_source=clip_pretrained,
                seed=2,
                show_images=False,
            )

            # Test Standard CLIP
            criterion_std = CLIPCosineSimilarity(
                "warm sunset lighting",
                model,
                tokenizer,
                device=self.device,
                preprocess=preprocess_eval,
            )

            optimize_with_criterion(
                scene=scene,
                learning_rate=0.05,
                n_iterations=self.n_test_iterations,
                criterion=criterion_std,
                starting_multiplier_std=(0.1, 0.1, 0.1),
                output_subdirectory_name=subdir,
                n_results=1,
                render_color_space_converter=self.color_converter,
                require_physically_plausible_multipliers=True,
                title_prefix="Standard CLIP Test",
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

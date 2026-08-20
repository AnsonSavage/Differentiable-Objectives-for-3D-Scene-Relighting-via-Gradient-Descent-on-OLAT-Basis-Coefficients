"""Test 03: Image CLIP Embedding Cosine Similarity (image_clip_cosine_example.ipynb)."""

import unittest

import torch

from losses.image_image import ImageImageCLIPLoss
from tests.common import BaseWorkflowTest
from utils.model.model_utils import create_clip_model_and_tokenizer
from utils.optimize import optimize_with_criterion


class Test03ImageCLIPCosine(BaseWorkflowTest):
    """Mirror image_clip_cosine_example.ipynb using ImageImageCLIPLoss."""

    def test_image_clip_cosine(self):
        scene = self._get_scene("SciFiRobot", num_lights=4)
        subdir = "test_image_clip_cosine_example"

        try:
            clip_model_name = "ViT-B-16-SigLIP-512"
            clip_pretrained = "webli"
            fine_tune = "siglip_blend-training-data_64-output-dim.pt" if self.download_hf else None

            model, tokenizer, preprocess_eval = create_clip_model_and_tokenizer(
                clip_model_name,
                device=self.device,
                pretrained=clip_pretrained,
                fine_tune=fine_tune,
            )

            # Synthetic reference target image tensor
            target_image = torch.rand(3, 64, 64, device=self.device)

            criterion = ImageImageCLIPLoss(
                reference_image=target_image,
                clip_model=model,
                preprocess=preprocess_eval,
                device=self.device,
            )

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
                title_prefix=f"ImageImageCLIP ({clip_model_name})",
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

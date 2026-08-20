"""Test 01: Aesthetic Score Optimization (aesthetic_score_example.ipynb)."""
from __future__ import annotations

import unittest
from losses.aesthetic import AestheticLossMaximize, LAIONAestheticScorer
from tests.common import BaseWorkflowTest
from utils.optimize import optimize_with_criterion


class Test01AestheticScore(BaseWorkflowTest):
    """Mirror aesthetic_score_example.ipynb using LAIONAestheticScorer."""

    def test_aesthetic_score(self):
        scene = self._get_scene("SciFiRobot", num_lights=4)
        subdir = "test_aesthetic_score_example"

        try:
            scorer = LAIONAestheticScorer(device=self.device)
            criterion = AestheticLossMaximize(scorer)

            optimize_with_criterion(
                scene=scene,
                learning_rate=0.04,
                n_iterations=self.n_test_iterations,
                criterion=criterion,
                starting_multiplier_std=0.1,
                output_subdirectory_name=subdir,
                n_results=1,
                render_color_space_converter=self.color_converter,
                require_physically_plausible_multipliers=True,
                title_prefix="Aesthetic Score Test",
                device=self.device,
                save_every=1,
                show_images=False,
            )
        finally:
            self._cleanup_dir(subdir)


if __name__ == "__main__":
    unittest.main()

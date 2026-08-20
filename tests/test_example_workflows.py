"""Aggregated test case combining all 8 workflow test classes for unittest discovery."""
from __future__ import annotations

import unittest
from tests.test_01_aesthetic_score import Test01AestheticScore
from tests.test_02_flux_relight import Test02FluxRelight
from tests.test_03_image_clip_cosine import Test03ImageCLIPCosine
from tests.test_04_image_perceptual import Test04ImagePerceptual
from tests.test_05_scheduling import Test05Scheduling
from tests.test_06_text_clip_cosine import Test06TextCLIPCosine
from tests.test_07_text_clip_finetune import Test07TextCLIPFineTune
from tests.test_08_vgg_varying_lights_ablation import Test08VGGVaryingLightsAblation


class TestExampleWorkflows(
    Test01AestheticScore,
    Test02FluxRelight,
    Test03ImageCLIPCosine,
    Test04ImagePerceptual,
    Test05Scheduling,
    Test06TextCLIPCosine,
    Test07TextCLIPFineTune,
    Test08VGGVaryingLightsAblation,
):
    """Aggregate test suite containing all 8 workflow tests."""
    pass


if __name__ == "__main__":
    unittest.main()

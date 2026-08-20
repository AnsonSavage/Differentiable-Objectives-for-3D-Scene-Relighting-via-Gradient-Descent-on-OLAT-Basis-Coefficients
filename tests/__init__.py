"""Test package for Differentiable Objectives 3D Scene Relighting.

Allows running individual tests, subsets of tests, or all tests either programmatically
or via the command-line test runner.

Python Usage:
    >>> import tests
    >>> tests.list_tests()
    >>> tests.run(1)                       # Run test 1
    >>> tests.run([1, 3, 5])               # Run tests 1, 3, and 5
    >>> tests.run('aesthetic')             # Run by keyword
    >>> tests.run(['flux', 'scheduling'])  # Run multiple by keywords
    >>> tests.run_all()                    # Run all tests
    >>> tests.run_interactive()            # Interactive prompt to choose tests
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Type

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class TestItem:
    id: int
    key: str
    title: str
    notebook: str
    module_name: str
    class_name: str

    def get_test_class(self) -> Type[unittest.TestCase]:
        """Lazy-load the test class so listing and importing remain instantaneous."""
        mod = importlib.import_module(f"tests.{self.module_name}")
        return getattr(mod, self.class_name)


TEST_REGISTRY: list[TestItem] = [
    TestItem(
        id=1,
        key="aesthetic_score",
        title="Aesthetic Score Optimization",
        notebook="aesthetic_score_example.ipynb",
        module_name="test_01_aesthetic_score",
        class_name="Test01AestheticScore",
    ),
    TestItem(
        id=2,
        key="flux_relight",
        title="Flux Diffusion Relighting",
        notebook="flux_relight_example.ipynb",
        module_name="test_02_flux_relight",
        class_name="Test02FluxRelight",
    ),
    TestItem(
        id=3,
        key="image_clip_cosine",
        title="Image CLIP Embedding Cosine Similarity",
        notebook="image_clip_cosine_example.ipynb",
        module_name="test_03_image_clip_cosine",
        class_name="Test03ImageCLIPCosine",
    ),
    TestItem(
        id=4,
        key="image_perceptual",
        title="Image Perceptual Losses (SSIM, LPIPS, VGG-16)",
        notebook="image_perceptual_example.ipynb",
        module_name="test_04_image_perceptual",
        class_name="Test04ImagePerceptual",
    ),
    TestItem(
        id=5,
        key="scheduling",
        title="Scheduling & Parameter Constraints",
        notebook="scheduling_example.ipynb",
        module_name="test_05_scheduling",
        class_name="Test05Scheduling",
    ),
    TestItem(
        id=6,
        key="text_clip_cosine",
        title="Text CLIP Cosine & Directional Similarity",
        notebook="text_clip_cosine_similarity_example.ipynb",
        module_name="test_06_text_clip_cosine",
        class_name="Test06TextCLIPCosine",
    ),
    TestItem(
        id=7,
        key="text_clip_finetune",
        title="Text CLIP Fine-tuned vs Standard Weights",
        notebook="text_clip_finetune_example.ipynb",
        module_name="test_07_text_clip_finetune",
        class_name="Test07TextCLIPFineTune",
    ),
    TestItem(
        id=8,
        key="vgg_varying_lights",
        title="VGG Varying Light Rigs Ablation",
        notebook="vgg_varying_lights_ablation.ipynb",
        module_name="test_08_vgg_varying_lights_ablation",
        class_name="Test08VGGVaryingLightsAblation",
    ),
]


def list_tests() -> list[TestItem]:
    """Prints and returns all registered tests."""
    print("\nAvailable Tests:")
    print("-" * 75)
    for item in TEST_REGISTRY:
        print(f"  [{item.id}] {item.title}")
        print(f"      Key: '{item.key}' | Source: {item.notebook}")
    print("-" * 75 + "\n")
    return list(TEST_REGISTRY)


def _resolve_selection(selection: Any) -> list[TestItem]:
    """Resolve user selection (int, string, sequence of ints/strings) to TestItems."""
    if selection is None or selection == "all" or selection == ["all"]:
        return list(TEST_REGISTRY)

    if isinstance(selection, (int, str)):
        items_to_match = [selection]
    else:
        items_to_match = list(selection)

    resolved: list[TestItem] = []
    for sel in items_to_match:
        matched = False
        if isinstance(sel, int) or (isinstance(sel, str) and sel.strip().isdigit()):
            target_id = int(sel)
            for item in TEST_REGISTRY:
                if item.id == target_id and item not in resolved:
                    resolved.append(item)
                    matched = True
                    break
        elif isinstance(sel, str):
            query = sel.strip().lower()
            for item in TEST_REGISTRY:
                if (query in item.key.lower() or query in item.title.lower() or query in item.notebook.lower()) and item not in resolved:
                    resolved.append(item)
                    matched = True

        if not matched:
            print(f"Warning: No tests matched '{sel}'")

    return resolved


def run(
    tests: int | str | Sequence[int | str] | None = None,
    device: str | None = None,
    real_olats: bool = False,
    download_hf: bool = False,
    cleanup: bool = False,
    verbose: bool = False,
) -> bool:
    """Run specified tests or all tests.

    Args:
        tests: Test ID (e.g. 1), test key/substring ('aesthetic'), list of IDs/keys, or None for all.
        device: Device to use ('cpu' or 'cuda').
        real_olats: Whether to test against real scenes (downloads if needed).
        download_hf: Whether to test downloading fine-tuned weights from Hugging Face.
        cleanup: Whether to delete generated test output directories (default False: saves results to OPTIMIZATION_RUNS/).
        verbose: Verbose output.

    Returns:
        bool: True if all selected tests passed, False otherwise.
    """
    if device:
        os.environ["DEVICE"] = device
    if real_olats:
        os.environ["USE_REAL_OLATS"] = "1"
    if download_hf:
        os.environ["DOWNLOAD_HF"] = "1"
    os.environ["CLEANUP_RUNS"] = "1" if cleanup else "0"

    selected_items = _resolve_selection(tests)
    if not selected_items:
        print("No valid tests selected to run.")
        return False

    import torch
    target_device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("  3D SCENE RELIGHTING TEST RUNNER")
    print("=" * 75)
    print(f"  Target Device     : {target_device}")
    print(f"  Scene Mode        : {'Real OLAT Datasets (HuggingFace)' if real_olats else 'Fast In-Memory DummyScenes'}")
    print(f"  HF Checkpoints    : {'Download & Test' if download_hf else 'Standard Pretrained (Fast)'}")
    print(f"  Output Directory  : {'Deleted after test (--cleanup)' if cleanup else 'Saved to OPTIMIZATION_RUNS/'}")
    print(f"  Selected Tests    : {len(selected_items)} of {len(TEST_REGISTRY)}")
    print("=" * 75)

    passed = 0
    failed = 0
    errors: list[tuple[str, str]] = []
    overall_start = time.time()

    loader = unittest.TestLoader()

    for idx, item in enumerate(selected_items, start=1):
        print(f"\n[{idx}/{len(selected_items)}] Running: [{item.id}] {item.title} ({item.notebook}) ...", flush=True)
        t0 = time.time()
        try:
            cls = item.get_test_class()
            suite = loader.loadTestsFromTestCase(cls)
            result = unittest.TestResult()
            suite.run(result)
            t_elapsed = time.time() - t0

            if result.wasSuccessful():
                print(f"      \033[92mPASSED\033[0m ({t_elapsed:.2f}s)", flush=True)
                passed += 1
            else:
                print(f"      \033[91mFAILED\033[0m ({t_elapsed:.2f}s)", flush=True)
                failed += 1
                for failure in result.failures + result.errors:
                    test_case, err = failure
                    errors.append((item.title, err))
        except Exception as e:
            t_elapsed = time.time() - t0
            import traceback
            print(f"      \033[91mFAILED\033[0m ({t_elapsed:.2f}s): {e}", flush=True)
            failed += 1
            errors.append((item.title, traceback.format_exc()))

    total_time = time.time() - overall_start

    print("\n" + "=" * 75)
    print("  TEST SUMMARY")
    print("=" * 75)
    print(f"  Total Run     : {len(selected_items)}")
    print(f"  Passed        : \033[92m{passed}\033[0m")
    print(f"  Failed        : \033[91m{failed}\033[0m" if failed else f"  Failed        : 0")
    print(f"  Total Time    : {total_time:.2f}s")
    print("=" * 75)

    if errors:
        print("\nFailures Detail:")
        for name, msg in errors:
            print(f"\n--- {name} ---")
            print(msg)
        return False

    print("\n\033[92mALL SELECTED TESTS COMPLETED SUCCESSFULLY!\033[0m\n")
    return True


def run_all(**kwargs) -> bool:
    """Run all registered tests."""
    return run(tests=None, **kwargs)


def run_interactive() -> bool:
    """Interactive CLI menu to choose and run tests."""
    print("=" * 75)
    print("  3D SCENE RELIGHTING - INTERACTIVE TEST SELECTION")
    print("=" * 75)
    list_tests()
    print("Enter test numbers separated by spaces (e.g., '1 3 5'), a keyword (e.g., 'clip'),")
    print("or press Enter / type 'all' to run all tests.")
    try:
        user_input = input("\nSelect tests > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return False

    if not user_input or user_input.lower() == "all":
        return run_all()

    tokens = [tok for tok in user_input.replace(",", " ").split() if tok]
    return run(tokens)


__all__ = [
    "TEST_REGISTRY",
    "TestItem",
    "list_tests",
    "run",
    "run_all",
    "run_interactive",
]

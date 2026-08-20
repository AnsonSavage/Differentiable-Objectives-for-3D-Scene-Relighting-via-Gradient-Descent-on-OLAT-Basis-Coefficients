"""CLI entry point for running tests via `python -m tests`."""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tests


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Modular test runner for Differentiable Objectives 3D Scene Relighting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t", "--tests",
        nargs="*",
        default=None,
        help="One or more test IDs (e.g. 1 3 5) or keywords (e.g. aesthetic scheduling).",
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Run all tests.",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available tests.",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactively select tests from a menu.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--real-olats",
        action="store_true",
        help="Use real OLAT datasets from examples/ (downloads from HF if needed).",
    )
    parser.add_argument(
        "--download-hf",
        action="store_true",
        help="Download and test fine-tuned model checkpoints from Hugging Face.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete generated test optimization run outputs on disk (by default, runs are saved in OPTIMIZATION_RUNS/).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output.",
    )

    args = parser.parse_args()

    if args.list:
        tests.list_tests()
        return 0

    if args.interactive:
        success = tests.run_interactive()
        return 0 if success else 1

    selected = args.tests
    if args.all or (selected is None and not args.interactive):
        selected = None

    success = tests.run(
        tests=selected,
        device=args.device,
        real_olats=args.real_olats,
        download_hf=args.download_hf,
        cleanup=args.cleanup,
        verbose=args.verbose,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

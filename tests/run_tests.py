#!/usr/bin/env python3
"""
Test runner script for the Differentiable Objectives 3D Relighting repository.

Supports selecting specific tests by index (e.g. 1 3 5), by keyword ('aesthetic'),
or running the full test suite.

Usage:
    python tests/run_tests.py                      # Run all tests
    python tests/run_tests.py -t 1 3               # Run tests 1 and 3
    python tests/run_tests.py -t aesthetic clip    # Run by keywords
    python tests/run_tests.py --list               # List all available tests
    python tests/run_tests.py --interactive        # Prompt interactively
    python tests/run_tests.py --device cpu         # Force CPU execution
    python tests/run_tests.py --real-olats         # Use real OLAT scenes
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.__main__ import main

if __name__ == "__main__":
    sys.exit(main())

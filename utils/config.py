"""Global project configuration settings."""

import os
from pathlib import Path

# Project root directory (parent of utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default base directory for optimization experiment runs
# Can be overridden by setting the OPTIMIZATION_RUNS_DIR environment variable
DEFAULT_OPTIMIZATION_RUNS_DIR = os.environ.get(
    "OPTIMIZATION_RUNS_DIR",
    str(PROJECT_ROOT / "OPTIMIZATION_RUNS"),
)

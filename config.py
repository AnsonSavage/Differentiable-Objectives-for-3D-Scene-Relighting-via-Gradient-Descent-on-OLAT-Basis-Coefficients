"""Global project configuration settings."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Default base directory for optimization experiment runs
# Can be overridden by setting the OPTIMIZATION_RUNS_DIR environment variable
DEFAULT_OPTIMIZATION_RUNS_DIR = os.environ.get(
    "OPTIMIZATION_RUNS_DIR",
    str(PROJECT_ROOT / "OPTIMIZATION_RUNS"),
)

# Default base directory for model weights and cached models
DEFAULT_MODEL_WEIGHTS_DIR = os.environ.get(
    "MODEL_WEIGHTS_DIR",
    str(PROJECT_ROOT / "MODEL_WEIGHTS"),
)
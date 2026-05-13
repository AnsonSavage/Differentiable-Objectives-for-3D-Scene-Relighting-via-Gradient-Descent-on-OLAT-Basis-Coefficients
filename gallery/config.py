"""Configuration for the gallery app."""
import os
from pathlib import Path

# Base directory is the gallery folder
BASE_DIR = Path(__file__).parent

# Runs directory - must be provided via environment variable.
# TODO: PATH_UPDATE gallery runs directory
RUNS_DIR_NAME = os.environ.get("GALLERY_RUNS_DIR", "")

# Resolve the runs directory path.
if not RUNS_DIR_NAME:
    raise ValueError("GALLERY_RUNS_DIR must be set. TODO: PATH_UPDATE gallery runs directory")
if Path(RUNS_DIR_NAME).is_absolute():
    RUNS_DIR = Path(RUNS_DIR_NAME)
else:
    RUNS_DIR = BASE_DIR.parent / RUNS_DIR_NAME

# Favorites file location
# NOTE: favorites.json is now treated as the "hearts" list (pretty images).
FAVORITES_FILE = BASE_DIR / "favorites.json"

# Stars file location (prompt/image matches)
STARS_FILE = BASE_DIR / "stars.json"

# Image extensions to recognize
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

def get_runs_dir_relative_to_static() -> str:
    """
    Get the runs directory path relative to the static folder for frontend use.
    This returns a relative path for use in JavaScript/HTML.
    """
    try:
        # Get relative path from static folder to runs directory
        static_dir = BASE_DIR / "static"
        rel_path = os.path.relpath(RUNS_DIR, static_dir)
        # Convert to forward slashes for web compatibility
        return rel_path.replace(os.sep, '/')
    except ValueError:
        # If on different drives (Windows), return absolute path
        return str(RUNS_DIR)

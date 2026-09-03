"""Configuration and persistent settings for the web gallery application."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Base directory is the gallery folder
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]

# Default runs directory matches the default output folder of the example notebooks
DEFAULT_RUNS_DIR = REPO_ROOT / "OPTIMIZATION_RUNS"

# Persist the selected runs directory so it survives between sessions.
SETTINGS_FILE = BASE_DIR / "gallery_settings.json"

# Favorites file location (hearts list - pretty images)
FAVORITES_FILE = BASE_DIR / "favorites.json"

# Stars file location (stars list - prompt/image matches)
STARS_FILE = BASE_DIR / "stars.json"

# Image extensions to recognize
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

RUNS_DIR_NAME: str | None = None
RUNS_DIR: Path | None = None


def _load_settings() -> dict[str, Any]:
    """Load persisted gallery settings from JSON file."""
    if not SETTINGS_FILE.exists():
        return {}
    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_settings(settings: dict[str, Any]) -> None:
    """Save gallery settings dictionary to JSON file."""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_FILE.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, sort_keys=True)


def _resolve_runs_dir(value: str) -> Path:
    """Resolve relative or expanded path string against the repository root."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def get_saved_runs_dir_name() -> str | None:
    """Get the currently saved runs directory name from settings, if set.

    Returns:
        String path or None if unset.
    """
    settings = _load_settings()
    value = settings.get("runs_dir")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def get_runs_dir() -> Path:
    """Get the resolved Path to the active runs directory.

    Returns:
        Path object pointing to the active experiment runs directory.
    """
    global RUNS_DIR
    if RUNS_DIR is None:
        refresh_runs_dir_from_settings()
    assert RUNS_DIR is not None
    return RUNS_DIR


def set_runs_dir_name(value: str) -> Path:
    """Update and persist the active experiment runs directory.

    Args:
        value: Path string to the new runs directory.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        NotADirectoryError: If the specified path is not a directory.
    """
    cleaned = value.strip()
    global RUNS_DIR_NAME, RUNS_DIR

    if not cleaned or cleaned.lower() == "default":
        DEFAULT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        _save_settings({})
        RUNS_DIR_NAME = str(DEFAULT_RUNS_DIR)
        RUNS_DIR = DEFAULT_RUNS_DIR
        return DEFAULT_RUNS_DIR

    resolved = _resolve_runs_dir(cleaned)
    if not resolved.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {cleaned}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Runs directory is not a directory: {cleaned}")

    _save_settings({"runs_dir": cleaned})
    RUNS_DIR_NAME = cleaned
    RUNS_DIR = resolved
    return resolved


def reset_to_default_runs_dir() -> Path:
    """Reset the active runs directory to DEFAULT_RUNS_DIR.

    Returns:
        Resolved Path object pointing to DEFAULT_RUNS_DIR.
    """
    return set_runs_dir_name("")


def refresh_runs_dir_from_settings() -> Path:
    """Reload active runs directory from persisted settings or fallback to default.

    Returns:
        Active runs directory Path.
    """
    global RUNS_DIR_NAME, RUNS_DIR
    saved = get_saved_runs_dir_name()

    if saved:
        try:
            resolved = _resolve_runs_dir(saved)
            if resolved.exists() and resolved.is_dir():
                RUNS_DIR_NAME = saved
                RUNS_DIR = resolved
                return resolved
        except Exception:
            pass

    DEFAULT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR_NAME = str(DEFAULT_RUNS_DIR)
    RUNS_DIR = DEFAULT_RUNS_DIR
    return DEFAULT_RUNS_DIR


# Initialize on load
refresh_runs_dir_from_settings()


def get_runs_dir_relative_to_static() -> str:
    """Get the runs directory path relative to the static folder for frontend use.

    Returns:
        Relative path string with forward slashes.
    """
    runs_dir = get_runs_dir()
    try:
        static_dir = BASE_DIR / "static"
        rel_path = Path(os.path.relpath(runs_dir, static_dir))
        return str(rel_path).replace("\\", "/")
    except Exception:
        try:
            rel_path = os.path.relpath(runs_dir, BASE_DIR / "static")
            return rel_path.replace(os.sep, "/")
        except ValueError:
            return str(runs_dir)

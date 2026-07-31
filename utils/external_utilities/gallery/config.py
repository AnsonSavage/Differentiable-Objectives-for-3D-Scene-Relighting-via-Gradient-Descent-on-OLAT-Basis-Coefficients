"""Configuration for the gallery app."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Base directory is the gallery folder
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]

# Persist the selected runs directory so it survives between sessions.
SETTINGS_FILE = BASE_DIR / "gallery_settings.json"

# Favorites file location
# NOTE: favorites.json is now treated as the "hearts" list (pretty images).
FAVORITES_FILE = BASE_DIR / "favorites.json"

# Stars file location (prompt/image matches)
STARS_FILE = BASE_DIR / "stars.json"

# Image extensions to recognize
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

RUNS_DIR_NAME: str | None = None
RUNS_DIR: Path | None = None


def _load_settings() -> dict[str, Any]:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_settings(settings: dict[str, Any]) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_FILE.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, sort_keys=True)


def _resolve_runs_dir(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def get_saved_runs_dir_name() -> str | None:
    settings = _load_settings()
    value = settings.get("runs_dir")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def get_runs_dir() -> Path | None:
    return RUNS_DIR


def is_configured() -> bool:
    return RUNS_DIR is not None


def set_runs_dir_name(value: str) -> Path:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Runs directory cannot be empty")

    resolved = _resolve_runs_dir(cleaned)
    if not resolved.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {cleaned}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Runs directory is not a directory: {cleaned}")

    _save_settings({"runs_dir": cleaned})

    global RUNS_DIR_NAME, RUNS_DIR
    RUNS_DIR_NAME = cleaned
    RUNS_DIR = resolved
    return resolved


def refresh_runs_dir_from_settings() -> Path | None:
    global RUNS_DIR_NAME, RUNS_DIR
    saved = get_saved_runs_dir_name()
    if not saved:
        RUNS_DIR_NAME = None
        RUNS_DIR = None
        return None

    try:
        resolved = _resolve_runs_dir(saved)
    except Exception:
        RUNS_DIR_NAME = saved
        RUNS_DIR = None
        return None

    if not resolved.exists() or not resolved.is_dir():
        RUNS_DIR_NAME = saved
        RUNS_DIR = None
        return None

    RUNS_DIR_NAME = saved
    RUNS_DIR = resolved
    return resolved


refresh_runs_dir_from_settings()


def get_runs_dir_relative_to_static() -> str:
    """
    Get the runs directory path relative to the static folder for frontend use.
    This returns a relative path for use in JavaScript/HTML.
    """
    if RUNS_DIR is None:
        return ""

    try:
        static_dir = BASE_DIR / "static"
        rel_path = Path(os.path.relpath(RUNS_DIR, static_dir))
        return str(rel_path).replace("\\", "/")
    except Exception:
        try:
            rel_path = os.path.relpath(RUNS_DIR, BASE_DIR / "static")
            return rel_path.replace(os.sep, "/")
        except ValueError:
            return str(RUNS_DIR)

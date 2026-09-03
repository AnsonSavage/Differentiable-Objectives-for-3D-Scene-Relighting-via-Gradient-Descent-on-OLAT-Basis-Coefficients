"""Flask web application for browsing and filtering relighting experiment runs."""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_file

# Ensure local gallery directory can be imported reliably
_gallery_dir = Path(__file__).resolve().parent
if str(_gallery_dir) not in sys.path:
    sys.path.insert(0, str(_gallery_dir))

import gallery_config

app = Flask(__name__, template_folder="templates", static_folder="static")
_lock = threading.Lock()

# Simple caches (invalidated manually if needed)
_scenes_cache: set[str] | None = None
_prompts_cache: dict[str, set[str]] = {}
_loss_models_cache: dict[str, set[str]] = {}
_reference_images_cache: dict[str, set[str]] = {}


def _clear_gallery_caches() -> None:
    """Invalidate all in-memory metadata caches."""
    global _scenes_cache, _prompts_cache, _loss_models_cache, _reference_images_cache
    _scenes_cache = None
    _prompts_cache = {}
    _loss_models_cache = {}
    _reference_images_cache = {}


def _get_runs_dir() -> Path:
    """Return active experiment runs directory."""
    return gallery_config.get_runs_dir()


def load_json(path: Path) -> dict[str, Any]:
    """Safely load JSON dictionary from a file path.

    Args:
        path: Path to JSON file.

    Returns:
        Loaded dictionary or empty dict on failure.
    """
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def get_settings_path(run_dir: Path) -> Path:
    """Return path to settings.json in a run folder."""
    return run_dir / "settings.json"


def get_all_runs(subdir: str | None = None) -> list[Path]:
    """Get all run directory paths, optionally within a subdirectory.

    Args:
        subdir: Optional subdirectory name.

    Returns:
        List of Path objects for each run folder.
    """
    runs_dir = _get_runs_dir()
    if runs_dir is None or not runs_dir.exists():
        return []

    base_path = runs_dir / subdir if subdir else runs_dir
    if not base_path.exists():
        return []

    return [p for p in base_path.iterdir() if p.is_dir()]


def get_subdirectories() -> list[str]:
    """Get sorted list of subdirectories within the active runs folder.

    Returns:
        List of subdirectory name strings.
    """
    runs_dir = _get_runs_dir()
    if runs_dir is None or not runs_dir.exists():
        return []
    return sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])


def extract_scene_from_run_dir(run_dir: Path) -> str:
    """Extract scene name from run folder naming convention (<scene>_<criterion>_...)."""
    name = run_dir.name
    parts = name.split("_")
    return parts[0]


def gather_scenes(subdir: str | None = None) -> list[str]:
    """Gather unique scene names from all run directories.

    Args:
        subdir: Optional subdirectory filter.

    Returns:
        Sorted list of unique scene names.
    """
    global _scenes_cache
    if subdir is None and _scenes_cache is not None:
        return sorted(_scenes_cache)
    scenes: set[str] = set()
    for run_dir in get_all_runs(subdir):
        scenes.add(extract_scene_from_run_dir(run_dir))
    if subdir is None:
        _scenes_cache = scenes
    return sorted(scenes)


def gather_prompts_for_scene(scene: str, subdir: str | None = None) -> list[str]:
    """Gather unique text prompts used for a scene across runs.

    Args:
        scene: Scene identifier string.
        subdir: Optional subdirectory filter.

    Returns:
        Sorted list of prompt strings.
    """
    scene = scene.strip()
    if not scene:
        return []
    if scene in _prompts_cache:
        return sorted(_prompts_cache[scene])
    prompts: set[str] = set()
    for run_dir in get_all_runs(subdir):
        if extract_scene_from_run_dir(run_dir) != scene:
            continue
        settings_path = get_settings_path(run_dir)
        if not settings_path.exists():
            continue
        settings = load_json(settings_path)
        prompts_dict = settings.get("prompts", {})
        tp = prompts_dict.get("clip_target_text_prompt") or prompts_dict.get("clip_text_prompt") or prompts_dict.get("target_text")
        if tp:
            prompts.add(tp)
    _prompts_cache[scene] = prompts
    return sorted(prompts)


def _extract_loss_model_from_settings(settings: dict[str, Any]) -> str | None:
    """Extract vision/CLIP model name from run settings."""
    clip_model = settings.get("clip_model") or {}
    name = clip_model.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _extract_reference_image_from_settings(settings: dict[str, Any]) -> str | None:
    """Extract reference image filename from settings."""
    prompts = settings.get("prompts") or {}
    ref_path = prompts.get("reference_image_path")
    if isinstance(ref_path, str) and ref_path.strip():
        import os
        return os.path.basename(ref_path.strip())
    return None


def gather_loss_models_for_scene(scene: str, subdir: str | None = None) -> list[str]:
    """Gather unique loss model architectures used for a scene.

    Args:
        scene: Scene name.
        subdir: Optional subdirectory filter.

    Returns:
        Sorted list of loss model strings.
    """
    scene = scene.strip()
    if not scene:
        return []
    if subdir is None and scene in _loss_models_cache:
        return sorted(_loss_models_cache[scene])
    loss_models: set[str] = set()
    for run_dir in get_all_runs(subdir):
        if extract_scene_from_run_dir(run_dir) != scene:
            continue
        settings_path = get_settings_path(run_dir)
        if not settings_path.exists():
            continue
        settings = load_json(settings_path)
        lm = _extract_loss_model_from_settings(settings)
        if lm:
            loss_models.add(lm)
    if subdir is None:
        _loss_models_cache[scene] = loss_models
    return sorted(loss_models)


def gather_reference_images_for_scene(scene: str, subdir: str | None = None) -> list[str]:
    """Gather unique reference image filenames for a scene.

    Args:
        scene: Scene name.
        subdir: Optional subdirectory filter.

    Returns:
        Sorted list of reference image filenames.
    """
    scene = scene.strip()
    if not scene:
        return []
    if subdir is None and scene in _reference_images_cache:
        return sorted(_reference_images_cache[scene])
    ref_images: set[str] = set()
    for run_dir in get_all_runs(subdir):
        if extract_scene_from_run_dir(run_dir) != scene:
            continue
        settings_path = get_settings_path(run_dir)
        if not settings_path.exists():
            continue
        settings = load_json(settings_path)
        ref = _extract_reference_image_from_settings(settings)
        if ref:
            ref_images.add(ref)
    if subdir is None:
        _reference_images_cache[scene] = ref_images
    return sorted(ref_images)


def collect_images(
    scene: str,
    target_prompt: str | None = None,
    loss_model: str | None = None,
    reference_image: str | None = None,
    subdir: str | None = None,
) -> list[dict[str, Any]]:
    """Collect images for a scene, optionally filtered by prompt, model, and reference.

    Args:
        scene: Scene identifier string.
        target_prompt: Optional prompt text filter.
        loss_model: Optional loss model filter.
        reference_image: Optional reference image filter.
        subdir: Optional subdirectory filter.

    Returns:
        List of image metadata dictionaries.
    """
    images: list[dict[str, Any]] = []
    runs_dir = _get_runs_dir()
    if runs_dir is None:
        return images
    for run_dir in get_all_runs(subdir):
        if extract_scene_from_run_dir(run_dir) != scene:
            continue
        settings_path = get_settings_path(run_dir)
        if not settings_path.exists():
            continue
        settings = load_json(settings_path)
        prompts_dict = settings.get("prompts", {})
        prompt = prompts_dict.get("clip_target_text_prompt") or prompts_dict.get("clip_text_prompt") or prompts_dict.get("target_text")
        if target_prompt and prompt != target_prompt:
            continue

        run_loss_model = _extract_loss_model_from_settings(settings)
        if loss_model and run_loss_model != loss_model:
            continue

        run_ref_image = _extract_reference_image_from_settings(settings)
        if reference_image and run_ref_image != reference_image:
            continue

        run_images: list[dict[str, Any]] = []
        for img_path in sorted(run_dir.glob("image_iter*_loss_*.png")):
            relative_path = str(img_path.relative_to(runs_dir))
            iteration = _parse_iteration(img_path.name)
            opt_index = _parse_opt_index(img_path.name)
            run_images.append({
                "id": relative_path,
                "run": run_dir.name,
                "filename": img_path.name,
                "path": relative_path,
                "loss": _parse_loss(img_path.name),
                "iteration": iteration,
                "opt_index": opt_index,
            })
        if run_images and all(img.get("opt_index", -1) < 0 for img in run_images):
            for img in run_images:
                img["opt_index"] = 0
        images.extend(run_images)
    images.sort(key=lambda img: (
        img.get("run", ""),
        img.get("iteration", -1),
        img.get("opt_index", -1),
        img.get("filename", ""),
    ))
    return images


def _parse_iteration(filename: str) -> int:
    """Parse iteration integer from image filename."""
    try:
        part = filename.split("iter")[1]
        return int(part.split("_")[0])
    except Exception:
        return -1


def _parse_loss(filename: str) -> float:
    """Parse loss float value from image filename."""
    try:
        after = filename.split("loss_")[1]
        return float(after.split("_")[0])
    except Exception:
        return float("nan")


def _parse_opt_index(filename: str) -> int:
    """Parse optimization solution index from image filename."""
    try:
        match = re.search(r"_opt(\d+)_", filename)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return -1


def _load_image_list(path: Path) -> dict[str, Any]:
    """Load JSON file containing image list."""
    if path.exists():
        try:
            with path.open("r") as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("images", []), list):
                    return data
        except Exception:
            pass
    return {"images": []}


def _save_image_list(path: Path, data: dict[str, Any]) -> None:
    """Thread-safe save of image list data to JSON."""
    with _lock:
        with path.open("w") as f:
            json.dump(data, f, indent=2)


def load_hearts() -> dict[str, Any]:
    """Load favorited images list."""
    return _load_image_list(gallery_config.FAVORITES_FILE)


def save_hearts(hearts: dict[str, Any]) -> None:
    """Save favorited images list."""
    _save_image_list(gallery_config.FAVORITES_FILE, hearts)


def load_stars() -> dict[str, Any]:
    """Load starred prompt-match images list."""
    return _load_image_list(gallery_config.STARS_FILE)


def save_stars(stars: dict[str, Any]) -> None:
    """Save starred prompt-match images list."""
    _save_image_list(gallery_config.STARS_FILE, stars)


@app.route("/")
def index():
    """Render main gallery view."""
    return render_template("index.html")


@app.route("/favorites")
def favorites_view():
    """Render favorites gallery view."""
    return render_template("favorites.html")


@app.route("/stars")
def stars_view():
    """Render starred prompt matches view."""
    return render_template("stars.html")


@app.route("/api/config")
def api_config():
    """Provide frontend configuration, including runs directory info."""
    runs_dir = _get_runs_dir()
    is_default = runs_dir == gallery_config.DEFAULT_RUNS_DIR
    return jsonify({
        "runs_dir": gallery_config.get_runs_dir_relative_to_static(),
        "runs_dir_name": runs_dir.name,
        "runs_dir_absolute": str(runs_dir.absolute()),
        "runs_dir_input": gallery_config.get_saved_runs_dir_name() or str(gallery_config.DEFAULT_RUNS_DIR),
        "is_default": is_default,
        "default_runs_dir": str(gallery_config.DEFAULT_RUNS_DIR),
    })


@app.route("/api/config", methods=["POST"])
def api_config_update():
    """Update active runs directory via JSON payload."""
    body = request.get_json(force=True, silent=True) or {}
    runs_dir_value = body.get("runs_dir", "")
    reset = body.get("reset", False)

    if reset:
        runs_dir_value = ""

    try:
        resolved = gallery_config.set_runs_dir_name(str(runs_dir_value))
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    _clear_gallery_caches()
    is_default = resolved == gallery_config.DEFAULT_RUNS_DIR
    return jsonify({
        "runs_dir": gallery_config.get_runs_dir_relative_to_static(),
        "runs_dir_name": resolved.name,
        "runs_dir_absolute": str(resolved.absolute()),
        "runs_dir_input": gallery_config.get_saved_runs_dir_name() or str(gallery_config.DEFAULT_RUNS_DIR),
        "is_default": is_default,
        "default_runs_dir": str(gallery_config.DEFAULT_RUNS_DIR),
    })


@app.route("/api/subdirs")
def api_subdirs():
    """List subdirectories within the active runs directory."""
    return jsonify({"subdirs": get_subdirectories()})


@app.route("/api/gallery")
def api_gallery():
    """Return filtered list of image records for gallery grid."""
    scene = request.args.get("scene")
    target_prompt = request.args.get("target_prompt")
    loss_model = request.args.get("loss_model")
    reference_image = request.args.get("reference_image")
    subdir = request.args.get("subdir")
    include_first_param = request.args.get("include_first", "0")
    include_first = str(include_first_param).lower() in {"1", "true", "yes", "on"}
    if not scene:
        return jsonify({"error": "Missing scene"}), 400
    images = collect_images(
        scene,
        target_prompt if target_prompt else None,
        loss_model if loss_model else None,
        reference_image if reference_image else None,
        subdir,
    )
    if not include_first:
        images = [img for img in images if img.get("iteration", -1) >= 2]
    return jsonify({"count": len(images), "images": images})


@app.route("/api/scenes")
def api_scenes():
    """Return list of all scene names."""
    subdir = request.args.get("subdir")
    return jsonify({"scenes": gather_scenes(subdir)})


@app.route("/api/prompts")
def api_prompts():
    """Return list of text prompts for a scene."""
    scene = request.args.get("scene")
    subdir = request.args.get("subdir")
    if not scene:
        return jsonify({"error": "Missing scene"}), 400
    prompts = gather_prompts_for_scene(scene, subdir)
    return jsonify({"scene": scene, "prompts": ["All"] + prompts})


@app.route("/api/loss_models")
def api_loss_models():
    """Return list of loss model architectures for a scene."""
    scene = request.args.get("scene")
    subdir = request.args.get("subdir")
    if not scene:
        return jsonify({"error": "Missing scene"}), 400
    loss_models = gather_loss_models_for_scene(scene, subdir)
    return jsonify({"scene": scene, "loss_models": ["All"] + loss_models})


@app.route("/api/reference_images")
def api_reference_images():
    """List unique reference images for a scene."""
    scene = request.args.get("scene")
    subdir = request.args.get("subdir")
    if not scene:
        return jsonify({"error": "Missing scene"}), 400
    ref_images = gather_reference_images_for_scene(scene, subdir)
    return jsonify({"scene": scene, "reference_images": ["All"] + ref_images})


@app.route("/api/metadata")
def api_metadata():
    """Return tooltip and full metadata for a specific image."""
    runs_dir = _get_runs_dir()
    image_id = request.args.get("id")
    if not image_id or "/" not in image_id:
        return jsonify({"error": "Invalid id"}), 400

    parts = image_id.rsplit("/", 1)
    if len(parts) != 2:
        return jsonify({"error": "Invalid id format"}), 400

    dir_path, fname = parts
    run_dir = runs_dir / dir_path

    if not run_dir.exists():
        return jsonify({"error": f"Run not found: {dir_path}"}), 404

    settings = load_json(get_settings_path(run_dir))
    resources = load_json(run_dir / "resources_summary.json")
    data = {
        "settings": settings,
        "resources": resources,
    }
    tooltip = {
        "run": run_dir.name,
        "iteration": _parse_iteration(fname),
        "opt_index": _parse_opt_index(fname),
        "loss": _parse_loss(fname),
        "clip_model": settings.get("clip_model", {}).get("name"),
        "loss_model": _extract_loss_model_from_settings(settings),
        "criterion": settings.get("criterion", {}).get("type"),
        "target_prompt": settings.get("prompts", {}).get("clip_target_text_prompt") or settings.get("prompts", {}).get("clip_text_prompt") or settings.get("prompts", {}).get("target_text"),
        "initial_prompt": settings.get("prompts", {}).get("clip_initial_text_prompt"),
        "reference_image_path": settings.get("prompts", {}).get("reference_image_path"),
    }
    return jsonify({"id": image_id, "tooltip": tooltip, "full": data})


def _safe_resolve_reference_image(path_str: str) -> Path:
    """Resolve and validate a reference image path inside repo root."""
    if not path_str:
        raise FileNotFoundError("Missing reference image path")
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = (gallery_config.REPO_ROOT / candidate)
    resolved = candidate.resolve(strict=True)

    repo_root = gallery_config.REPO_ROOT.resolve(strict=True)
    try:
        resolved.relative_to(repo_root)
    except Exception as e:
        raise PermissionError("Reference image is outside repo root") from e

    if not resolved.is_file():
        raise FileNotFoundError("Reference image not found")
    return resolved


@app.route("/api/reference_image")
def api_reference_image():
    """Serve a reference image (optionally as a 256px thumbnail)."""
    path_str = request.args.get("path")
    thumb_param = request.args.get("thumb", "0")
    thumb = str(thumb_param).lower() in {"1", "true", "yes", "on"}

    try:
        img_path = _safe_resolve_reference_image(path_str)
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception:
        return jsonify({"error": "Invalid reference image path"}), 400

    if not thumb:
        return send_file(img_path)

    try:
        from PIL import Image
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im.thumbnail((256, 256))
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=85, optimize=True)
            buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    except Exception:
        return send_file(img_path)


def _handle_toggle_list(list_path: Path):
    """Handle adding/removing items from a persisted list."""
    data = _load_image_list(list_path)
    if request.method == "GET":
        return jsonify(data)

    body = request.get_json(force=True, silent=True) or {}
    img_id = body.get("id")
    if not img_id:
        return jsonify({"error": "Missing id"}), 400

    images: list[str] = data.setdefault("images", [])
    changed = False

    if request.method == "POST":
        if img_id not in images:
            images.append(img_id)
            changed = True
    elif request.method == "DELETE":
        if img_id in images:
            images.remove(img_id)
            changed = True

    if changed:
        _save_image_list(list_path, data)
    return jsonify(data)


@app.route("/api/hearts", methods=["GET", "POST", "DELETE"])
def api_hearts():
    """Get or modify favorited image IDs."""
    return _handle_toggle_list(gallery_config.FAVORITES_FILE)


@app.route("/api/stars", methods=["GET", "POST", "DELETE"])
def api_stars():
    """Get or modify starred prompt match image IDs."""
    return _handle_toggle_list(gallery_config.STARS_FILE)


@app.route("/image/<path:subpath>")
def serve_image(subpath: str):
    """Serve a run image file by relative path from active runs directory."""
    runs_dir = _get_runs_dir()
    img_path = runs_dir / subpath
    if not img_path.exists() or img_path.suffix.lower() not in gallery_config.IMAGE_EXTENSIONS:
        abort(404)
    return send_file(img_path)


def create_app() -> Flask:
    """Application factory returning configured Flask instance."""
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

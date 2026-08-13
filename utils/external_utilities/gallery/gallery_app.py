from __future__ import annotations
import json
import os
import re
import sys
import threading
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Set
from flask import Flask, jsonify, send_file, request, render_template, abort

# Ensure local gallery directory can be imported reliably
_gallery_dir = Path(__file__).resolve().parent
if str(_gallery_dir) not in sys.path:
    sys.path.insert(0, str(_gallery_dir))

import gallery_config

app = Flask(__name__, template_folder="templates", static_folder="static")
_lock = threading.Lock()

# Simple caches (invalidated manually if needed)
_scenes_cache: Set[str] | None = None
_prompts_cache: Dict[str, Set[str]] = {}
_loss_models_cache: Dict[str, Set[str]] = {}
_reference_images_cache: Dict[str, Set[str]] = {}


def _clear_gallery_caches() -> None:
    global _scenes_cache, _prompts_cache, _loss_models_cache, _reference_images_cache
    _scenes_cache = None
    _prompts_cache = {}
    _loss_models_cache = {}
    _reference_images_cache = {}


def _get_runs_dir() -> Path:
    return gallery_config.get_runs_dir()


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return {}

def get_settings_path(run_dir: Path) -> Path:
    return run_dir / 'settings.json'


def get_all_runs(subdir: str | None = None) -> List[Path]:
    """Get all run directories, optionally filtered by subdirectory."""
    runs_dir = _get_runs_dir()
    if runs_dir is None or not runs_dir.exists():
        return []
    
    base_path = runs_dir / subdir if subdir else runs_dir
    if not base_path.exists():
        return []
    
    return [p for p in base_path.iterdir() if p.is_dir()]


def get_subdirectories() -> List[str]:
    """Get list of subdirectories within RUNS_DIR."""
    runs_dir = _get_runs_dir()
    if runs_dir is None or not runs_dir.exists():
        return []
    return sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])


def extract_scene_from_run_dir(run_dir: Path) -> str:
    # Directory naming convention: <scene>_<criterion>_<model>_<timestamp>
    name = run_dir.name
    parts = name.split('_')
    return parts[0]


def gather_scenes(subdir: str | None = None) -> List[str]:
    """Gather unique scenes, optionally filtered by subdirectory."""
    global _scenes_cache
    # Note: We don't cache when using subdir filter
    if subdir is None and _scenes_cache is not None:
        return sorted(_scenes_cache)
    scenes: Set[str] = set()
    for run_dir in get_all_runs(subdir):
        scenes.add(extract_scene_from_run_dir(run_dir))
    if subdir is None:
        _scenes_cache = scenes
    return sorted(scenes)


def gather_prompts_for_scene(scene: str, subdir: str | None = None) -> List[str]:
    """Gather unique prompts for a scene, optionally filtered by subdirectory."""
    scene = scene.strip()
    if not scene:
        return []
    if scene in _prompts_cache:
        return sorted(_prompts_cache[scene])
    prompts: Set[str] = set()
    for run_dir in get_all_runs(subdir):
        if extract_scene_from_run_dir(run_dir) != scene:
            continue
        settings_path = get_settings_path(run_dir)
        if not settings_path.exists():
            continue
        settings = load_json(settings_path)
        prompts_dict = settings.get('prompts', {})
        tp = prompts_dict.get('clip_target_text_prompt') or prompts_dict.get('clip_text_prompt') or prompts_dict.get('target_text')
        if tp:
            prompts.add(tp)
    _prompts_cache[scene] = prompts
    return sorted(prompts)


def _extract_loss_model_from_settings(settings: Dict[str, Any]) -> str | None:
    clip_model = settings.get("clip_model") or {}
    name = clip_model.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _extract_reference_image_from_settings(settings: Dict[str, Any]) -> str | None:
    """Extract reference image filename from settings."""
    prompts = settings.get("prompts") or {}
    ref_path = prompts.get("reference_image_path")
    if isinstance(ref_path, str) and ref_path.strip():
        # Return just the filename, not full path
        import os
        return os.path.basename(ref_path.strip())
    return None


def gather_loss_models_for_scene(scene: str, subdir: str | None = None) -> List[str]:
    """Gather unique loss models for a scene, optionally filtered by subdirectory."""
    scene = scene.strip()
    if not scene:
        return []
    # Cache only when not using subdir filter
    if subdir is None and scene in _loss_models_cache:
        return sorted(_loss_models_cache[scene])
    loss_models: Set[str] = set()
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


def gather_reference_images_for_scene(scene: str, subdir: str | None = None) -> List[str]:
    """Gather unique reference image filenames for a scene, optionally filtered by subdirectory."""
    scene = scene.strip()
    if not scene:
        return []
    # Cache only when not using subdir filter
    if subdir is None and scene in _reference_images_cache:
        return sorted(_reference_images_cache[scene])
    ref_images: Set[str] = set()
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
) -> List[Dict[str, Any]]:
    """Collect images for a scene, optionally filtered by prompt, loss model, reference image, and subdirectory."""
    images: List[Dict[str, Any]] = []
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
        # Skip prompt filtering if target_prompt is None or empty
        if target_prompt and prompt != target_prompt:
            continue

        run_loss_model = _extract_loss_model_from_settings(settings)
        if loss_model and run_loss_model != loss_model:
            continue

        # Filter by reference image if specified
        run_ref_image = _extract_reference_image_from_settings(settings)
        if reference_image and run_ref_image != reference_image:
            continue

        run_images: List[Dict[str, Any]] = []
        for img_path in sorted(run_dir.glob("image_iter*_loss_*.png")):
            # Create path relative to RUNS_DIR for the /image/ route
            relative_path = str(img_path.relative_to(runs_dir))
            iteration = _parse_iteration(img_path.name)
            opt_index = _parse_opt_index(img_path.name)
            run_images.append({
                "id": relative_path,  # Use full relative path as ID for consistency
                "run": run_dir.name,
                "filename": img_path.name,
                "path": relative_path,  # Full relative path from RUNS_DIR
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
    # image_iter0250_opt0_loss_0.9377_timestamp.png
    try:
        part = filename.split('iter')[1]
        return int(part.split('_')[0])
    except Exception:
        return -1


def _parse_loss(filename: str) -> float:
    try:
        after = filename.split('loss_')[1]
        return float(after.split('_')[0])
    except Exception:
        return float('nan')


def _parse_opt_index(filename: str) -> int:
    try:
        match = re.search(r"_opt(\d+)_", filename)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return -1


def _load_image_list(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            with path.open('r') as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get('images', []), list):
                    return data
        except Exception:
            pass
    return {"images": []}


def _save_image_list(path: Path, data: Dict[str, Any]):
    with _lock:
        with path.open('w') as f:
            json.dump(data, f, indent=2)


def load_hearts() -> Dict[str, Any]:
    # favorites.json now stores the "hearts" list
    return _load_image_list(gallery_config.FAVORITES_FILE)


def save_hearts(hearts: Dict[str, Any]):
    _save_image_list(gallery_config.FAVORITES_FILE, hearts)


def load_stars() -> Dict[str, Any]:
    return _load_image_list(gallery_config.STARS_FILE)


def save_stars(stars: Dict[str, Any]):
    _save_image_list(gallery_config.STARS_FILE, stars)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favorites')
def favorites_view():
    return render_template('favorites.html')


@app.route('/stars')
def stars_view():
    return render_template('stars.html')


@app.route('/api/config')
def api_config():
    """Provide frontend configuration, including the runs directory path."""
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


@app.route('/api/config', methods=['POST'])
def api_config_update():
    body = request.get_json(force=True, silent=True) or {}
    runs_dir_value = body.get('runs_dir', '')
    reset = body.get('reset', False)

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


@app.route('/api/subdirs')
def api_subdirs():
    """List subdirectories within the runs directory."""
    return jsonify({"subdirs": get_subdirectories()})


@app.route('/api/gallery')
def api_gallery():
    scene = request.args.get('scene')
    target_prompt = request.args.get('target_prompt')
    loss_model = request.args.get('loss_model')
    reference_image = request.args.get('reference_image')
    subdir = request.args.get('subdir')
    include_first_param = request.args.get('include_first', '0')
    include_first = str(include_first_param).lower() in {'1','true','yes','on'}
    if not scene:
        return jsonify({"error": "Missing scene"}), 400
    runs_dir = _get_runs_dir()
    # Empty target_prompt means "all prompts"
    # Empty loss_model means "all loss models"
    # Empty reference_image means "all reference images"
    images = collect_images(
        scene,
        target_prompt if target_prompt else None,
        loss_model if loss_model else None,
        reference_image if reference_image else None,
        subdir,
    )
    # Optionally filter out first-iteration images (iteration == 1) by default
    if not include_first:
        images = [img for img in images if img.get('iteration', -1) >= 2]
    return jsonify({"count": len(images), "images": images})


@app.route('/api/scenes')
def api_scenes():
    subdir = request.args.get('subdir')
    return jsonify({"scenes": gather_scenes(subdir)})


@app.route('/api/prompts')
def api_prompts():
    scene = request.args.get('scene')
    subdir = request.args.get('subdir')
    if not scene:
        return jsonify({"error": "Missing scene"}), 400
    prompts = gather_prompts_for_scene(scene, subdir)
    # Add "All" as the first option
    return jsonify({"scene": scene, "prompts": ["All"] + prompts})


@app.route('/api/loss_models')
def api_loss_models():
    scene = request.args.get('scene')
    subdir = request.args.get('subdir')
    if not scene:
        return jsonify({"error": "Missing scene"}), 400
    loss_models = gather_loss_models_for_scene(scene, subdir)
    return jsonify({"scene": scene, "loss_models": ["All"] + loss_models})


@app.route('/api/reference_images')
def api_reference_images():
    """List unique reference images for a scene."""
    scene = request.args.get('scene')
    subdir = request.args.get('subdir')
    if not scene:
        return jsonify({"error": "Missing scene"}), 400
    ref_images = gather_reference_images_for_scene(scene, subdir)
    return jsonify({"scene": scene, "reference_images": ["All"] + ref_images})


@app.route('/api/metadata')
def api_metadata():
    runs_dir = _get_runs_dir()
    image_id = request.args.get('id')  # format: run_dir/filename or subdir/run_dir/filename
    if not image_id or '/' not in image_id:
        return jsonify({"error": "Invalid id"}), 400
    
    # Split to get filename (last part) and directory path (everything before)
    parts = image_id.rsplit('/', 1)
    if len(parts) != 2:
        return jsonify({"error": "Invalid id format"}), 400
    
    dir_path, fname = parts
    run_dir = runs_dir / dir_path
    
    if not run_dir.exists():
        return jsonify({"error": f"Run not found: {dir_path}"}), 404
    
    settings = load_json(get_settings_path(run_dir))
    resources = load_json(run_dir / 'resources_summary.json')
    data = {
        "settings": settings,
        "resources": resources,
    }
    # pick specific quick metadata fields for tooltip
    tooltip = {
        "run": run_dir.name,  # Just the run directory name
        "iteration": _parse_iteration(fname),
        "opt_index": _parse_opt_index(fname),
        "loss": _parse_loss(fname),
        "clip_model": settings.get('clip_model', {}).get('name'),
        "loss_model": _extract_loss_model_from_settings(settings),
        "criterion": settings.get('criterion', {}).get('type'),
        "target_prompt": settings.get('prompts', {}).get('clip_target_text_prompt') or settings.get('prompts', {}).get('clip_text_prompt') or settings.get('prompts', {}).get('target_text'),
        "initial_prompt": settings.get('prompts', {}).get('clip_initial_text_prompt'),
        "reference_image_path": settings.get('prompts', {}).get('reference_image_path'),
    }
    return jsonify({"id": image_id, "tooltip": tooltip, "full": data})


def _safe_resolve_reference_image(path_str: str) -> Path:
    """Resolve and validate a reference image path.

    Only allows files within the repo root (BASE_DIR.parent) to avoid arbitrary file reads.
    """
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


@app.route('/api/reference_image')
def api_reference_image():
    """Serve a reference image (optionally as a small thumbnail).

    Query params:
      - path: absolute or repo-relative path
      - thumb: 1/true to return a ~256px thumbnail
    """
    path_str = request.args.get('path')
    thumb_param = request.args.get('thumb', '0')
    thumb = str(thumb_param).lower() in {'1', 'true', 'yes', 'on'}

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
            im = im.convert('RGB')
            im.thumbnail((256, 256))
            buf = BytesIO()
            im.save(buf, format='JPEG', quality=85, optimize=True)
            buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')
    except Exception:
        # Fallback to original if thumbnail generation fails
        return send_file(img_path)


def _handle_toggle_list(list_path: Path):
    data = _load_image_list(list_path)
    if request.method == 'GET':
        return jsonify(data)

    body = request.get_json(force=True, silent=True) or {}
    img_id = body.get('id')
    if not img_id:
        return jsonify({"error": "Missing id"}), 400

    images: List[str] = data.setdefault('images', [])
    changed = False

    if request.method == 'POST':
        if img_id not in images:
            images.append(img_id)
            changed = True
    elif request.method == 'DELETE':
        if img_id in images:
            images.remove(img_id)
            changed = True

    if changed:
        _save_image_list(list_path, data)
    return jsonify(data)


@app.route('/api/hearts', methods=['GET', 'POST', 'DELETE'])
def api_hearts():
    return _handle_toggle_list(gallery_config.FAVORITES_FILE)


@app.route('/api/stars', methods=['GET', 'POST', 'DELETE'])
def api_stars():
    return _handle_toggle_list(gallery_config.STARS_FILE)


@app.route('/image/<path:subpath>')
def serve_image(subpath):
    runs_dir = _get_runs_dir()
    img_path = runs_dir / subpath
    if not img_path.exists() or img_path.suffix.lower() not in gallery_config.IMAGE_EXTENSIONS:
        abort(404)
    return send_file(img_path)


def create_app():
    return app


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

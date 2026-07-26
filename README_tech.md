# OLAT CLIP Optimization Tests

This repository contains code for optimizing OLAT (One Light At a Time) lighting using CLIP models.

## Repository Structure

### `/prompt_experiments/`
Contains scripts for running prompt-based optimization experiments:
- `prompt_experiment_batch_run.py` - Main script for running CLIP optimization experiments
- `cache_models.py` - Script to cache CLIP models locally
- `submit_prompt_experiment.sh` - SLURM batch submission script
- `submit_prompt_experiment_array_mode.sh` - SLURM array job submission script
- Output/error logs from SLURM jobs (`.out`, `.err` files)

### `/gallery/`
Web-based gallery application for browsing and curating optimization results:
- `gallery_app.py` - Flask application for viewing optimization runs
- `collect_favorites.py` - Script to collect favorite images
- `favorites.json` - Storage for favorited images
- `README_GALLERY.md` - Detailed documentation for the gallery app
- `templates/` - HTML templates for the web interface
- `static/` - CSS and JavaScript files for the gallery

### Root Directory
- `examples/example_scenes.py` - Scene definitions for different 3D models and asset loading helpers
- `requirements.txt` - Python dependencies
- `utils/` - Utility modules (losses, training, color conversion, etc.)
- `blender_utils/` - Blender rendering utilities
- `optimization_runs/` - Output directory for optimization results
- `examples/example_olats/` - Example OLAT assets
- Example assets are downloaded automatically from Hugging Face if they are missing locally.

## Running the Gallery

```bash
cd gallery
python gallery_app.py
```

Then open http://localhost:5000 in your browser.

## Running Experiments

To submit a prompt optimization experiment to SLURM:

```bash
cd prompt_experiments
sbatch submit_prompt_experiment_array_mode.sh
```

See `/gallery/README_GALLERY.md` for more details on the gallery features.


## Scene Directory Structure

The framework supports two types of scene data inputs:

### 1. Per-Light EXR Directory (`OLATDirScene`)
Per-light scenes must contain an `optimizable_lights/` subdirectory with individual `.exr` passes for each light to be optimized. Any non-optimizable background or base light pass (e.g., `base_lighting.exr`) must be placed directly in the root scene directory, **not** inside `optimizable_lights/`.

```text
my_scene/
├── base_lighting.exr       (optional non-optimized light pass)
├── alpha.exr               (optional alpha mask)
└── optimizable_lights/     (directory containing ONLY optimizable light passes)
    ├── light_pass_001.exr
    ├── light_pass_002.exr
    └── ...
```

### 2. Multi-Layer EXR File (`MultiLayerEXRScene`)
Multi-layer EXR scenes store all light passes within a single `.exr` file, where per-light channels are grouped by a designated light keyword (default `'LGT'`).

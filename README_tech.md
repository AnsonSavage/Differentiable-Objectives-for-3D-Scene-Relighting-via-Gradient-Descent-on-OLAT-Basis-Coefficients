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
- `scenes.py` - Scene definitions for different 3D models
- `requirements.txt` - Python dependencies
- `utils/` - Utility modules (losses, training, color conversion, etc.)
- `blender_utils/` - Blender rendering utilities
- `optimization_runs/` - Output directory for optimization results
- `render/` - Rendered images

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

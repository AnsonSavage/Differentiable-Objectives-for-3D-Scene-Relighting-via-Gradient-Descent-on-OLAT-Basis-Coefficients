# OLAT CLIP Optimization Tests

This repository contains code for optimizing OLAT (One Light At a Time) lighting using CLIP models.

## Repository Structure

### Root Directory
- `examples/` - Example notebooks demoing the use of the optimization framework with various losses
- `losses/` - Loss function objectives (CLIP, SSIM, LPIPS, Aesthetic, etc.)
- `utils/` - Core utility modules (training, color conversion, record keeping, etc.) and `external_utilities/`
- `optimization_runs/` - Output directory for optimization results
- `config.py` - Global project configuration settings


### `/utils/external_utilities/`
Standalone tools and helper applications for rendering assets and viewing results:
- **`gallery/`**: Web-based Flask gallery application for browsing, filtering, and curating optimization run images (`gallery_app.py`, `collect_favorites.py`, `templates/`, `static/`). See [`utils/external_utilities/gallery/README.md`](utils/external_utilities/gallery/README.md) for detailed documentation on how to run it.
- **`blender_utils/`**: Blender rendering utilities and add-ons (`olat_render_addon.py`) for generating OLAT datasets from 3D scenes.


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

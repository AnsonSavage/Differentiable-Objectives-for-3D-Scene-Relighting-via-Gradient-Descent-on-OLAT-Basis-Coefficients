# Differentiable Objectives for 3D Scene Relighting via Gradient Descent on OLAT Basis Coefficients
This repository contains the source code for the 2026 Eurographics Short Paper, _Differentiable Objectives for 3D Scene Relighting via Gradient Descent on OLAT Basis Coefficients_.

* Read the short paper [here](https://diglib.eg.org/handle/10.2312/egs20261021).
* Read the entire thesis (which contains additional results and experiments) [here](https://arks.lib.byu.edu/ark:/34234/q2f2b88884).

## Repository Structure
- `examples/` - Example notebooks demoing the use of the optimization framework with various losses
- `losses/` - Loss function objectives (CLIP, SSIM, LPIPS, Aesthetic, etc.)
- `utils/` - Core utility modules (optimization, color conversion, record keeping, etc.) 
  - `external_utilities/` - Standalone tools and helper applications
    - `gallery/`: Web-based Flask gallery application for browsing, filtering, and curating optimization run images (`gallery_app.py`, `collect_favorites.py`, `templates/`, `static/`). See [`utils/external_utilities/gallery/README.md`](utils/external_utilities/gallery/README.md) for detailed documentation on how to run it.
    - `blender_utils/`: Blender rendering utilities and add-ons (`olat_render_addon.py`) for generating OLAT datasets from 3D scenes.
- `config.py` - Global project configuration settings (including the local directories where optimization results are stored and where model weights are downloaded)


## Scene Directory Structure

The framework supports OLATs stored in two structures:

### 1. Per-Light EXR Directory (used by `OLATDirScene`)
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

### 2. Multi-Layer EXR File (used by `MultiLayerEXRScene`)
Multi-layer EXR scenes store all light passes within a single `.exr` file, where per-light channels are grouped by a designated light keyword found in a substring of the channel name (default `'LGT'`).

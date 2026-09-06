# Blender OLAT Render Add-on

A Blender add-on (`olat_render_addon.py`) designed to automate rendering One-Light-At-A-Time (OLAT) images in OpenEXR format, structured for for loading into this project via `utils.scene.OLATDirScene`.

---

## Features

- **Automatic Light Detection**: Detects lamp objects (`LIGHT`), emissive mesh geometry (`MESH` with emission nodes/principled shaders), and world environment background lighting. Separates mixed emissive materials from meshes into dedicated light objects automatically.
- **Dome Light Generator**: Procedurally generates a hemisphere dome array of directional emission quad lights (configurable subdivision levels) centered at the origin for standard light-stage setups.
- **Optimizable vs. Static Partitioning**:
  - **Optimizable Lights**: Rendered individually into standalone linear OpenEXR passes (`olat_<light_name>.exr`).
  - **Static Lights**: Grouped and pre-rendered into a combined linear OpenEXR background pass (`base_lighting.exr`).
- **Metadata Generation**: Automatically outputs `olat_metadata.json` mapping output filenames back to their original Blender object names.

---

## Installation in Blender

1. In Blender, navigate to **Edit** > **Preferences** > **Add-ons**.
2. Click **Install...** at the top right and select `olat_render_addon.py` (or install via the **Extensions** menu in Blender 4.2+).
3. Enable the checkbox for **Render: OLAT Render Tools**.
4. The controls will appear in the **Properties Panel** > **Render Properties** tab under the **OLAT Render** panel.

---

## Usage Workflow

1. **Setup Lights**:
   - Click **Detect Lights** to automatically discover lights in your scene, OR
   - Click **Create Dome Lights** to generate a light-stage dome array over your scene.
2. **Configure Pass Types**:
   - In the panel hierarchy list, toggle:
     - **Enabled** (checkbox): Whether the light will be rendered.
     - **Optimizable** (toggle): If active, the light is rendered as an individual OLAT pass. If inactive, the light is merged into the static background pass.
3. **Set Output Directory**:
   - Specify the destination path in **Output Directory**.
4. **Render**:
   - Click **Render OLAT**. The add-on will switch the render format to OpenEXR and sequentially render each pass.
5. **Re-apply Learned Multipliers in Blender**:
   - In your optimization run directory, use `final_multipliers_rgb.json` (written by `optimize_with_criterion`).
   - In Blender's **OLAT Render** panel set:
     - **Metadata JSON** to the rendered scene's `olat_metadata.json`
     - **Multipliers File** to `final_multipliers_rgb.json` (or `.pt` if PyTorch is available in Blender Python)
     - **Result Index** for multi-result optimization runs (`n_results > 1`)
   - Click **Apply Learned Multipliers** to multiply the learned RGB values onto mapped light colors (including emissive mesh lights and world background nodes).

---

## Generated Directory Structure

Rendering produces the exact folder structure expected by `OLATDirScene`:

```text
<output_directory>/
├── optimizable_lights/
│   ├── olat_DomeLight_000.exr
│   ├── olat_DomeLight_001.exr
│   └── ...
├── base_lighting.exr          # Generated if static/non-optimizable lights are present
└── olat_metadata.json         # Mapping of filenames to Blender light objects
```

---

## Loading into the Relighting Pipeline

Load the rendered directory in Python using `utils.scene.OLATDirScene`:

```python
from utils.scene import OLATDirScene

scene = OLATDirScene(
    name="my_blender_scene",
    description="Synthetic OLAT dataset rendered with Blender",
    path_to_olat_dir="path/to/output_directory",
    name_of_non_optimized_lights_file="base_lighting.exr",  # Set to None if no static lights were rendered
    include_alpha_mask=False,                               # Set to True if an alpha.exr is provided
    device="cuda",
)

# Inspect the loaded scene
scene.display_scene()
```

# Differentiable Objectives for 3D Scene Relighting via Gradient Descent on OLAT Basis Coefficients
![Readme Teaser Image](readme_images/figure_1.png)

This repository contains the source code for the 2026 Eurographics Short Paper, _Differentiable Objectives for 3D Scene Relighting via Gradient Descent on OLAT Basis Coefficients_, which describes a method for relighting 3D scenes after rendering by using objectives such as CLIP similarity or NST and adjusting lights to match those objectives.

* Read the short paper [here](https://diglib.eg.org/handle/10.2312/egs20261021).
* Read the entire thesis (which contains additional results and experiments) [here](https://arks.lib.byu.edu/ark:/34234/q2f2b88884).
* Watch the demo video <a href="https://www.youtube.com/watch?v=px7gxgCySMQ" target="_blank" rel="noopener noreferrer">on YouTube</a>:

<div style="text-align: center;">
  <a href="https://www.youtube.com/watch?v=px7gxgCySMQ" target="_blank" rel="noopener noreferrer">
    <img src="readme_images/video_thumbnail.jpg" alt="Watch the Demo Video" width="600">
  </a>
</div>

## Getting Started
This code is tested to be compatible with **Python 3.12** and **Python 3.13** and the modules in `requirements.txt` (note: `git` must be installed for the `CLIP` dependency). Install the required dependencies by running:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> **Windows users:**
> - **CUDA support:** On Windows, `pip install -r requirements.txt` installs the CPU-only build of `torch` from PyPI. If you have an NVIDIA GPU, install CUDA-enabled `torch==2.8.0` and `torchvision==0.23.0` inside your active virtual environment first (available for CUDA 12.6, 12.8, and 12.9 — e.g., `pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128`, or see [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)) before running `pip install -r requirements.txt`.
> - **Long Paths (`MAX_PATH`):** `pip install` and saving results can hit Windows' 260-character path limit (`OSError: [Errno 2]` or `WinError 206`) due to the long repo name. To fix this, enable Windows Long Paths by opening **PowerShell as Administrator** (Start Menu → search `PowerShell` → **Run as administrator**), running the command below, and then restarting your terminal/IDE ([more info](https://pip.pypa.io/warnings/enable-long-paths)):
>   ```powershell
>   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
>   ```

Check out the example notebooks in the `examples/` directory to get started. 

## Repository Structure
- `examples/` - Example notebooks demoing the use of the optimization framework with various losses
- `losses/` - Loss function objectives (CLIP, SSIM, LPIPS, Aesthetic, etc.)
- `utils/` - Core utility modules (optimization, color conversion, record keeping, etc.) 
  - `external_utilities/` - Standalone tools and helper applications
    - `gallery/`: Web-based Flask gallery application for browsing, filtering, and curating optimization run images (`gallery_app.py`, `templates/`, `static/`). See [`utils/external_utilities/gallery/README.md`](utils/external_utilities/gallery/README.md) for detailed documentation on how to run it.
    - `blender_utils/`: Blender rendering utilities and add-ons (`olat_render_addon.py`) for generating OLAT datasets from 3D scenes.
- `config.py` - Global project configuration settings (including the local directories where optimization results are stored and where model weights are downloaded)
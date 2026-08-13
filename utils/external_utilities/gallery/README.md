# Optimization Run Gallery

A lightweight Flask web app to browse optimization run images filtered by scene and target prompt, inspect metadata on hover, and curate favorites.

## Features

- **Filter by Scene & Prompt**: View images from optimization runs filtered by scene and target prompt
- **Hover Tooltips**: Quick metadata preview (model, criterion, prompts, loss, iteration)
- **Grouped View**: Optional matrix view with iterations as rows and batch items as columns
- **Modal View**: Click to enlarge images with full metadata and copyable filesystem paths
- **Hearts**: Heart/unheart images to curate a "pretty" collection (persisted in `favorites.json`)
- **Stars**: Star/unstar images to curate a "matches prompt/image well" collection (persisted in `stars.json`)
- **Zero Configuration Default**: Automatically defaults to `OPTIMIZATION_RUNS/` (the default output folder of the optimization notebooks)
- **Easy Directory Switching**: Switch to any custom runs directory directly in the UI or reset to default at any time
- **Clean Architecture**: Shared JavaScript utilities, centralized configuration, no code duplication

## Quick Start

```bash
# Run from repository root:
python utils/external_utilities/gallery/gallery_app.py

# Or run from the gallery folder:
cd utils/external_utilities/gallery && python gallery_app.py
```

Then open: **http://localhost:5000** (or http://127.0.0.1:5000)

## Usage

1. **Select Base Directory**: Choose a subdirectory inside `OPTIMIZATION_RUNS/` (e.g. `text_clip_cosine_similarity_example/`)
2. **Select Scene**: Dropdown auto-populates from run folder names
3. **Select Target Prompt**: Dropdown fills with unique prompts for that scene
4. **Load Images**: Click "Load" to display the gallery
5. **Browse**: Hover for quick info, click to enlarge
6. **Heart/Star**: Use ♥/♡ and ★/☆ to save into either list
7. **View Lists**: Navigate to `/favorites` (Hearts) or `/stars` (Stars)

## Configuration & Runs Directory

- **Default Directory**: Out of the box, the app points directly to `OPTIMIZATION_RUNS/` at the root of the repository.
- **Changing Directory**: Click **"Change runs directory"** in the top navigation bar to enter an absolute path or repo-relative path.
- **Resetting to Default**: Click **"Reset to Default"** in the directory modal to revert to `OPTIMIZATION_RUNS/`.
- **Persistence**: Any custom directory selection is saved in `gallery_settings.json` and reused across launches.

## Architecture

```
gallery/
├── gallery_config.py         # Central configuration (paths, defaults, settings)
├── gallery_app.py           # Flask backend (API endpoints, image serving)
├── static/
│   ├── shared.js           # Shared utilities (used by all pages)
│   ├── gallery.js          # Main gallery page logic
│   ├── favorites.js        # Hearts / Favorites page logic
│   ├── stars.js            # Stars page logic
│   └── styles.css          # Dark-theme styles
├── templates/
│   ├── index.html          # Main gallery page
│   ├── favorites.html      # Hearts page
│   └── stars.html          # Stars page
├── favorites.json          # Persisted hearts list
└── stars.json              # Persisted stars list
```

## API Endpoints

- `GET /` - Main gallery page
- `GET /favorites` - Hearts page
- `GET /stars` - Stars page
- `GET /api/config` - Configuration (runs directory path, default status)
- `POST /api/config` - Update or reset runs directory
- `GET /api/subdirs` - List subdirectories within runs directory
- `GET /api/scenes` - List unique scenes
- `GET /api/prompts?scene=<name>` - Prompts for a scene
- `GET /api/loss_models?scene=<name>` - Loss models for a scene
- `GET /api/reference_images?scene=<name>` - Reference images for a scene
- `GET /api/gallery?scene=<name>&target_prompt=<text>` - Filtered images
- `GET /api/metadata?id=<path>` - Image metadata
- `GET /api/hearts` - Get hearts list
- `POST /api/hearts` - Add to hearts
- `DELETE /api/hearts` - Remove from hearts
- `GET /api/stars` - Get stars list
- `POST /api/stars` - Add to stars
- `DELETE /api/stars` - Remove from stars
- `GET /image/<path>` - Serve image file
- `GET /api/reference_image?path=<path>` - Serve reference image / thumbnail

## File Formats

### Run Directory Structure

The gallery expects run directories with this naming convention:

```
<scene>_<criterion>_<model>_<timestamp>/
  ├── settings.json          # Run settings and prompt info
  ├── image_iter0001_opt0_loss_0.123_timestamp.png
  ├── image_iter0050_opt0_loss_0.098_timestamp.png
  └── ...
```

### favorites.json / stars.json

```json
{
  "images": [
    "subdir_name/run_dir_name/image_filename.png"
  ]
}
```

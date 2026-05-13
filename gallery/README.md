# Optimization Run Gallery

A lightweight Flask web app to browse optimization run images filtered by scene and target prompt, inspect metadata on hover, and curate favorites.

## Features

- **Filter by Scene & Prompt**: View images from optimization runs filtered by scene and target prompt
- **Hover Tooltips**: Quick metadata preview (model, criterion, prompts, loss, iteration)
- **Grouped View**: Optional matrix view with iterations as rows and batch items as columns
- **Modal View**: Click to enlarge images with full metadata and copyable filesystem paths
- **Hearts**: Heart/unheart images to curate a “pretty” collection (persisted in `favorites.json`)
- **Stars**: Star/unstar images to curate a “matches prompt/image well” collection (persisted in `stars.json`)
- **Configurable**: Point to different optimization runs directories via environment variable
- **Clean Architecture**: Shared JavaScript utilities, centralized configuration, no code duplication

## Quick Start

```bash
# Install dependencies (Flask required)
pip install -r requirements.txt

# Run the gallery (requires GALLERY_RUNS_DIR)
python gallery_app.py

# Or specify a custom runs directory
GALLERY_RUNS_DIR="" python gallery_app.py  # TODO: PATH_UPDATE gallery runs directory
```

Then open: **http://localhost:5000**

## Usage

1. **Select Scene**: Dropdown auto-populates from run folder names
2. **Select Target Prompt**: Dropdown fills with unique prompts for that scene
3. **Load Images**: Click "Load" to display the gallery
4. **Browse**: Hover for quick info, click to enlarge
5. **Heart/Star**: Use ♥/♡ and ★/☆ to save into either list
6. **View Lists**: Navigate to `/favorites` (Hearts) or `/stars` (Stars)

## Configuration

### Environment Variable (Recommended)

Change the runs directory without modifying code:

```bash
# Relative path (from parent of gallery folder)
export GALLERY_RUNS_DIR="my_custom_runs"
python gallery_app.py

# Absolute path
export GALLERY_RUNS_DIR=""  # TODO: PATH_UPDATE gallery runs directory
python gallery_app.py

# Inline
GALLERY_RUNS_DIR="alternative_runs" python gallery_app.py
```

### Direct Config Edit

Edit `config.py` to set the default:

```python
RUNS_DIR_NAME = os.environ.get("GALLERY_RUNS_DIR", "")  # TODO: PATH_UPDATE gallery runs directory
```

### How It Works

- All paths centralized in `config.py`
- Backend serves config via `/api/config` endpoint
- Frontend dynamically loads paths from backend
- Works with both relative and absolute paths

## Architecture

```
gallery/
├── config.py                 # Central configuration (paths, constants)
├── gallery_app.py           # Flask backend (API endpoints)
├── collect_favorites.py     # Script to copy favorites to a folder
├── static/
│   ├── shared.js           # Shared utilities (used by both pages)
│   ├── gallery.js          # Main gallery page logic
│   ├── favorites.js        # Favorites page logic
│   └── styles.css
├── templates/
│   ├── index.html          # Main gallery page
│   └── favorites.html      # Favorites page
└── favorites.json          # Persisted favorites list
└── stars.json              # Persisted stars list
```

### Data Flow

```
Backend (Python)
  ├─> config.py defines RUNS_DIR
  ├─> gallery_app.py reads from RUNS_DIR
  └─> /api/config sends path to frontend

Frontend (JavaScript)
  ├─> shared.js loads config from /api/config
  ├─> gallery.js/favorites.js use config for image paths
  └─> Images served via /image/<relative_path>
```

## API Endpoints

- `GET /` - Main gallery page
- `GET /favorites` - Hearts page
- `GET /stars` - Stars page
- `GET /api/config` - Configuration (runs directory path)
- `GET /api/scenes` - List of unique scenes
- `GET /api/prompts?scene=<name>` - Prompts for a scene
- `GET /api/gallery?scene=<name>&target_prompt=<text>` - Filtered images
- `GET /api/metadata?id=<path>` - Image metadata
- `GET /api/hearts` - Get hearts list
- `POST /api/hearts` - Add to hearts
- `DELETE /api/hearts` - Remove from hearts
- `GET /api/stars` - Get stars list
- `POST /api/stars` - Add to stars
- `DELETE /api/stars` - Remove from stars
- `GET /api/favorites` - Alias for hearts (backwards compat)
- `POST /api/favorites` - Alias for hearts
- `DELETE /api/favorites` - Alias for hearts
- `GET /image/<path>` - Serve image file

## File Formats

### Run Directory Structure

The gallery expects run directories with this naming convention:

```
<scene>_<criterion>_<model>_<timestamp>/
  ├── settings.json          # Must contain prompts.clip_target_text_prompt
  ├── image_iter0001_loss_0.123.png
  ├── image_iter0050_loss_0.098.png
  └── ...
```

Previewed image filenames usually follow this pattern:

```
image_iter####_opt#_loss_#.####_timestamp.png
```

Where `iter####` is the optimization iteration, `opt#` is the batch / optimization slot, and `loss` is the recorded score for that image.

### favorites.json

```json
{
  "images": [
    "run_dir_name/image_filename.png",
    "another_run/another_image.png"
  ]
}

### stars.json

Same format as `favorites.json`, but stores starred images:

```json
{
  "images": [
    "run_dir_name/image_filename.png"
  ]
}
```
```

## Extending

Ideas for future enhancements:

- **Pagination**: Handle thousands of images efficiently
- **Fuzzy Search**: Search/filter within prompts
- **Sorting**: By loss, iteration, timestamp
- **Bulk Actions**: Export/download multiple favorites
- **Metadata Panel**: Dedicated panel for full JSON settings
- **Cache Invalidation**: Live refresh when new runs appear

## Technical Details

### Image Path Resolution

1. Backend creates relative paths from `RUNS_DIR`: `"scene_model/image.png"`
2. Frontend adds `/image/` prefix: `"/image/scene_model/image.png"`
3. Flask route prepends `RUNS_DIR`: `RUNS_DIR / "scene_model/image.png"`

### Code Organization

- **Shared utilities** (`shared.js`): Functions used by both gallery and favorites pages
- **No duplication**: Modal handlers, card generation, config loading all centralized
- **Type safety**: Flask serves typed JSON responses
- **Security**: Path traversal protection in image serving

## Troubleshooting

**Images not loading?**
- Check `RUNS_DIR` path in config
- Verify run directories contain `settings.json` and image files
- Check browser console for errors

**Scenes dropdown empty?**
- Ensure run directories follow naming convention: `<scene>_<criterion>_<model>_<timestamp>`
- Check that runs directory exists and is readable

**Favorites not persisting?**
- Ensure `favorites.json` has write permissions
- Check browser console for API errors

## Safety

- Only serves files from configured `RUNS_DIR`
- No arbitrary path traversal allowed
- Intended for local/trusted network use (no authentication)

## Requirements

- Python 3.7+
- Flask
- Modern web browser with JavaScript enabled

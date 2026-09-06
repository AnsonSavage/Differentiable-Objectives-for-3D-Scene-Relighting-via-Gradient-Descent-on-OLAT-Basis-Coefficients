"""Helpers for accessing bundled example target reference images."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOCAL_EXAMPLE_REFERENCE_IMAGES_DIR = BASE_DIR / "EXAMPLE_REFERENCE_IMAGES"


def get_example_reference_image_path(filename: str = "golden_hour_01.jpg") -> str:
    """Get absolute path to an example reference image.

    Args:
        filename: Name of the reference image file in EXAMPLE_REFERENCE_IMAGES.

    Returns:
        String path to the reference image file.

    Raises:
        FileNotFoundError: If the reference image file does not exist.
    """
    image_path = LOCAL_EXAMPLE_REFERENCE_IMAGES_DIR / filename
    if not image_path.exists():
        raise FileNotFoundError(
            f"Example reference image '{filename}' not found at: {image_path}"
        )
    return str(image_path)

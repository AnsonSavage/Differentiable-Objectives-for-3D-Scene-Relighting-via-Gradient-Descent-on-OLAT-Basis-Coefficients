import torch


def resize_then_crop(image, target_height, target_width):
    """
    Resize an image without stretching it

    Args:
        image: A tensor of shape (B, C, H, W).
        target_height: The desired height of the output image.
        target_width: The desired width of the output image.

    Returns:
        A tensor of shape (B, C, target_height, target_width).
    """

    _, _, h, w = image.shape
    scale = max(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Handle numeric precision issues to ensure we don't end up smaller than target
    new_h = max(new_h, target_height)
    new_w = max(new_w, target_width)

    # Scale the image
    resized = torch.nn.functional.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # Crop the image to the target size
    crop_top = (new_h - target_height) // 2
    crop_left = (new_w - target_width) // 2
    return resized[:, :, crop_top:crop_top + target_height, crop_left:crop_left + target_width]
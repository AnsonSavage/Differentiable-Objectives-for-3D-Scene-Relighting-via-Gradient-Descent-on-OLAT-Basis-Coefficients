"""Image and tensor plotting / visualization helper functions."""
import os
import time

import matplotlib.pyplot as plt


def display_numpy_array(numpy_array, title: str | None = None, ax: plt.Axes | None = None, show: bool = True):
    """Display an [H, W, C] NumPy array using matplotlib.

    Args:
        numpy_array: Array of shape [H, W, C].
        title: Optional title string.
        ax: Optional matplotlib Axes to draw into. If None, draws in active pyplot figure.
        show: Whether to call plt.show().
    """
    if ax is None:
        plt.imshow(numpy_array)
        if title:
            plt.title(title)
        plt.axis("off")
        if show:
            plt.show()
    else:
        ax.imshow(numpy_array)
        if title:
            ax.set_title(title)
        ax.axis("off")


def display_tensor(tensor, title: str | None = None, ax: plt.Axes | None = None, show: bool = True):
    """Convert an image tensor [C, H, W] or [1, C, H, W] to NumPy and display it.

    Args:
        tensor: Image tensor of shape [C, H, W] or [1, C, H, W].
        title: Optional title string.
        ax: Optional matplotlib Axes.
        show: Whether to call plt.show().
    """
    numpy_array = tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    display_numpy_array(numpy_array, title=title, ax=ax, show=show)


def display_image_batch_grid(
    image_batch,
    title: str | None = None,
    max_cols: int = 4,
    show: bool = True,
    save_path: str | None = None,
    save_each: bool = False,
    is_last: bool = False,
    save_index: int | None = None,
):
    """Display a batch of images [N, C, H, W] in a subplot grid.

    Args:
        image_batch: Tensor of shape [N, C, H, W].
        title: Optional figure title.
        max_cols: Maximum number of columns in the grid.
        show: Whether to call plt.show().
        save_path: Optional file path or directory path to save the grid.
        save_each: If True, saves every call; if False, saves only when is_last is True.
        is_last: Set True on the final iteration to trigger saving if save_path is given.
        save_index: Optional iteration index used in saved filename.

    Returns:
        Tuple of (fig, axes) containing the matplotlib Figure and flat list of Axes.
    """
    n_results = int(image_batch.shape[0])
    cols = int(min(max_cols, n_results)) if n_results > 0 else 1
    rows = int((n_results + cols - 1) // cols) if n_results > 0 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    for i in range(rows * cols):
        ax = axes[i]
        if i < n_results:
            display_tensor(
                image_batch[i],
                title=f"opt {i}",
                ax=ax,
                show=False,
            )
        else:
            ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None and (save_each or is_last):
        try:
            if save_path.endswith(os.sep) or os.path.isdir(save_path):
                os.makedirs(save_path, exist_ok=True)
                stamp = int(time.time())
                fname = f"display_grid_{stamp}" # TODO: This should be cleaned up. We should explicitly state when we want the name to be display_grid, rather than just passing a folder
                if save_index is not None:
                    fname = f"display_grid_{save_index}_{stamp}"
                out_path = os.path.join(save_path, fname + ".png")
            else:
                root, ext = os.path.splitext(save_path)
                if save_each:
                    if save_index is not None:
                        out_path = f"{root}_{save_index}{ext or '.png'}"
                    else:
                        out_path = f"{root}_{int(time.time())}{ext or '.png'}"
                else:
                    out_path = save_path

            fig.savefig(out_path, bbox_inches="tight")
        except (OSError, ValueError) as e:
            print(f"WARNING: failed to save display grid to {save_path}: {e}")

    if show:
        plt.show()
    return fig, axes
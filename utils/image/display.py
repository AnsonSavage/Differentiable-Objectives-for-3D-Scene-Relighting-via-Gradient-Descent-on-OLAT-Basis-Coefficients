import os
import time

import matplotlib.pyplot as plt


def display_numpy_array(numpy_array, title=None, ax=None, show=True):
    """Display a HxWxC numpy array. If an Axes is provided, draw there; otherwise use plt.

    Args:
        numpy_array: HxWxC numpy array
        title: optional title string
        ax: optional matplotlib Axes to draw into
        show: whether to call plt.show() (caller should manage when drawing into subplots)
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

def display_tensor(tensor, title=None, ax=None, show=True):
    """Takes a tensor, converts it to a numpy array [H, W, C] and displays it.

    Accepts an Axes instance to allow drawing into subplot grids and a `show` flag.
    """
    numpy_array = tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    display_numpy_array(numpy_array, title=title, ax=ax, show=show)


def display_image_batch_grid(
    image_batch,
    title=None,
    max_cols=4,
    show=True,
    save_path=None,
    save_each=False,
    is_last=False,
    save_index=None,
):
    """Display a batch of images [N, C, H, W] in a subplot grid.

    Args:
        image_batch: Tensor of shape [N, C, H, W]
        title: Optional figure title
        max_cols: Max number of columns in the grid
        show: Whether to call plt.show()

    Additional args for saving:
        save_path: If provided, either a directory path or a file path where the
            grid image should be saved. If a directory is given (or path ends
            with os.sep), a file will be generated inside it using a timestamp.
        save_each: If True, save on every call. If False, only save when
            `is_last` is True.
        is_last: Set True when this call corresponds to the final iteration
            and you want to save the final grid when `save_path` is provided.
        save_index: Optional integer used to make filenames unique (e.g. the
            current save iteration or optimization iteration).

    Returns:
        (fig, axes): Matplotlib figure and flattened axes list
    """
    n_results = int(image_batch.shape[0])
    cols = int(min(max_cols, n_results)) if n_results > 0 else 1
    rows = int((n_results + cols - 1) // cols) if n_results > 0 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    # Flatten 2D numpy array of axes into a simple 1D list/array
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
    # Save the figure if requested
    if save_path is not None and (save_each or is_last):
        # If save_path is directory-like, ensure it exists and generate filename
        try:
            if save_path.endswith(os.sep) or os.path.isdir(save_path):
                os.makedirs(save_path, exist_ok=True)
                stamp = int(time.time())
                fname = f"display_grid_{stamp}"
                if save_index is not None:
                    fname = f"display_grid_{save_index}_{stamp}"
                out_path = os.path.join(save_path, fname + ".png")
            else:
                # save_path looks like a file. If saving repeatedly, insert a
                # timestamp or index to avoid silent overwrites.
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
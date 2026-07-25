"""
Seeding utility for reproducible PyTorch experiments.

Usage:
    from utils.seed import set_global_seed
    resolved_seed = set_global_seed(seed)
"""
from __future__ import annotations

import torch


def set_global_seed(seed: int | None = None) -> int:
    """Set global PyTorch RNG seed (CPU and CUDA) and return the resolved seed.

    Args:
        seed: Optional seed integer. If None, a seed is generated via PyTorch.

    Returns:
        int: The resolved seed integer applied to PyTorch.
    """
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


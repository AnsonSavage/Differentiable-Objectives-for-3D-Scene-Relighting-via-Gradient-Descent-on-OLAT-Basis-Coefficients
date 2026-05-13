"""
Seeding utilities for reproducible experiments.

Usage:
    from utils.seed import set_global_seed
    resolved_seed = set_global_seed(seed)

This will seed Python's random, NumPy, and PyTorch (CPU and CUDA if available).
It returns the resolved integer seed (either the provided one or a randomly chosen one).
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: Optional[int] = None) -> int:
    """Set global RNG seeds across libraries and return the resolved seed.

    Args:
        seed: Optional seed to use. If None, a cryptographically secure random seed
              will be chosen in the 32-bit range.

    Returns:
        int: The seed that was applied to all RNGs.
    """
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)

    # Python's random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU and CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed

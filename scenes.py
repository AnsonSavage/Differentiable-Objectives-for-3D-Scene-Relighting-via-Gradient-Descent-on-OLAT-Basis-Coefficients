"""Compatibility shim for the relocated example scene definitions.

The canonical implementation now lives in [examples/example_scenes.py](examples/example_scenes.py).
This module remains so existing notebooks and scripts that import `scenes`
continue to work.
"""

from examples.example_scenes import *  # noqa: F401,F403
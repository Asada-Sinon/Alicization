"""Underworld: a GPU-native 2D artificial-life sandbox.

Public surface for building and running a world.
"""

from .config import Config
from .metrics import Metrics
from .state import WorldState, init_state
from .step import build_step, make_scan, new_world

__all__ = [
    "Config",
    "Metrics",
    "WorldState",
    "init_state",
    "build_step",
    "make_scan",
    "new_world",
]

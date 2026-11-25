"""Scalar helpers for the per-frame simulation: angle wrapping and a clamp without NumPy dispatch."""

from __future__ import annotations

import math

import numpy as np


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) in radians to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _fast_clip(val: float, low: float, high: float) -> float:
    """Fast inline scalar clamp avoiding NumPy dispatch overhead."""
    if val < low:
        return low
    if val > high:
        return high
    return val

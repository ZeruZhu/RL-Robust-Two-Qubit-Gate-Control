"""Reward functions."""

from __future__ import annotations

import numpy as np


def log_infidelity_reward(fidelity: float, eps: float = 1e-12) -> float:
    clipped = min(max(float(fidelity), 0.0), 1.0)
    return float(-np.log10(max(eps, 1.0 - clipped)))

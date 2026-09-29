"""
Evaluation Metrics Module for MetaBench.

Calculates:
- Robustness Index
- Success Rate (within tolerance epsilon)
- Convergence Speed (evaluations to milestone threshold)
- Efficiency Trade-off Score
"""

from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd

from metabench.trajectory_tracker import TrajectoryTracker


def calculate_robustness_index(values: Union[List[float], np.ndarray], eps: float = 1e-6) -> float:
    """
    Robustness Index: Measures algorithm stability across random seeds.
    R = 1 / (1 + CV), where CV = std / (mean + eps)
    Bounds: [0, 1], where 1.0 represents perfect determinism and consistency across trials.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 1.0
    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)
    cv = std_val / (abs(mean_val) + eps)
    return float(1.0 / (1.0 + cv))


def calculate_success_rate(errors: Union[List[float], np.ndarray], tolerance: float = 1e-3) -> float:
    """
    Calculates proportion of independent trials that achieved error <= tolerance.
    Returns float in range [0.0, 1.0].
    """
    arr = np.asarray(errors, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    successful = np.sum(arr <= tolerance)
    return float(successful / len(arr))


def calculate_convergence_speed(
    curve: np.ndarray,
    threshold_fraction: float = 0.95,
    max_evals: int = 10000
) -> int:
    """
    Calculates number of function evaluations required to reach 95% of total improvement.
    curve: (K, 2) array of [evaluations, best_fitness].
    """
    if len(curve) < 2:
        return max_evals

    evals = curve[:, 0]
    fits = curve[:, 1]
    f_start = fits[0]
    f_final = fits[-1]
    total_drop = f_start - f_final

    if total_drop <= 0:
        return max_evals

    target_val = f_start - threshold_fraction * total_drop
    reached_indices = np.where(fits <= target_val)[0]
    if len(reached_indices) > 0:
        return int(evals[reached_indices[0]])
    return max_evals


def calculate_composite_score(
    mean_error: float,
    robustness: float,
    speed_evals: int,
    max_evals: int = 10000
) -> float:
    """
    Composite efficiency score balancing accuracy, robustness, and speed:
    Higher score indicates superior overall performance [0, 100].
    """
    norm_acc = 1.0 / (1.0 + np.log1p(max(0.0, mean_error)))
    norm_speed = 1.0 - (speed_evals / max_evals)
    score = (0.50 * norm_acc + 0.30 * robustness + 0.20 * norm_speed) * 100.0
    return float(np.clip(score, 0.0, 100.0))

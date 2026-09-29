"""
Statistical Metrics Module for MetaBench.

Computes non-parametric and descriptive statistics across independent trials:
- Mean
- Variance
- Best (Minimum)
- Worst (Maximum)
- Median & Interquartile Range (IQR)
- Wilcoxon Signed-Rank Test & Friedman Rank Test
"""

from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy import stats


def compute_trial_statistics(values: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    Computes the 4 requested non-parametric/descriptive statistical metrics:
    - mean
    - variance
    - best
    - worst
    Along with median, standard deviation, and IQR.
    """
    arr = np.asarray(values, dtype=np.float64)
    # Filter out NaNs if any
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "mean": float("nan"),
            "variance": float("nan"),
            "std": float("nan"),
            "best": float("nan"),
            "worst": float("nan"),
            "median": float("nan"),
            "iqr": float("nan"),
            "count": 0
        }

    mean_val = float(np.mean(arr))
    var_val = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    best_val = float(np.min(arr))
    worst_val = float(np.max(arr))
    med_val = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr_val = float(q75 - q25)

    return {
        "mean": mean_val,
        "variance": var_val,
        "std": std_val,
        "best": best_val,
        "worst": worst_val,
        "median": med_val,
        "iqr": iqr_val,
        "count": len(arr)
    }


def wilcoxon_signed_rank_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Performs two-sided Wilcoxon signed-rank test between two algorithms' trial results.
    Returns statistic, p_value, and significance conclusion.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diff = a - b

    # If completely identical
    if np.all(diff == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "winner": "Tie"
        }

    # Filter out zero differences for Wilcoxon
    non_zero = diff[diff != 0]
    if len(non_zero) < 5:
        # Too few non-zero differences for asymptotic approximation
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "winner": "Tie"
        }

    res = stats.wilcoxon(a, b, zero_method="pratt")
    p_val = float(res.pvalue)
    is_sig = p_val < alpha

    winner = "Tie"
    if is_sig:
        if np.median(a) < np.median(b):
            winner = "A"
        else:
            winner = "B"

    return {
        "statistic": float(res.statistic),
        "p_value": p_val,
        "significant": is_sig,
        "winner": winner
    }


def friedman_ranking(results_df: pd.DataFrame, algo_col: str = "algorithm", bench_col: str = "benchmark", score_col: str = "mean_fitness") -> pd.DataFrame:
    """
    Computes average Friedman rank for each algorithm across all benchmarks.
    Rank 1 is the best performing algorithm.
    """
    pivot = results_df.pivot(index=bench_col, columns=algo_col, values=score_col)
    # Rank algorithms per benchmark (row-wise), lowest score gets rank 1
    ranks = pivot.rank(axis=1, ascending=True)
    avg_ranks = ranks.mean(axis=0).reset_index()
    avg_ranks.columns = ["algorithm", "average_rank"]
    return avg_ranks.sort_values(by="average_rank").reset_index(drop=True)

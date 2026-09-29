"""Unit tests for Statistical and Performance Metrics."""

import pytest
import numpy as np
import pandas as pd
from metabench.statistical_metrics import (
    compute_trial_statistics,
    wilcoxon_signed_rank_test,
    friedman_ranking
)
from metabench.metrics import (
    calculate_robustness_index,
    calculate_success_rate,
    calculate_convergence_speed,
    calculate_composite_score
)


def test_compute_trial_statistics():
    sample = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = compute_trial_statistics(sample)
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["best"] == pytest.approx(1.0)
    assert stats["worst"] == pytest.approx(5.0)
    assert stats["variance"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(3.0)
    assert stats["iqr"] == pytest.approx(2.0)


def test_wilcoxon_signed_rank_test():
    a = np.array([1.0, 2.0, 1.5, 1.2, 0.8, 1.1, 0.9, 1.3])
    b = np.array([10.0, 12.0, 11.5, 9.2, 8.8, 10.1, 10.9, 11.3])
    res = wilcoxon_signed_rank_test(a, b)
    assert res["significant"] is True
    assert res["winner"] == "A"


def test_friedman_ranking():
    df = pd.DataFrame([
        {"algorithm": "DE", "benchmark": "Sphere", "mean_fitness": 0.001},
        {"algorithm": "PSO", "benchmark": "Sphere", "mean_fitness": 0.05},
        {"algorithm": "GA", "benchmark": "Sphere", "mean_fitness": 0.20},
        {"algorithm": "DE", "benchmark": "Rastrigin", "mean_fitness": 0.01},
        {"algorithm": "PSO", "benchmark": "Rastrigin", "mean_fitness": 1.5},
        {"algorithm": "GA", "benchmark": "Rastrigin", "mean_fitness": 3.0},
    ])
    ranks = friedman_ranking(df, score_col="mean_fitness")
    assert ranks.iloc[0]["algorithm"] == "DE"
    assert ranks.iloc[0]["average_rank"] == 1.0


def test_robustness_and_success_rate():
    errors = np.array([0.0001, 0.0005, 0.0002, 0.05, 0.10])
    succ = calculate_success_rate(errors, tolerance=1e-3)
    assert succ == pytest.approx(0.60)

    rob = calculate_robustness_index(errors)
    assert 0.0 <= rob <= 1.0

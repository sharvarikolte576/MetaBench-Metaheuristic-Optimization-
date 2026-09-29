"""Unit tests for Core Framework, Telemetry, and Convergence Analysis."""

import pytest
import numpy as np
from metabench.benchmarks import create_benchmark
from metabench.framework import OptimizationProblem
from metabench.trajectory_tracker import TrajectoryTracker, TelemetryRecord
from metabench.convergence_analysis import ConvergenceAnalyzer
from metabench.hyperparameter_config import get_preset_configs, get_sensitivity_sweep_grid


def test_trajectory_tracker_and_diversity():
    tracker = TrajectoryTracker(record_interval=1)
    
    # Population of 4 points
    pop = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 2.0],
        [2.0, 2.0]
    ])
    div = tracker.compute_diversity(pop)
    assert div > 0.0

    tracker.record_step(
        iteration=0,
        evaluations=4,
        population=pop,
        fitnesses=np.array([10.0, 8.0, 6.0, 4.0]),
        best_fitness=4.0,
        best_position=np.array([2.0, 2.0])
    )

    assert len(tracker.history) == 1
    df = tracker.to_dataframe()
    assert len(df) == 1
    assert "population_diversity" in df.columns


def test_convergence_analyzer():
    tracker = TrajectoryTracker()
    # Mock progress
    for i in range(10):
        tracker.history.append(TelemetryRecord(
            iteration=i,
            evaluations=(i + 1) * 100,
            best_fitness=100.0 / (i + 1),
            mean_fitness=150.0 / (i + 1),
            std_fitness=10.0,
            population_diversity=5.0 / (i + 1)
        ))

    analysis = ConvergenceAnalyzer.analyze(tracker, max_evals=1000)
    assert "mean_velocity" in analysis
    assert "stagnation_ratio" in analysis
    assert "convergence_auc" in analysis
    assert analysis["diversity_loss_rate"] > 0.0


def test_hyperparameter_presets_and_sweep():
    for algo in ["PSO", "GA", "DE", "SA"]:
        presets = get_preset_configs(algo)
        assert "standard" in presets
        assert "high_exploration" in presets
        assert "high_exploitation" in presets

        grid = get_sensitivity_sweep_grid(algo)
        assert len(grid) >= 3

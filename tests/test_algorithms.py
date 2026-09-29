"""Unit tests for Metaheuristic Algorithms in MetaBench."""

import pytest
import numpy as np
from metabench.benchmarks import create_benchmark
from metabench.framework import OptimizationProblem
from metabench.algorithms import (
    create_algorithm,
    ParticleSwarmOptimization,
    GeneticAlgorithm,
    DifferentialEvolution,
    SimulatedAnnealing
)
from metabench.trajectory_tracker import TrajectoryTracker


@pytest.mark.parametrize("algo_name", ["PSO", "GA", "DE", "SA"])
def test_algorithm_budget_compliance_and_bounds(algo_name):
    bench = create_benchmark("Sphere", dim=5)
    max_budget = 1000
    problem = OptimizationProblem(bench, dimension=5, max_evaluations=max_budget)
    tracker = TrajectoryTracker(record_interval=2)

    algo = create_algorithm(algo_name)
    result = algo.optimize(problem, seed=42, tracker=tracker)

    # Strict budget constraint: must NEVER exceed max_evaluations
    assert result.total_evaluations <= max_budget
    assert problem.evaluations_count <= max_budget

    # Valid fitness and position
    assert np.isfinite(result.best_fitness)
    assert len(result.best_position) == 5

    # Bounds preservation: must not violate search space
    low, high = bench.get_bounds(5)
    assert np.all(result.best_position >= low - 1e-9)
    assert np.all(result.best_position <= high + 1e-9)

    # Telemetry should be populated
    assert len(tracker.history) > 0


@pytest.mark.parametrize("algo_name", ["PSO", "GA", "DE", "SA"])
def test_algorithm_optimization_progress(algo_name):
    """Ensure algorithm achieves substantial fitness improvement on a simple convex problem."""
    bench = create_benchmark("Sphere", dim=3)
    problem = OptimizationProblem(bench, dimension=3, max_evaluations=2000)

    algo = create_algorithm(algo_name)
    result = algo.optimize(problem, seed=123)

    # The Sphere function has optimum 0.0. After 2000 evaluations, error should be low
    assert result.error_to_optimum < 5.0

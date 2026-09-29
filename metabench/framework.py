"""
Core Framework Module for MetaBench.

Defines Problem abstraction with evaluation budget enforcement,
BaseMetaheuristic abstract interface, and standardized OptimizationResult.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Callable
import time
import numpy as np

from metabench.benchmarks import BenchmarkFunction
from metabench.trajectory_tracker import TrajectoryTracker


class BudgetExceededException(Exception):
    """Raised when an algorithm attempts to exceed the strict evaluation budget."""
    pass


class OptimizationProblem:
    """
    Encapsulates the objective function, dimensional bounds,
    and strict evaluation counting to enforce an unyielding budget.
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        dimension: Optional[int] = None,
        max_evaluations: int = 10000
    ):
        self.benchmark = benchmark
        if not benchmark.is_scalable:
            self.dimension = benchmark.dimension
        else:
            self.dimension = dimension if dimension is not None else benchmark.dimension
        self.max_evaluations = max_evaluations
        self.evaluations_count: int = 0
        self.lower_bounds, self.upper_bounds = benchmark.get_bounds(self.dimension)
        self.bound_ranges = self.upper_bounds - self.lower_bounds

    def reset(self):
        """Reset evaluation counter."""
        self.evaluations_count = 0

    @property
    def remaining_evaluations(self) -> int:
        return max(0, self.max_evaluations - self.evaluations_count)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate single vector or population batch with strict budget enforcement.
        If a batch size exceeds the remaining budget, only the remaining slice is evaluated.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        is_1d = (x_arr.ndim == 1)
        if is_1d:
            x_arr = x_arr.reshape(1, -1)

        batch_size = x_arr.shape[0]
        if self.evaluations_count >= self.max_evaluations:
            # Budget fully exhausted
            if is_1d:
                return float("inf")
            return np.full(batch_size, float("inf"))

        actual_evals = min(batch_size, self.max_evaluations - self.evaluations_count)
        eval_slice = x_arr[:actual_evals]

        # Clamp/Project to boundary to maintain valid search space domain
        clipped_slice = np.clip(eval_slice, self.lower_bounds, self.upper_bounds)
        vals = self.benchmark.evaluate(clipped_slice)
        if np.ndim(vals) == 0:
            vals = np.array([vals])

        self.evaluations_count += actual_evals

        if actual_evals < batch_size:
            # Pad truncated evaluations with infinity
            padded = np.full(batch_size, float("inf"))
            padded[:actual_evals] = vals
            vals = padded

        if is_1d:
            return float(vals[0])
        return vals

    def clamp(self, x: np.ndarray) -> np.ndarray:
        """Vectorized clamping within search space bounds."""
        return np.clip(x, self.lower_bounds, self.upper_bounds)


@dataclass
class OptimizationResult:
    """Standardized output container for a single optimization trial."""
    algorithm_name: str
    benchmark_name: str
    dimension: int
    seed: int
    best_position: np.ndarray
    best_fitness: float
    error_to_optimum: float
    total_evaluations: int
    iterations: int
    execution_time_sec: float
    hyperparameters: Dict[str, Any]
    telemetry: TrajectoryTracker

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert scalar metrics to flat dictionary."""
        return {
            "algorithm": self.algorithm_name,
            "benchmark": self.benchmark_name,
            "dimension": self.dimension,
            "seed": self.seed,
            "best_fitness": self.best_fitness,
            "error_to_optimum": self.error_to_optimum,
            "total_evaluations": self.total_evaluations,
            "iterations": self.iterations,
            "execution_time_sec": self.execution_time_sec,
        }


class BaseMetaheuristic(ABC):
    """Abstract Base Class for all Metaheuristic Optimization Algorithms."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def optimize(
        self,
        problem: OptimizationProblem,
        seed: Optional[int] = None,
        tracker: Optional[TrajectoryTracker] = None
    ) -> OptimizationResult:
        """Run optimization trial adhering strictly to problem.max_evaluations."""
        pass

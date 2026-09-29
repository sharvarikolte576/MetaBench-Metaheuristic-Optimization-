"""
Trajectory Tracker Module for MetaBench.

Tracks multi-dimensional trajectory telemetry, fitness evolution,
stagnation checkpoints, and population diversity dynamics over optimization iterations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class TelemetryRecord:
    """A single snapshot of optimization state at iteration t."""
    iteration: int
    evaluations: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    population_diversity: float  # Mean distance to population centroid
    best_position: Optional[np.ndarray] = None
    step_size: float = 0.0


class TrajectoryTracker:
    """Records, aggregates, and exports multi-dimensional telemetry data."""

    def __init__(self, record_interval: int = 1, record_positions: bool = True):
        self.record_interval = max(1, record_interval)
        self.record_positions = record_positions
        self.history: List[TelemetryRecord] = []
        self._last_best_fitness: float = float("inf")
        self.evaluations_counter: int = 0

    def compute_diversity(self, population: np.ndarray) -> float:
        """
        Compute population dispersion / diversity metric:
        mean Euclidean distance from each individual to the population centroid.
        """
        if population is None or len(population) == 0:
            return 0.0
        pop_arr = np.atleast_2d(population)
        if len(pop_arr) <= 1:
            return 0.0
        centroid = np.mean(pop_arr, axis=0)
        distances = np.linalg.norm(pop_arr - centroid, axis=1)
        return float(np.mean(distances))

    def record_step(
        self,
        iteration: int,
        evaluations: int,
        population: np.ndarray,
        fitnesses: np.ndarray,
        best_fitness: float,
        best_position: np.ndarray,
        step_size: float = 0.0
    ):
        """Record telemetry snapshot at the current optimization step."""
        self.evaluations_counter = evaluations
        
        # Only record at designated intervals, or when a new global best is found, or first/last
        should_record = (
            iteration % self.record_interval == 0
            or best_fitness < self._last_best_fitness
            or iteration == 0
        )
        if not should_record:
            return

        fits = np.asarray(fitnesses, dtype=np.float64)
        finite_fits = fits[np.isfinite(fits)]
        mean_fit = float(np.mean(finite_fits)) if finite_fits.size > 0 else float(best_fitness)
        std_fit = float(np.std(finite_fits)) if finite_fits.size > 0 else 0.0
        diversity = self.compute_diversity(population)

        pos_copy = best_position.copy() if (self.record_positions and best_position is not None) else None

        record = TelemetryRecord(
            iteration=iteration,
            evaluations=evaluations,
            best_fitness=float(best_fitness),
            mean_fitness=mean_fit,
            std_fitness=std_fit,
            population_diversity=diversity,
            best_position=pos_copy,
            step_size=float(step_size)
        )
        self.history.append(record)
        self._last_best_fitness = best_fitness

    def to_dataframe(self) -> pd.DataFrame:
        """Convert recorded telemetry history into a pandas DataFrame."""
        rows = []
        for r in self.history:
            rows.append({
                "iteration": r.iteration,
                "evaluations": r.evaluations,
                "best_fitness": r.best_fitness,
                "mean_fitness": r.mean_fitness,
                "std_fitness": r.std_fitness,
                "population_diversity": r.population_diversity,
                "step_size": r.step_size
            })
        return pd.DataFrame(rows)

    def get_convergence_curve(self) -> np.ndarray:
        """Returns array of shape (K, 2): column 0 is evaluations, column 1 is best_fitness."""
        if not self.history:
            return np.empty((0, 2))
        evals = [r.evaluations for r in self.history]
        fits = [r.best_fitness for r in self.history]
        return np.column_stack([evals, fits])

    def get_diversity_curve(self) -> np.ndarray:
        """Returns array of shape (K, 2): column 0 is evaluations, column 1 is diversity."""
        if not self.history:
            return np.empty((0, 2))
        evals = [r.evaluations for r in self.history]
        divs = [r.population_diversity for r in self.history]
        return np.column_stack([evals, divs])

"""
Parameter Sensitivity Analysis Module for MetaBench.

Conducts systematic hyperparameter sensitivity evaluations (One-At-A-Time and Morris screening),
measuring fitness variance, parameter elasticity, and isolating key mathematical drivers
of convergence behavior.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from metabench.benchmarks import BenchmarkFunction, create_benchmark
from metabench.framework import OptimizationProblem
from metabench.algorithms import create_algorithm
from metabench.hyperparameter_config import get_default_config, get_sensitivity_sweep_grid


class ParameterSensitivityAnalyzer:
    """Evaluates the sensitivity of algorithm performance to hyperparameter variations."""

    def __init__(self, budget: int = 5000, seeds_per_config: int = 5):
        self.budget = budget
        self.seeds_per_config = seeds_per_config

    def run_oat_sensitivity(
        self,
        algorithm_name: str,
        benchmark: BenchmarkFunction,
        dimension: int = 10
    ) -> pd.DataFrame:
        """
        Run One-At-A-Time (OAT) parameter variation sweep.
        Varies one hyperparameter across its grid while holding others at default.
        """
        base_config = get_default_config(algorithm_name)
        grid = get_sensitivity_sweep_grid(algorithm_name)
        results = []

        for param_name, param_values in grid.items():
            baseline_val = base_config.get(param_name)

            for val in param_values:
                cfg = dict(base_config)
                cfg[param_name] = val

                fitness_samples = []
                for seed in range(self.seeds_per_config):
                    problem = OptimizationProblem(
                        benchmark=benchmark,
                        dimension=dimension,
                        max_evaluations=self.budget
                    )
                    algo = create_algorithm(algorithm_name, cfg)
                    res = algo.optimize(problem, seed=1000 + seed)
                    fitness_samples.append(res.best_fitness)

                fits = np.array(fitness_samples)
                results.append({
                    "algorithm": algorithm_name,
                    "benchmark": benchmark.name,
                    "parameter": param_name,
                    "parameter_value": str(val),
                    "is_baseline": (val == baseline_val),
                    "mean_fitness": float(np.mean(fits)),
                    "std_fitness": float(np.std(fits)),
                    "best_fitness": float(np.min(fits)),
                    "worst_fitness": float(np.max(fits)),
                    "samples": len(fits)
                })

        return pd.DataFrame(results)

    @staticmethod
    def calculate_parameter_importance(oat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates relative parameter importance based on between-group variance:
        Var(mean_fitness) across parameter settings.
        """
        records = []
        for param, group in oat_df.groupby("parameter"):
            mean_vals = group["mean_fitness"].values
            variance_across_levels = float(np.var(mean_vals))
            min_mean = float(np.min(mean_vals))
            max_mean = float(np.max(mean_vals))
            span = max_mean - min_mean

            records.append({
                "parameter": param,
                "variance": variance_across_levels,
                "dynamic_range": span,
                "optimal_value": group.loc[group["mean_fitness"].idxmin()]["parameter_value"]
            })

        df = pd.DataFrame(records)
        total_var = df["variance"].sum()
        if total_var > 0:
            df["sensitivity_index"] = df["variance"] / total_var
        else:
            df["sensitivity_index"] = 0.0

        return df.sort_values(by="sensitivity_index", ascending=False).reset_index(drop=True)

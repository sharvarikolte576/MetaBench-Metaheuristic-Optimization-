"""
Benchmark Runner Module for MetaBench.

Executes batch experimental campaigns up to 2,400 independent trials
(4 algorithms x 20 benchmarks x 30 seeds) under strict 10,000-evaluation budgets.
Supports multi-core parallel execution and checkpointing.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm

from metabench.benchmarks import create_benchmark, get_all_benchmark_names
from metabench.framework import OptimizationProblem, OptimizationResult
from metabench.algorithms import create_algorithm
from metabench.trajectory_tracker import TrajectoryTracker
from metabench.statistical_metrics import compute_trial_statistics
from metabench.metrics import calculate_robustness_index, calculate_success_rate


def _run_single_trial_task(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Top-level worker function for multiprocessing trial execution.
    args contains:
        algo_name, bench_name, dim, seed, budget, config, record_telemetry
    """
    algo_name = args["algo_name"]
    bench_name = args["bench_name"]
    dim = args["dim"]
    seed = args["seed"]
    budget = args["budget"]
    config = args.get("config", None)
    record_telemetry = args.get("record_telemetry", False)

    bench = create_benchmark(bench_name, dim=dim)
    prob = OptimizationProblem(bench, dimension=dim, max_evaluations=budget)
    algo = create_algorithm(algo_name, config=config)

    # Conditionally keep detailed telemetry to conserve memory over 2400 runs
    tracker = TrajectoryTracker(record_interval=5) if record_telemetry else None

    result = algo.optimize(prob, seed=seed, tracker=tracker)

    curve_data = None
    if tracker is not None and len(tracker.history) > 0:
        curve_data = tracker.get_convergence_curve().tolist()

    return {
        "algorithm": algo_name,
        "benchmark": bench_name,
        "dimension": prob.dimension,
        "seed": seed,
        "best_fitness": result.best_fitness,
        "error_to_optimum": result.error_to_optimum,
        "total_evaluations": result.total_evaluations,
        "iterations": result.iterations,
        "execution_time_sec": result.execution_time_sec,
        "convergence_curve": curve_data
    }


class BenchmarkRunner:
    """Orchestrates large-scale benchmarking campaigns across metaheuristics."""

    def __init__(
        self,
        algorithms: Optional[List[str]] = None,
        benchmarks: Optional[List[str]] = None,
        dimension: int = 10,
        budget: int = 10000,
        num_seeds: int = 30,
        seed_start: int = 42,
        max_workers: Optional[int] = None
    ):
        self.algorithms = algorithms or ["PSO", "GA", "DE", "SA"]
        self.benchmarks = benchmarks or get_all_benchmark_names()
        self.dimension = dimension
        self.budget = budget
        self.num_seeds = num_seeds
        self.seed_start = seed_start
        self.max_workers = max_workers or min(os.cpu_count() or 4, 8)

    @property
    def total_trials_count(self) -> int:
        return len(self.algorithms) * len(self.benchmarks) * self.num_seeds

    def run(
        self,
        record_sample_telemetry: bool = True,
        sample_telemetry_seeds: int = 3,
        show_progress: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Executes all trials in parallel.
        Returns:
            (raw_trials_df, summary_stats_df, telemetry_curves_dict)
        """
        tasks = []
        for algo in self.algorithms:
            for bench in self.benchmarks:
                for s_idx in range(self.num_seeds):
                    seed = self.seed_start + s_idx
                    # Only record full telemetry on first few seeds to keep RAM low
                    record_tel = record_sample_telemetry and (s_idx < sample_telemetry_seeds)
                    tasks.append({
                        "algo_name": algo,
                        "bench_name": bench,
                        "dim": self.dimension,
                        "seed": seed,
                        "budget": self.budget,
                        "config": None,
                        "record_telemetry": record_tel
                    })

        raw_results = []
        telemetry_curves: Dict[str, List[np.ndarray]] = {}

        # Run parallel execution using ProcessPoolExecutor
        # For Windows and safe execution with high task count, chunking or standard pool
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_run_single_trial_task, task) for task in tasks]
            
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc=f"Executing {len(tasks)} MetaBench Trials")

            for future in iterator:
                res = future.result()
                curve = res.pop("convergence_curve")
                raw_results.append(res)

                if curve is not None:
                    key = f"{res['algorithm']}_{res['benchmark']}"
                    if key not in telemetry_curves:
                        telemetry_curves[key] = []
                    telemetry_curves[key].append(np.array(curve))

        raw_df = pd.DataFrame(raw_results)

        # Aggregate the 4 statistical metrics (mean, variance, best, worst)
        summary_rows = []
        grouped = raw_df.groupby(["algorithm", "benchmark"])

        for (algo, bench), group in grouped:
            errors = group["error_to_optimum"].values
            fits = group["best_fitness"].values
            stats_err = compute_trial_statistics(errors)
            stats_fit = compute_trial_statistics(fits)
            robustness = calculate_robustness_index(errors)
            success_rate = calculate_success_rate(errors, tolerance=1e-3)
            mean_time = float(np.mean(group["execution_time_sec"]))

            summary_rows.append({
                "algorithm": algo,
                "benchmark": bench,
                "trials": len(group),
                # 4 core metrics for error to global optimum
                "mean_error": stats_err["mean"],
                "variance_error": stats_err["variance"],
                "best_error": stats_err["best"],
                "worst_error": stats_err["worst"],
                "median_error": stats_err["median"],
                "iqr_error": stats_err["iqr"],
                # 4 core metrics for raw fitness
                "mean_fitness": stats_fit["mean"],
                "variance_fitness": stats_fit["variance"],
                "best_fitness": stats_fit["best"],
                "worst_fitness": stats_fit["worst"],
                # Operational & Robustness metrics
                "robustness_index": robustness,
                "success_rate": success_rate,
                "avg_execution_time_sec": mean_time
            })

        summary_df = pd.DataFrame(summary_rows)
        return raw_df, summary_df, telemetry_curves

"""
MetaBench Experiment Runner CLI.

Usage:
    python run_experiments.py --full-benchmark --workers 4
    python run_experiments.py --quick-demo
    python run_experiments.py --sensitivity
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from metabench.benchmarks import create_benchmark, get_all_benchmark_names
from metabench.benchmark_runner import BenchmarkRunner
from metabench.results_analyzer import ResultsAnalyzer
from metabench.parameter_sensitivity import ParameterSensitivityAnalyzer
from metabench.visuals import (
    plot_convergence_curves,
    plot_error_boxplots,
    plot_diversity_dynamics,
    plot_parameter_sensitivity_heatmap,
    plot_radar_tradeoff
)
from metabench.framework import OptimizationProblem
from metabench.algorithms import create_algorithm
from metabench.trajectory_tracker import TrajectoryTracker

console = Console()


def run_quick_demo():
    """Run a fast demonstration across 4 algorithms and 4 key landscape topologies."""
    console.print(Panel.fit("[bold cyan]MetaBench – Running Quick Demo Campaign[/bold cyan]"))
    benchmarks = ["Sphere", "Rosenbrock", "Rastrigin", "Ackley"]
    runner = BenchmarkRunner(
        algorithms=["PSO", "GA", "DE", "SA"],
        benchmarks=benchmarks,
        dimension=10,
        budget=5000,
        num_seeds=5,
        max_workers=min(4, os.cpu_count() or 4)
    )

    t0 = time.time()
    raw_df, summary_df, curves = runner.run(record_sample_telemetry=True, show_progress=True)
    elapsed = time.time() - t0

    console.print(f"\n[green]Completed {len(raw_df)} trials across {len(benchmarks)} benchmarks in {elapsed:.2f} seconds![/green]\n")

    analyzer = ResultsAnalyzer(raw_df, summary_df)
    tradeoff = analyzer.generate_tradeoff_matrix()

    table = Table(title="Algorithmic Performance Trade-Off (Demo)")
    table.add_column("Algorithm", style="bold cyan")
    table.add_column("Friedman Rank", justify="right")
    table.add_column("Mean Error", justify="right")
    table.add_column("Robustness Index", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Avg Time (s)", justify="right")
    table.add_column("Composite Score", justify="right", style="bold green")

    for _, row in tradeoff.iterrows():
        table.add_row(
            row["algorithm"],
            f"{row['average_rank']:.2f}",
            f"{row['mean_error']:.4e}",
            f"{row['robustness_index']:.3f}",
            f"{row['success_rate']*100:.1f}%",
            f"{row['avg_execution_time_sec']:.3f}",
            f"{row['composite_score']:.1f}"
        )
    console.print(table)


def run_full_campaign(max_workers: int = 4, budget: int = 10000, num_seeds: int = 30):
    """
    Executes the full 2,400-trial benchmark campaign:
    4 Algorithms x 20 Benchmarks x 30 Seeds = 2,400 Trials under strict 10,000 FE budget.
    """
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    all_benchmarks = get_all_benchmark_names()
    algorithms = ["PSO", "GA", "DE", "SA"]

    total_expected = len(algorithms) * len(all_benchmarks) * num_seeds
    console.print(Panel.fit(
        f"[bold cyan]MetaBench: Executing Full {total_expected}-Trial Benchmark Campaign[/bold cyan]\n"
        f"Algorithms: {', '.join(algorithms)} | Benchmarks: {len(all_benchmarks)} | "
        f"Seeds: {num_seeds} | Evaluation Budget: {budget:,} FEs"
    ))

    runner = BenchmarkRunner(
        algorithms=algorithms,
        benchmarks=all_benchmarks,
        dimension=10,
        budget=budget,
        num_seeds=num_seeds,
        max_workers=max_workers
    )

    t0 = time.time()
    raw_df, summary_df, curves = runner.run(
        record_sample_telemetry=True,
        sample_telemetry_seeds=5,
        show_progress=True
    )
    total_time = time.time() - t0

    console.print(f"\n[bold green]Successfully completed all {len(raw_df)} trials in {total_time:.2f} seconds ({len(raw_df)/total_time:.1f} trials/sec)![/bold green]\n")

    # Save CSV tables
    raw_csv_path = os.path.join("results", "raw_trials_2400.csv")
    summary_csv_path = os.path.join("results", "benchmark_summary.csv")
    raw_df.to_csv(raw_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    console.print(f"Saved raw trial data to [yellow]{raw_csv_path}[/yellow]")
    console.print(f"Saved statistical summary table to [yellow]{summary_csv_path}[/yellow]")

    # Analyzer
    analyzer = ResultsAnalyzer(raw_df, summary_df)
    tradeoff_df = analyzer.generate_tradeoff_matrix()
    cat_df = analyzer.compute_category_summary()

    tradeoff_csv_path = os.path.join("results", "tradeoff_matrix.csv")
    cat_csv_path = os.path.join("results", "category_summary.csv")
    tradeoff_df.to_csv(tradeoff_csv_path, index=False)
    cat_df.to_csv(cat_csv_path, index=False)

    # Print summary table to console
    table = Table(title=f"MetaBench Global Performance Trade-Off Matrix ({total_expected} Trials)")
    table.add_column("Algorithm", style="bold cyan")
    table.add_column("Friedman Rank", justify="right")
    table.add_column("Mean Error", justify="right")
    table.add_column("Robustness Index", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Avg Time (s)", justify="right")
    table.add_column("Composite Score", justify="right", style="bold green")

    for _, row in tradeoff_df.iterrows():
        table.add_row(
            row["algorithm"],
            f"{row['average_rank']:.2f}",
            f"{row['mean_error']:.4e}",
            f"{row['robustness_index']:.3f}",
            f"{row['success_rate']*100:.1f}%",
            f"{row['avg_execution_time_sec']:.3f}",
            f"{row['composite_score']:.1f}"
        )
    console.print(table)

    # Generate Figures
    console.print("\n[cyan]Generating publication-quality figures in figures/...[/cyan]")
    key_benchmarks = ["Sphere", "Rosenbrock", "Rastrigin", "Ackley", "Griewank", "Schwefel"]

    conv_fig_path = os.path.join("figures", "convergence_comparison.png")
    plot_convergence_curves(curves, benchmarks=key_benchmarks, save_path=conv_fig_path)
    console.print(f"Saved convergence curves to [yellow]{conv_fig_path}[/yellow]")

    box_fig_path = os.path.join("figures", "final_error_boxplots.png")
    plot_error_boxplots(raw_df, benchmarks=["Sphere", "Rosenbrock", "Rastrigin", "Ackley"], save_path=box_fig_path)
    console.print(f"Saved boxplots to [yellow]{box_fig_path}[/yellow]")

    radar_fig_path = os.path.join("figures", "algorithmic_tradeoff_radar.png")
    plot_radar_tradeoff(tradeoff_df, save_path=radar_fig_path)
    console.print(f"Saved radar chart to [yellow]{radar_fig_path}[/yellow]")

    # Single-run diversity demonstration on Rastrigin
    rastrigin = create_benchmark("Rastrigin", dim=10)
    trackers = {}
    for algo_name in ["PSO", "GA", "DE", "SA"]:
        prob = OptimizationProblem(rastrigin, dimension=10, max_evaluations=10000)
        algo = create_algorithm(algo_name)
        tr = TrajectoryTracker(record_interval=5)
        algo.optimize(prob, seed=42, tracker=tr)
        trackers[algo_name] = tr

    div_fig_path = os.path.join("figures", "diversity_dynamics.png")
    plot_diversity_dynamics(trackers, bench_name="Rastrigin (10D)", save_path=div_fig_path)
    console.print(f"Saved diversity dynamics to [yellow]{div_fig_path}[/yellow]")

    # Parameter Sensitivity
    run_sensitivity_study()


def run_sensitivity_study():
    """Runs parameter sensitivity analysis for PSO and DE."""
    console.print("\n[cyan]Running Hyperparameter Sensitivity Analysis (OAT)...[/cyan]")
    analyzer = ParameterSensitivityAnalyzer(budget=5000, seeds_per_config=5)
    bench = create_benchmark("Rastrigin", dim=10)

    pso_oat = analyzer.run_oat_sensitivity("PSO", bench, dimension=10)
    pso_oat.to_csv(os.path.join("results", "pso_parameter_sensitivity.csv"), index=False)

    de_oat = analyzer.run_oat_sensitivity("DE", bench, dimension=10)
    de_oat.to_csv(os.path.join("results", "de_parameter_sensitivity.csv"), index=False)

    # Plot heatmaps
    pso_sens_fig = os.path.join("figures", "pso_parameter_sensitivity_heatmap.png")
    plot_parameter_sensitivity_heatmap(pso_oat, save_path=pso_sens_fig)
    console.print(f"Saved PSO sensitivity heatmap to [yellow]{pso_sens_fig}[/yellow]")

    de_sens_fig = os.path.join("figures", "de_parameter_sensitivity_heatmap.png")
    plot_parameter_sensitivity_heatmap(de_oat, save_path=de_sens_fig)
    console.print(f"Saved DE sensitivity heatmap to [yellow]{de_sens_fig}[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="MetaBench – Metaheuristic Optimization Framework CLI")
    parser.add_argument("--quick-demo", action="store_true", help="Run quick 4-benchmark demonstration")
    parser.add_argument("--full-benchmark", action="store_true", help="Run full 2,400 trials campaign")
    parser.add_argument("--sensitivity", action="store_true", help="Run hyperparameter sensitivity analysis")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--budget", type=int, default=10000, help="Function evaluations budget per trial")
    parser.add_argument("--seeds", type=int, default=30, help="Number of independent seeds per benchmark (default: 30)")

    args = parser.parse_args()

    if args.quick_demo:
        run_quick_demo()
    elif args.sensitivity:
        os.makedirs("results", exist_ok=True)
        os.makedirs("figures", exist_ok=True)
        run_sensitivity_study()
    elif args.full_benchmark:
        run_full_campaign(max_workers=args.workers, budget=args.budget, num_seeds=args.seeds)
    else:
        # Default action if no arguments provided: run full campaign!
        run_full_campaign(max_workers=args.workers, budget=args.budget, num_seeds=args.seeds)


if __name__ == "__main__":
    main()

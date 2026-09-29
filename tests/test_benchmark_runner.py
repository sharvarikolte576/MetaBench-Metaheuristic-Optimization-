"""Unit tests for Benchmark Runner and Results Analyzer."""

import pytest
import pandas as pd
from metabench.benchmark_runner import BenchmarkRunner
from metabench.results_analyzer import ResultsAnalyzer


def test_mini_benchmark_campaign():
    # Fast test with 2 algorithms, 2 benchmarks, 3 seeds, budget=500
    runner = BenchmarkRunner(
        algorithms=["PSO", "DE"],
        benchmarks=["Sphere", "Booth"],
        dimension=2,
        budget=500,
        num_seeds=3,
        max_workers=2
    )
    raw_df, summary_df, curves = runner.run(record_sample_telemetry=True, show_progress=False)

    # Expected trials = 2 * 2 * 3 = 12
    assert len(raw_df) == 12
    # Expected summaries = 2 * 2 = 4
    assert len(summary_df) == 4

    analyzer = ResultsAnalyzer(raw_df, summary_df)
    tradeoff = analyzer.generate_tradeoff_matrix()
    assert len(tradeoff) == 2
    assert "average_rank" in tradeoff.columns
    assert "composite_score" in tradeoff.columns

    md_table = analyzer.format_statistical_table_markdown()
    assert "mean_error" in md_table


def test_non_scalable_benchmark_runner():
    # Bukin N.6 is 2D and non-scalable; runner initialized with dim=10 must handle it gracefully
    runner = BenchmarkRunner(
        algorithms=["PSO", "DE", "GA", "SA"],
        benchmarks=["Bukin N.6", "Sphere"],
        dimension=10,
        budget=300,
        num_seeds=2,
        max_workers=2
    )
    raw_df, summary_df, _ = runner.run(show_progress=False)
    assert len(raw_df) == 4 * 2 * 2  # 16 trials
    bukin_trials = raw_df[raw_df["benchmark"] == "Bukin N.6"]
    assert len(bukin_trials) == 8
    assert (bukin_trials["dimension"] == 2).all()


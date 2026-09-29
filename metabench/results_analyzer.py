"""
Results Analyzer Module for MetaBench.

Evaluates performance trade-offs, landscape category breakdowns,
pairwise Wilcoxon significance matrices, and ranking summaries.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from metabench.benchmarks import create_benchmark
from metabench.statistical_metrics import wilcoxon_signed_rank_test, friedman_ranking
from metabench.metrics import calculate_composite_score


class ResultsAnalyzer:
    """Analyzes benchmark campaigns and synthesizes comparative trade-offs."""

    def __init__(self, raw_trials_df: pd.DataFrame, summary_df: pd.DataFrame):
        self.raw_df = raw_trials_df.copy()
        self.summary_df = summary_df.copy()
        self._enrich_landscape_metadata()

    def _enrich_landscape_metadata(self):
        """Annotates summary table with benchmark landscape characteristics."""
        landscape_map = {}
        for bench_name in self.summary_df["benchmark"].unique():
            try:
                b = create_benchmark(bench_name)
                landscape_map[bench_name] = {
                    "landscape_type": b.landscape_type,
                    "separability": b.separability,
                    "dimension": b.dimension
                }
            except Exception:
                landscape_map[bench_name] = {
                    "landscape_type": "unknown",
                    "separability": "unknown",
                    "dimension": 10
                }

        self.summary_df["landscape_type"] = self.summary_df["benchmark"].map(
            lambda b: landscape_map.get(b, {}).get("landscape_type", "unknown")
        )
        self.summary_df["separability"] = self.summary_df["benchmark"].map(
            lambda b: landscape_map.get(b, {}).get("separability", "unknown")
        )

    def compute_category_summary(self) -> pd.DataFrame:
        """Computes mean error, robustness, and execution time grouped by landscape type."""
        grouped = self.summary_df.groupby(["landscape_type", "algorithm"]).agg({
            "mean_error": "mean",
            "median_error": "mean",
            "robustness_index": "mean",
            "success_rate": "mean",
            "avg_execution_time_sec": "mean"
        }).reset_index()
        return grouped.sort_values(by=["landscape_type", "mean_error"])

    def compute_pairwise_wilcoxon(self, benchmark_name: str) -> pd.DataFrame:
        """
        Computes pairwise Wilcoxon signed-rank test between all algorithms
        for a specific benchmark function.
        """
        bench_trials = self.raw_df[self.raw_df["benchmark"] == benchmark_name]
        algos = sorted(bench_trials["algorithm"].unique())
        records = []

        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                a1 = algos[i]
                a2 = algos[j]
                s1 = bench_trials[bench_trials["algorithm"] == a1].sort_values("seed")["best_fitness"].values
                s2 = bench_trials[bench_trials["algorithm"] == a2].sort_values("seed")["best_fitness"].values

                min_len = min(len(s1), len(s2))
                test_res = wilcoxon_signed_rank_test(s1[:min_len], s2[:min_len])

                winner_name = "Tie"
                if test_res["winner"] == "A":
                    winner_name = a1
                elif test_res["winner"] == "B":
                    winner_name = a2

                records.append({
                    "benchmark": benchmark_name,
                    "comparison": f"{a1} vs {a2}",
                    "stat": test_res["statistic"],
                    "p_value": test_res["p_value"],
                    "significant": test_res["significant"],
                    "winner": winner_name
                })

        return pd.DataFrame(records)

    def generate_tradeoff_matrix(self) -> pd.DataFrame:
        """
        Aggregates key trade-offs across all 20 benchmarks:
        - Accuracy Rank (Friedman)
        - Mean Robustness Index
        - Overall Success Rate
        - Mean Run Time (Efficiency)
        - Composite Score
        """
        friedman_df = friedman_ranking(self.summary_df, algo_col="algorithm", bench_col="benchmark", score_col="mean_error")
        agg = self.summary_df.groupby("algorithm").agg({
            "mean_error": "mean",
            "robustness_index": "mean",
            "success_rate": "mean",
            "avg_execution_time_sec": "mean"
        }).reset_index()

        merged = pd.merge(agg, friedman_df, on="algorithm")

        # Composite score
        scores = []
        for _, row in merged.iterrows():
            sc = calculate_composite_score(
                mean_error=row["mean_error"],
                robustness=row["robustness_index"],
                speed_evals=int(row["avg_execution_time_sec"] * 5000),  # scaled proxy
                max_evals=10000
            )
            scores.append(sc)

        merged["composite_score"] = scores
        return merged.sort_values(by="average_rank").reset_index(drop=True)

    def format_statistical_table_markdown(self) -> str:
        """Formats the 4 core non-parametric statistical metrics (Mean, Variance, Best, Worst) into Markdown."""
        cols = [
            "algorithm", "benchmark", "trials",
            "mean_error", "variance_error", "best_error", "worst_error",
            "robustness_index", "success_rate"
        ]
        subset = self.summary_df[cols].copy()
        # Format floating points nicely
        for col in ["mean_error", "variance_error", "best_error", "worst_error"]:
            subset[col] = subset[col].apply(lambda x: f"{x:.4e}" if abs(x) >= 1e-4 or abs(x) == 0 else f"{x:.2e}")
        subset["robustness_index"] = subset["robustness_index"].apply(lambda x: f"{x:.3f}")
        subset["success_rate"] = subset["success_rate"].apply(lambda x: f"{x * 100:.1f}%")

        return subset.to_markdown(index=False)

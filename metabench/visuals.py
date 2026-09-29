"""
Visualization Module for MetaBench.

Generates publication-quality figures:
1. Convergence curves with confidence intervals
2. Final error distribution boxplots
3. Population diversity / exploration-exploitation phase transition curves
4. Hyperparameter sensitivity heatmaps
5. Algorithmic trade-off radar charts
"""

from typing import List, Dict, Any, Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set styling aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.autolayout": True
})

ALGO_COLORS = {
    "PSO": "#1f77b4",  # Blue
    "GA": "#2ca02c",   # Green
    "DE": "#d62728",   # Crimson / Red
    "SA": "#ff7f0e"    # Orange
}


def plot_convergence_curves(
    telemetry_curves: Dict[str, List[np.ndarray]],
    benchmarks: List[str],
    algorithms: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots mean convergence curves with shaded interquartile bands
    for each algorithm across specified benchmarks.
    """
    algos = algorithms or ["PSO", "GA", "DE", "SA"]
    n_bench = len(benchmarks)
    cols = min(3, n_bench)
    rows = int(np.ceil(n_bench / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, bench in enumerate(benchmarks):
        ax = axes_flat[idx]
        for algo in algos:
            key = f"{algo}_{bench}"
            curves = telemetry_curves.get(key, [])
            if not curves:
                continue

            # Interpolate curves onto a common evaluation grid
            common_evals = np.linspace(0, 10000, 200)
            interp_fits = []
            for c in curves:
                if len(c) > 1:
                    ev = c[:, 0]
                    fv = np.maximum(c[:, 1], 1e-16)
                    # Use log-space interpolation for stability
                    interp_fv = np.interp(common_evals, ev, np.log10(fv))
                    interp_fits.append(interp_fv)

            if interp_fits:
                interp_arr = np.array(interp_fits)
                mean_log_fit = np.mean(interp_arr, axis=0)
                std_log_fit = np.std(interp_arr, axis=0)

                color = ALGO_COLORS.get(algo, "#333333")
                ax.plot(common_evals, mean_log_fit, label=algo, color=color, lw=2)
                ax.fill_between(
                    common_evals,
                    mean_log_fit - std_log_fit,
                    mean_log_fit + std_log_fit,
                    color=color,
                    alpha=0.18
                )

        ax.set_title(f"Convergence: {bench}", fontweight="bold")
        ax.set_xlabel("Function Evaluations")
        ax.set_ylabel(r"$\log_{10}(\text{Best Fitness})$")
        ax.set_xlim(0, 10000)
        ax.legend(frameon=True)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_error_boxplots(
    raw_df: pd.DataFrame,
    benchmarks: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots boxplots of final error to optimum across 30 runs for each algorithm.
    """
    subset = raw_df[raw_df["benchmark"].isin(benchmarks)].copy()
    # Log-transform error with small eps for clean visualization
    subset["log10_error"] = np.log10(np.maximum(subset["error_to_optimum"], 1e-14))

    n_bench = len(benchmarks)
    cols = min(2, n_bench)
    rows = int(np.ceil(n_bench / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, bench in enumerate(benchmarks):
        ax = axes_flat[idx]
        b_df = subset[subset["benchmark"] == bench]
        sns.boxplot(
            data=b_df,
            x="algorithm",
            y="log10_error",
            palette=ALGO_COLORS,
            ax=ax,
            width=0.45,
            fliersize=3
        )
        sns.stripplot(
            data=b_df,
            x="algorithm",
            y="log10_error",
            color="black",
            alpha=0.35,
            size=4,
            jitter=0.2,
            ax=ax
        )
        ax.set_title(f"Error Distribution: {bench}", fontweight="bold")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(r"$\log_{10}(\text{Error to Optimum})$")

    for j in range(idx + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_diversity_dynamics(
    trackers_dict: Dict[str, Any],
    bench_name: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots population diversity decline illustrating exploration to exploitation transition.
    trackers_dict: {algo_name: TrajectoryTracker}
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for algo, tracker in trackers_dict.items():
        div_curve = tracker.get_diversity_curve()
        if len(div_curve) > 0:
            evals = div_curve[:, 0]
            divs = div_curve[:, 1]
            color = ALGO_COLORS.get(algo, "#333333")
            ax.plot(evals, divs, label=algo, color=color, lw=2.2)

    ax.set_title(f"Population Diversity Dynamics: {bench_name}", fontweight="bold")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Mean Dispersion to Centroid")
    ax.set_xlim(0, 10000)
    ax.legend(frameon=True)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_parameter_sensitivity_heatmap(
    oat_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots heatmap of hyperparameter sensitivity across parameter variations.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = oat_df.pivot(index="parameter", columns="parameter_value", values="mean_fitness")

    # Normalize row-wise for visual comparability
    row_min = pivot.min(axis=1)
    row_max = pivot.max(axis=1)
    denom = np.maximum(1e-12, row_max - row_min)
    norm_pivot = pivot.sub(row_min, axis=0).div(denom, axis=0)

    sns.heatmap(
        norm_pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Normalized Mean Fitness (0=Optimal)"},
        ax=ax
    )
    ax.set_title(f"Hyperparameter Sensitivity Matrix ({oat_df['algorithm'].iloc[0]})", fontweight="bold")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Hyperparameter")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_radar_tradeoff(
    tradeoff_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots a radar (spider) chart evaluating multi-attribute trade-offs:
    Accuracy, Robustness, Speed, Success Rate, and Scalability.
    """
    categories = ["Accuracy", "Robustness", "Efficiency", "Success Rate", "Overall Score"]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))

    for _, row in tradeoff_df.iterrows():
        algo = row["algorithm"]
        # Normalize attributes between 0.1 and 1.0
        acc = 1.0 / (1.0 + np.log1p(max(0.0, row["mean_error"])))
        rob = float(np.clip(row["robustness_index"], 0.0, 1.0))
        eff = 1.0 / (1.0 + row["avg_execution_time_sec"])
        succ = float(np.clip(row["success_rate"], 0.0, 1.0))
        score = float(row["composite_score"] / 100.0)

        values = [acc, rob, eff, succ, score]
        values += values[:1]

        color = ALGO_COLORS.get(algo, "#333333")
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=algo, color=color)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, size=10, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.05)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.title("Algorithmic Performance Trade-Off Matrix", size=13, fontweight="bold", y=1.08)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig

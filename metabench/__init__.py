"""
MetaBench: Modular Python Research Framework for Metaheuristic Optimization.
"""

from metabench.benchmarks import (
    BenchmarkFunction,
    create_benchmark,
    get_all_benchmark_names,
    get_benchmark_suite
)
from metabench.algorithms import (
    create_algorithm,
    ParticleSwarmOptimization,
    GeneticAlgorithm,
    DifferentialEvolution,
    SimulatedAnnealing
)
from metabench.framework import (
    OptimizationProblem,
    OptimizationResult,
    BaseMetaheuristic
)
from metabench.trajectory_tracker import TrajectoryTracker, TelemetryRecord
from metabench.convergence_analysis import ConvergenceAnalyzer
from metabench.parameter_sensitivity import ParameterSensitivityAnalyzer
from metabench.statistical_metrics import (
    compute_trial_statistics,
    wilcoxon_signed_rank_test,
    friedman_ranking
)
from metabench.metrics import (
    calculate_robustness_index,
    calculate_success_rate,
    calculate_convergence_speed,
    calculate_composite_score
)
from metabench.benchmark_runner import BenchmarkRunner
from metabench.results_analyzer import ResultsAnalyzer

__version__ = "1.0.0"
__all__ = [
    "BenchmarkFunction",
    "create_benchmark",
    "get_all_benchmark_names",
    "get_benchmark_suite",
    "create_algorithm",
    "ParticleSwarmOptimization",
    "GeneticAlgorithm",
    "DifferentialEvolution",
    "SimulatedAnnealing",
    "OptimizationProblem",
    "OptimizationResult",
    "BaseMetaheuristic",
    "TrajectoryTracker",
    "TelemetryRecord",
    "ConvergenceAnalyzer",
    "ParameterSensitivityAnalyzer",
    "compute_trial_statistics",
    "wilcoxon_signed_rank_test",
    "friedman_ranking",
    "calculate_robustness_index",
    "calculate_success_rate",
    "calculate_convergence_speed",
    "calculate_composite_score",
    "BenchmarkRunner",
    "ResultsAnalyzer"
]

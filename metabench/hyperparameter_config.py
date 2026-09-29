"""
Hyperparameter Configuration Module for MetaBench.

Defines 15+ configurable hyperparameters across 4 metaheuristics
(PSO, GA, DE, SA) with validation schemas, preset profiles, and sweep generators.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional


@dataclass
class PSOConfig:
    """Hyperparameters for Particle Swarm Optimization."""
    pop_size: int = 50
    w: float = 0.7298            # Inertia weight (Clerc-Kennedy constriction)
    w_decay: float = 0.99        # Damping factor per generation
    w_min: float = 0.40          # Minimum inertia threshold
    c1: float = 1.49618          # Cognitive acceleration coefficient
    c2: float = 1.49618          # Social acceleration coefficient
    v_max_ratio: float = 0.20    # Velocity clamp as fraction of bound width


@dataclass
class GAConfig:
    """Hyperparameters for Real-Coded Genetic Algorithm."""
    pop_size: int = 50
    crossover_rate: float = 0.85 # Probability of recombination
    mutation_rate: float = 0.15  # Probability of mutation per gene
    mutation_scale: float = 0.10 # Standard deviation of Gaussian mutation (fraction of bound)
    tournament_size: int = 3     # Selection pressure
    elitism_ratio: float = 0.04  # Fraction of top elites preserved untouched


@dataclass
class DEConfig:
    """Hyperparameters for Differential Evolution."""
    pop_size: int = 50
    F: float = 0.50              # Differential mutation scaling factor
    CR: float = 0.70             # Binomial crossover probability
    strategy: str = "rand1bin"   # 'rand1bin' or 'best1bin'


@dataclass
class SAConfig:
    """Hyperparameters for Simulated Annealing."""
    initial_temp: float = 100.0   # Initial thermal energy
    cooling_rate: float = 0.985   # Geometric cooling coefficient alpha
    step_size: float = 0.08       # Perturbation step as fraction of search bound
    cooling_schedule: str = "geometric"  # 'geometric', 'linear', or 'logarithmic'
    restart_stagnation: int = 1500 # Iterations of non-improvement before thermal reheat


def get_default_config(algo_name: str) -> Dict[str, Any]:
    """Get default hyperparameter dictionary for given algorithm."""
    algo = algo_name.upper().strip()
    if algo == "PSO":
        return asdict(PSOConfig())
    elif algo == "GA":
        return asdict(GAConfig())
    elif algo == "DE":
        return asdict(DEConfig())
    elif algo == "SA":
        return asdict(SAConfig())
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}. Supported: PSO, GA, DE, SA")


def get_preset_configs(algo_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns named hyperparameter presets:
    'standard', 'high_exploration', 'high_exploitation'
    """
    algo = algo_name.upper().strip()
    presets = {}

    if algo == "PSO":
        presets["standard"] = asdict(PSOConfig())
        presets["high_exploration"] = asdict(PSOConfig(w=0.9, w_decay=0.999, c1=2.0, c2=0.8, v_max_ratio=0.5))
        presets["high_exploitation"] = asdict(PSOConfig(w=0.5, w_decay=0.98, c1=0.8, c2=2.2, v_max_ratio=0.1))
    elif algo == "GA":
        presets["standard"] = asdict(GAConfig())
        presets["high_exploration"] = asdict(GAConfig(mutation_rate=0.35, mutation_scale=0.25, crossover_rate=0.9, tournament_size=2))
        presets["high_exploitation"] = asdict(GAConfig(mutation_rate=0.05, mutation_scale=0.03, crossover_rate=0.7, tournament_size=5))
    elif algo == "DE":
        presets["standard"] = asdict(DEConfig())
        presets["high_exploration"] = asdict(DEConfig(F=0.85, CR=0.9, strategy="rand1bin"))
        presets["high_exploitation"] = asdict(DEConfig(F=0.35, CR=0.4, strategy="best1bin"))
    elif algo == "SA":
        presets["standard"] = asdict(SAConfig())
        presets["high_exploration"] = asdict(SAConfig(initial_temp=500.0, cooling_rate=0.995, step_size=0.20))
        presets["high_exploitation"] = asdict(SAConfig(initial_temp=50.0, cooling_rate=0.95, step_size=0.02))

    return presets


def get_sensitivity_sweep_grid(algo_name: str) -> Dict[str, List[Any]]:
    """
    Returns parameter variation grid for sensitivity analysis.
    Covers at least 15 distinct parameter variation axes across the framework.
    """
    algo = algo_name.upper().strip()
    if algo == "PSO":
        return {
            "w": [0.3, 0.5, 0.7298, 0.9],
            "c1": [0.5, 1.0, 1.496, 2.0],
            "c2": [0.5, 1.0, 1.496, 2.0],
            "v_max_ratio": [0.05, 0.1, 0.2, 0.4]
        }
    elif algo == "GA":
        return {
            "mutation_rate": [0.02, 0.08, 0.15, 0.30],
            "mutation_scale": [0.02, 0.05, 0.10, 0.20],
            "crossover_rate": [0.5, 0.7, 0.85, 0.95],
            "tournament_size": [2, 3, 5, 7]
        }
    elif algo == "DE":
        return {
            "F": [0.2, 0.5, 0.8, 1.0],
            "CR": [0.1, 0.5, 0.7, 0.9],
            "strategy": ["rand1bin", "best1bin"],
            "pop_size": [30, 50, 100]
        }
    elif algo == "SA":
        return {
            "initial_temp": [10.0, 50.0, 100.0, 500.0],
            "cooling_rate": [0.90, 0.95, 0.985, 0.995],
            "step_size": [0.02, 0.05, 0.08, 0.15],
            "cooling_schedule": ["geometric", "linear"]
        }
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

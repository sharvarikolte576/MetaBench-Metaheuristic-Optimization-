"""
Convergence Dynamics Analysis Module for MetaBench.

Calculates mathematical explainability variables:
- Convergence velocity and acceleration
- Stagnation index and plateau durations
- Exploration vs exploitation phase transitions via population diversity
- Area Under the Convergence Curve (AUC)
"""

from typing import Dict, Any, List, Tuple
import numpy as np

from metabench.trajectory_tracker import TrajectoryTracker, TelemetryRecord


class ConvergenceAnalyzer:
    """Extracts analytical variables explaining convergence behavior."""

    @staticmethod
    def compute_convergence_velocity(
        curve: np.ndarray,
        eps: float = 1e-12
    ) -> Dict[str, float]:
        """
        Computes rate of change of log fitness across optimization milestones:
        curve: (K, 2) where col 0 is evaluations and col 1 is best_fitness.
        """
        if len(curve) < 2:
            return {
                "initial_velocity": 0.0,
                "terminal_velocity": 0.0,
                "mean_velocity": 0.0
            }

        evals = curve[:, 0]
        fits = np.maximum(curve[:, 1], eps)
        log_fits = np.log10(fits)

        # Delta log-fit over delta evaluations
        d_evals = np.diff(evals)
        d_log_fits = np.diff(log_fits)

        # Avoid divide-by-zero
        valid = d_evals > 0
        if not np.any(valid):
            return {
                "initial_velocity": 0.0,
                "terminal_velocity": 0.0,
                "mean_velocity": 0.0
            }

        velocities = -d_log_fits[valid] / d_evals[valid]  # positive when improving

        initial_vel = float(velocities[0]) if len(velocities) > 0 else 0.0
        terminal_vel = float(velocities[-1]) if len(velocities) > 0 else 0.0
        mean_vel = float(np.mean(velocities))

        return {
            "initial_velocity": initial_vel,
            "terminal_velocity": terminal_vel,
            "mean_velocity": mean_vel
        }

    @staticmethod
    def compute_stagnation_metrics(
        history: List[TelemetryRecord],
        eps: float = 1e-9
    ) -> Dict[str, Any]:
        """
        Quantifies stagnation behavior:
        - max_consecutive_stagnation_evals
        - stagnation_ratio (fraction of evaluations where best fitness made zero progress)
        """
        if len(history) < 2:
            return {
                "stagnation_ratio": 0.0,
                "max_stagnation_evals": 0,
                "total_improvements": 0
            }

        evals = [r.evaluations for r in history]
        fits = [r.best_fitness for r in history]

        stagnant_evals = 0
        max_stagnant_evals = 0
        current_streak = 0
        improvements = 0

        for i in range(1, len(history)):
            d_eval = evals[i] - evals[i - 1]
            if (fits[i - 1] - fits[i]) > eps:
                # Improvement made
                improvements += 1
                max_stagnant_evals = max(max_stagnant_evals, current_streak)
                current_streak = 0
            else:
                stagnant_evals += d_eval
                current_streak += d_eval

        max_stagnant_evals = max(max_stagnant_evals, current_streak)
        total_evals = max(1, evals[-1] - evals[0])
        stagnation_ratio = float(stagnant_evals / total_evals)

        return {
            "stagnation_ratio": min(1.0, stagnation_ratio),
            "max_stagnation_evals": int(max_stagnant_evals),
            "total_improvements": int(improvements)
        }

    @staticmethod
    def compute_exploration_exploitation(
        history: List[TelemetryRecord]
    ) -> Dict[str, Any]:
        """
        Evaluates population diversity curve to detect the exploration-to-exploitation
        transition point (when diversity drops below 15% of peak diversity).
        """
        if not history:
            return {
                "initial_diversity": 0.0,
                "final_diversity": 0.0,
                "diversity_loss_rate": 0.0,
                "exploitation_transition_eval": 0
            }

        divs = np.array([r.population_diversity for r in history], dtype=np.float64)
        evals = np.array([r.evaluations for r in history], dtype=np.float64)

        init_div = float(divs[0]) if len(divs) > 0 else 0.0
        final_div = float(divs[-1]) if len(divs) > 0 else 0.0

        max_div = np.max(divs) if len(divs) > 0 else 0.0
        threshold = 0.15 * max_div

        below_thresh = np.where(divs <= threshold)[0]
        if len(below_thresh) > 0:
            transition_eval = int(evals[below_thresh[0]])
        else:
            transition_eval = int(evals[-1])

        div_loss = (init_div - final_div) / max(1e-12, init_div)

        return {
            "initial_diversity": init_div,
            "final_diversity": final_div,
            "diversity_loss_rate": float(div_loss),
            "exploitation_transition_eval": transition_eval
        }

    @staticmethod
    def compute_auc(curve: np.ndarray, max_evals: int = 10000, eps: float = 1e-12) -> float:
        """
        Area Under the Log-Fitness Convergence Curve.
        Normalized by max_evals. Lower values indicate superior rapid convergence.
        """
        if len(curve) < 2:
            return 0.0

        evals = curve[:, 0]
        fits = np.maximum(curve[:, 1], eps)
        log_fits = np.log10(fits)

        # Trapezoidal numerical integration
        auc = np.trapz(log_fits, evals)
        return float(auc / max(1.0, max_evals))

    @classmethod
    def analyze(cls, tracker: TrajectoryTracker, max_evals: int = 10000) -> Dict[str, Any]:
        """Extract comprehensive convergence explainability summary from telemetry."""
        curve = tracker.get_convergence_curve()
        history = tracker.history

        vel = cls.compute_convergence_velocity(curve)
        stag = cls.compute_stagnation_metrics(history)
        exp = cls.compute_exploration_exploitation(history)
        auc = cls.compute_auc(curve, max_evals=max_evals)

        return {
            **vel,
            **stag,
            **exp,
            "convergence_auc": auc
        }

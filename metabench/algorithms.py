"""
Metaheuristic Algorithms Module for MetaBench.

Implements 4 distinct, vectorized metaheuristics:
1. Particle Swarm Optimization (PSO)
2. Real-Coded Genetic Algorithm (GA)
3. Differential Evolution (DE)
4. Simulated Annealing (SA)

All algorithms strictly respect the function evaluation budget, log telemetry,
and support scalable continuous search spaces up to 100-D.
"""

from typing import Dict, Any, Optional
import time
import numpy as np

from metabench.framework import BaseMetaheuristic, OptimizationProblem, OptimizationResult
from metabench.trajectory_tracker import TrajectoryTracker
from metabench.hyperparameter_config import (
    PSOConfig, GAConfig, DEConfig, SAConfig, get_default_config
)


# -----------------------------------------------------------------------------
# 1. Particle Swarm Optimization (PSO)
# -----------------------------------------------------------------------------
class ParticleSwarmOptimization(BaseMetaheuristic):
    """
    Standard Particle Swarm Optimization with Clerc-Kennedy constriction /
    inertia damping and velocity clamping.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = get_default_config("PSO")
        if config:
            cfg.update(config)
        super().__init__("PSO", cfg)

    def optimize(
        self,
        problem: OptimizationProblem,
        seed: Optional[int] = None,
        tracker: Optional[TrajectoryTracker] = None
    ) -> OptimizationResult:
        rng = np.random.default_rng(seed)
        start_time = time.perf_counter()

        pop_size = int(self.config.get("pop_size", 50))
        w = float(self.config.get("w", 0.7298))
        w_decay = float(self.config.get("w_decay", 0.99))
        w_min = float(self.config.get("w_min", 0.40))
        c1 = float(self.config.get("c1", 1.49618))
        c2 = float(self.config.get("c2", 1.49618))
        v_max_ratio = float(self.config.get("v_max_ratio", 0.20))

        dim = problem.dimension
        low, high = problem.lower_bounds, problem.upper_bounds
        bound_span = high - low
        v_max = v_max_ratio * bound_span

        if tracker is None:
            tracker = TrajectoryTracker()

        # Initialize swarm positions and velocities
        positions = rng.uniform(low, high, size=(pop_size, dim))
        velocities = rng.uniform(-v_max, v_max, size=(pop_size, dim))

        # Initial evaluation
        fitnesses = problem.evaluate(positions)
        pbest_positions = positions.copy()
        pbest_fitnesses = fitnesses.copy()

        gbest_idx = np.argmin(pbest_fitnesses)
        gbest_fitness = float(pbest_fitnesses[gbest_idx])
        gbest_position = pbest_positions[gbest_idx].copy()

        iteration = 0
        tracker.record_step(
            iteration=iteration,
            evaluations=problem.evaluations_count,
            population=positions,
            fitnesses=fitnesses,
            best_fitness=gbest_fitness,
            best_position=gbest_position,
            step_size=float(np.mean(np.linalg.norm(velocities, axis=1)))
        )

        while problem.remaining_evaluations > 0:
            iteration += 1

            # Update velocity
            r1 = rng.uniform(0.0, 1.0, size=(pop_size, dim))
            r2 = rng.uniform(0.0, 1.0, size=(pop_size, dim))
            cognitive = c1 * r1 * (pbest_positions - positions)
            social = c2 * r2 * (gbest_position - positions)
            velocities = w * velocities + cognitive + social

            # Clamp velocities
            velocities = np.clip(velocities, -v_max, v_max)

            # Update positions & boundary clamp
            positions = np.clip(positions + velocities, low, high)

            # Evaluate (handles partial batches at end of budget)
            remaining = problem.remaining_evaluations
            eval_size = min(pop_size, remaining)
            fitnesses = problem.evaluate(positions[:eval_size])

            # Update personal bests
            for i in range(eval_size):
                if fitnesses[i] < pbest_fitnesses[i]:
                    pbest_fitnesses[i] = fitnesses[i]
                    pbest_positions[i] = positions[i].copy()

            # Update global best
            current_min_idx = np.argmin(pbest_fitnesses[:pop_size])
            if pbest_fitnesses[current_min_idx] < gbest_fitness:
                gbest_fitness = float(pbest_fitnesses[current_min_idx])
                gbest_position = pbest_positions[current_min_idx].copy()

            # Damping of inertia weight
            w = max(w_min, w * w_decay)

            tracker.record_step(
                iteration=iteration,
                evaluations=problem.evaluations_count,
                population=positions[:eval_size],
                fitnesses=fitnesses[:eval_size],
                best_fitness=gbest_fitness,
                best_position=gbest_position,
                step_size=float(np.mean(np.linalg.norm(velocities[:eval_size], axis=1)))
            )

        elapsed = time.perf_counter() - start_time
        err = abs(gbest_fitness - problem.benchmark.global_minimum_val)

        return OptimizationResult(
            algorithm_name="PSO",
            benchmark_name=problem.benchmark.name,
            dimension=dim,
            seed=seed if seed is not None else -1,
            best_position=gbest_position,
            best_fitness=gbest_fitness,
            error_to_optimum=float(err),
            total_evaluations=problem.evaluations_count,
            iterations=iteration,
            execution_time_sec=elapsed,
            hyperparameters=dict(self.config),
            telemetry=tracker
        )


# -----------------------------------------------------------------------------
# 2. Real-Coded Genetic Algorithm (GA)
# -----------------------------------------------------------------------------
class GeneticAlgorithm(BaseMetaheuristic):
    """
    Real-Coded Genetic Algorithm featuring tournament selection,
    blend/arithmetic crossover, Gaussian mutation, and elitism preservation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = get_default_config("GA")
        if config:
            cfg.update(config)
        super().__init__("GA", cfg)

    def optimize(
        self,
        problem: OptimizationProblem,
        seed: Optional[int] = None,
        tracker: Optional[TrajectoryTracker] = None
    ) -> OptimizationResult:
        rng = np.random.default_rng(seed)
        start_time = time.perf_counter()

        pop_size = int(self.config.get("pop_size", 50))
        crossover_rate = float(self.config.get("crossover_rate", 0.85))
        mutation_rate = float(self.config.get("mutation_rate", 0.15))
        mutation_scale = float(self.config.get("mutation_scale", 0.10))
        tournament_size = int(self.config.get("tournament_size", 3))
        elitism_ratio = float(self.config.get("elitism_ratio", 0.04))

        dim = problem.dimension
        low, high = problem.lower_bounds, problem.upper_bounds
        bound_span = high - low
        n_elites = max(1, int(round(elitism_ratio * pop_size)))

        if tracker is None:
            tracker = TrajectoryTracker()

        # Initialize population
        population = rng.uniform(low, high, size=(pop_size, dim))
        fitnesses = problem.evaluate(population)

        best_idx = np.argmin(fitnesses)
        best_fitness = float(fitnesses[best_idx])
        best_position = population[best_idx].copy()

        iteration = 0
        tracker.record_step(
            iteration=iteration,
            evaluations=problem.evaluations_count,
            population=population,
            fitnesses=fitnesses,
            best_fitness=best_fitness,
            best_position=best_position,
            step_size=0.0
        )

        while problem.remaining_evaluations > 0:
            iteration += 1

            # Elitism: retain top n_elites
            sorted_indices = np.argsort(fitnesses)
            elites = population[sorted_indices[:n_elites]].copy()

            # Tournament Selection
            offspring = []
            needed_offspring = pop_size - n_elites

            while len(offspring) < needed_offspring:
                # Select parent 1
                cand1 = rng.integers(0, pop_size, size=tournament_size)
                p1_idx = cand1[np.argmin(fitnesses[cand1])]
                p1 = population[p1_idx]

                # Select parent 2
                cand2 = rng.integers(0, pop_size, size=tournament_size)
                p2_idx = cand2[np.argmin(fitnesses[cand2])]
                p2 = population[p2_idx]

                # Crossover (Arithmetic / Blend)
                if rng.random() < crossover_rate:
                    alpha = rng.uniform(0.0, 1.0, size=dim)
                    c1_pos = alpha * p1 + (1.0 - alpha) * p2
                    c2_pos = (1.0 - alpha) * p1 + alpha * p2
                else:
                    c1_pos = p1.copy()
                    c2_pos = p2.copy()

                offspring.append(c1_pos)
                if len(offspring) < needed_offspring:
                    offspring.append(c2_pos)

            offspring_arr = np.array(offspring[:needed_offspring])

            # Mutation: Gaussian perturbation
            mut_mask = rng.random(size=offspring_arr.shape) < mutation_rate
            mut_step = rng.normal(0.0, mutation_scale * bound_span, size=offspring_arr.shape)
            offspring_arr += mut_mask * mut_step

            # Clamp offspring to boundaries
            offspring_arr = np.clip(offspring_arr, low, high)

            # Assemble new population candidate
            new_population = np.vstack([elites, offspring_arr])

            # Evaluate offspring only (elites already have valid fitness)
            remaining = problem.remaining_evaluations
            eval_size = min(needed_offspring, remaining)
            new_fitnesses = np.empty(pop_size, dtype=np.float64)
            new_fitnesses[:n_elites] = fitnesses[sorted_indices[:n_elites]]

            if eval_size > 0:
                evaluated_offspring_fits = problem.evaluate(offspring_arr[:eval_size])
                new_fitnesses[n_elites : n_elites + eval_size] = evaluated_offspring_fits
            if eval_size < needed_offspring:
                new_fitnesses[n_elites + eval_size:] = float("inf")

            population = new_population
            fitnesses = new_fitnesses

            current_min_idx = np.argmin(fitnesses)
            if fitnesses[current_min_idx] < best_fitness:
                best_fitness = float(fitnesses[current_min_idx])
                best_position = population[current_min_idx].copy()

            tracker.record_step(
                iteration=iteration,
                evaluations=problem.evaluations_count,
                population=population,
                fitnesses=fitnesses,
                best_fitness=best_fitness,
                best_position=best_position,
                step_size=float(np.mean(np.linalg.norm(mut_step, axis=1)))
            )

        elapsed = time.perf_counter() - start_time
        err = abs(best_fitness - problem.benchmark.global_minimum_val)

        return OptimizationResult(
            algorithm_name="GA",
            benchmark_name=problem.benchmark.name,
            dimension=dim,
            seed=seed if seed is not None else -1,
            best_position=best_position,
            best_fitness=best_fitness,
            error_to_optimum=float(err),
            total_evaluations=problem.evaluations_count,
            iterations=iteration,
            execution_time_sec=elapsed,
            hyperparameters=dict(self.config),
            telemetry=tracker
        )


# -----------------------------------------------------------------------------
# 3. Differential Evolution (DE)
# -----------------------------------------------------------------------------
class DifferentialEvolution(BaseMetaheuristic):
    """
    Differential Evolution (DE) with DE/rand/1/bin and DE/best/1/bin strategies.
    Renowned for global optimization efficiency in continuous domains.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = get_default_config("DE")
        if config:
            cfg.update(config)
        super().__init__("DE", cfg)

    def optimize(
        self,
        problem: OptimizationProblem,
        seed: Optional[int] = None,
        tracker: Optional[TrajectoryTracker] = None
    ) -> OptimizationResult:
        rng = np.random.default_rng(seed)
        start_time = time.perf_counter()

        pop_size = int(self.config.get("pop_size", 50))
        F = float(self.config.get("F", 0.50))
        CR = float(self.config.get("CR", 0.70))
        strategy = str(self.config.get("strategy", "rand1bin")).lower()

        dim = problem.dimension
        low, high = problem.lower_bounds, problem.upper_bounds

        if tracker is None:
            tracker = TrajectoryTracker()

        # Initialize target vectors
        population = rng.uniform(low, high, size=(pop_size, dim))
        fitnesses = problem.evaluate(population)

        best_idx = np.argmin(fitnesses)
        best_fitness = float(fitnesses[best_idx])
        best_position = population[best_idx].copy()

        iteration = 0
        tracker.record_step(
            iteration=iteration,
            evaluations=problem.evaluations_count,
            population=population,
            fitnesses=fitnesses,
            best_fitness=best_fitness,
            best_position=best_position,
            step_size=0.0
        )

        while problem.remaining_evaluations > 0:
            iteration += 1
            mutant_vectors = np.empty_like(population)

            # Generate mutant vectors via vector difference
            for i in range(pop_size):
                candidates = [idx for idx in range(pop_size) if idx != i]
                r = rng.choice(candidates, size=3, replace=False)
                r1, r2, r3 = r[0], r[1], r[2]

                if strategy == "best1bin":
                    base = best_position
                    diff = population[r1] - population[r2]
                else:  # rand1bin
                    base = population[r1]
                    diff = population[r2] - population[r3]

                mutant_vectors[i] = base + F * diff

            # Binomial Crossover
            trial_vectors = population.copy()
            crossover_mask = rng.random(size=(pop_size, dim)) < CR
            # Ensure at least one component from mutant vector per individual
            j_rand = rng.integers(0, dim, size=pop_size)
            for i in range(pop_size):
                crossover_mask[i, j_rand[i]] = True

            trial_vectors[crossover_mask] = mutant_vectors[crossover_mask]

            # Boundary clamping
            trial_vectors = np.clip(trial_vectors, low, high)

            # Evaluate trial vectors
            remaining = problem.remaining_evaluations
            eval_size = min(pop_size, remaining)
            trial_fitnesses = problem.evaluate(trial_vectors[:eval_size])

            # Selection: Greedy replacement
            for i in range(eval_size):
                if trial_fitnesses[i] <= fitnesses[i]:
                    population[i] = trial_vectors[i]
                    fitnesses[i] = trial_fitnesses[i]
                    if trial_fitnesses[i] < best_fitness:
                        best_fitness = float(trial_fitnesses[i])
                        best_position = trial_vectors[i].copy()

            step = float(np.mean(np.linalg.norm(trial_vectors[:eval_size] - population[:eval_size], axis=1)))

            tracker.record_step(
                iteration=iteration,
                evaluations=problem.evaluations_count,
                population=population,
                fitnesses=fitnesses,
                best_fitness=best_fitness,
                best_position=best_position,
                step_size=step
            )

        elapsed = time.perf_counter() - start_time
        err = abs(best_fitness - problem.benchmark.global_minimum_val)

        return OptimizationResult(
            algorithm_name="DE",
            benchmark_name=problem.benchmark.name,
            dimension=dim,
            seed=seed if seed is not None else -1,
            best_position=best_position,
            best_fitness=best_fitness,
            error_to_optimum=float(err),
            total_evaluations=problem.evaluations_count,
            iterations=iteration,
            execution_time_sec=elapsed,
            hyperparameters=dict(self.config),
            telemetry=tracker
        )


# -----------------------------------------------------------------------------
# 4. Simulated Annealing (SA)
# -----------------------------------------------------------------------------
class SimulatedAnnealing(BaseMetaheuristic):
    """
    Continuous Simulated Annealing with adaptive step scaling,
    thermal annealing schedule, and reheat mechanism.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = get_default_config("SA")
        if config:
            cfg.update(config)
        super().__init__("SA", cfg)

    def optimize(
        self,
        problem: OptimizationProblem,
        seed: Optional[int] = None,
        tracker: Optional[TrajectoryTracker] = None
    ) -> OptimizationResult:
        rng = np.random.default_rng(seed)
        start_time = time.perf_counter()

        initial_temp = float(self.config.get("initial_temp", 100.0))
        cooling_rate = float(self.config.get("cooling_rate", 0.985))
        base_step_size = float(self.config.get("step_size", 0.08))
        schedule = str(self.config.get("cooling_schedule", "geometric")).lower()
        restart_stagnation = int(self.config.get("restart_stagnation", 1500))

        dim = problem.dimension
        low, high = problem.lower_bounds, problem.upper_bounds
        bound_span = high - low

        if tracker is None:
            tracker = TrajectoryTracker()

        # Initial point
        current_pos = rng.uniform(low, high, size=dim)
        current_fit = float(problem.evaluate(current_pos))

        best_pos = current_pos.copy()
        best_fit = current_fit
        temperature = initial_temp

        iteration = 0
        stagnation_counter = 0

        tracker.record_step(
            iteration=iteration,
            evaluations=problem.evaluations_count,
            population=current_pos.reshape(1, -1),
            fitnesses=np.array([current_fit]),
            best_fitness=best_fit,
            best_position=best_pos,
            step_size=0.0
        )

        while problem.remaining_evaluations > 0:
            iteration += 1

            # Dynamic step size proportional to temperature
            step_factor = base_step_size * max(0.01, (temperature / initial_temp) ** 0.5)
            perturbation = rng.normal(0.0, step_factor * bound_span, size=dim)
            candidate_pos = np.clip(current_pos + perturbation, low, high)

            candidate_fit = float(problem.evaluate(candidate_pos))
            delta = candidate_fit - current_fit

            # Metropolis acceptance criterion
            accept = False
            if delta < 0:
                accept = True
            else:
                prob = np.exp(-delta / max(1e-12, temperature))
                if rng.random() < prob:
                    accept = True

            if accept:
                current_pos = candidate_pos
                current_fit = candidate_fit

            # Check for new global best
            if current_fit < best_fit:
                best_fit = current_fit
                best_pos = current_pos.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Thermal Reheat / Restart if stagnated
            if stagnation_counter >= restart_stagnation:
                temperature = initial_temp * 0.5
                current_pos = rng.uniform(low, high, size=dim)
                current_fit = float(problem.evaluate(current_pos))
                stagnation_counter = 0

            # Cooling schedule
            if schedule == "linear":
                temperature = max(1e-6, initial_temp * (1.0 - iteration / problem.max_evaluations))
            elif schedule == "logarithmic":
                temperature = initial_temp / np.log(iteration + 2.0)
            else:  # geometric
                temperature = max(1e-8, temperature * cooling_rate)

            tracker.record_step(
                iteration=iteration,
                evaluations=problem.evaluations_count,
                population=current_pos.reshape(1, -1),
                fitnesses=np.array([current_fit]),
                best_fitness=best_fit,
                best_position=best_pos,
                step_size=float(np.linalg.norm(perturbation))
            )

        elapsed = time.perf_counter() - start_time
        err = abs(best_fit - problem.benchmark.global_minimum_val)

        return OptimizationResult(
            algorithm_name="SA",
            benchmark_name=problem.benchmark.name,
            dimension=dim,
            seed=seed if seed is not None else -1,
            best_position=best_pos,
            best_fitness=best_fit,
            error_to_optimum=float(err),
            total_evaluations=problem.evaluations_count,
            iterations=iteration,
            execution_time_sec=elapsed,
            hyperparameters=dict(self.config),
            telemetry=tracker
        )


def create_algorithm(name: str, config: Optional[Dict[str, Any]] = None) -> BaseMetaheuristic:
    """Factory function for instantiating metaheuristic algorithms."""
    algo = name.strip().upper()
    if algo == "PSO":
        return ParticleSwarmOptimization(config)
    elif algo == "GA":
        return GeneticAlgorithm(config)
    elif algo == "DE":
        return DifferentialEvolution(config)
    elif algo == "SA":
        return SimulatedAnnealing(config)
    else:
        raise ValueError(f"Unknown algorithm: {name}. Supported: PSO, GA, DE, SA")

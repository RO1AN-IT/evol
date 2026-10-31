"""Генетический алгоритм для целочисленного меню."""

import time
from dataclasses import dataclass

import numpy as np

from src.operators import swap_mutation, tournament_select, uniform_crossover, reset_mutation
from src.problem import Limits, is_feasible, n_days, n_meals, n_options, penalized_cost, total_cost
from src.repair import repair


@dataclass
class GaConfig:
    name: str
    display_name: str
    constraint_mode: str
    mutation: str
    population_size: int
    tournament_size: int
    crossover_probability: float
    reset_probability: float
    swaps_per_meal: int
    elite_count: int


@dataclass
class RunResult:
    method: str
    display_name: str
    seed: int
    best_genome: np.ndarray | None
    best_feasible_cost: float | None
    best_fitness: float
    best_fitness_genome: np.ndarray
    feasible: bool
    evaluations: int
    generations: int
    runtime_sec: float
    checkpoint_evals: np.ndarray
    checkpoint_cost: np.ndarray


def run_ga(
    config: GaConfig,
    limits: Limits,
    budget: int,
    seed: int,
) -> RunResult:
    if config.population_size < 30:
        raise ValueError("размер популяции должен быть не меньше 30")
    if config.constraint_mode not in {"repair", "penalty"}:
        raise ValueError("режим ограничений: repair или penalty")
    if config.mutation not in {"reset", "swap"}:
        raise ValueError("мутация: reset или swap")
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    population = rng.integers(0, n_options(), size=(config.population_size, n_days(), n_meals()))
    fitness, feasible_flags = _evaluate_population(population, config, limits, rng)
    evaluations = config.population_size
    best_cost, best_genome = _best_feasible(population, feasible_flags)
    checkpoint_evals = [evaluations]
    checkpoint_cost = [best_cost]
    generations = 0

    while evaluations < budget:
        offspring_count = min(config.population_size - config.elite_count, budget - evaluations)
        elite_index = np.argsort(fitness)[: config.elite_count]
        elites = population[elite_index].copy()
        elite_fitness = fitness[elite_index].copy()
        children = []
        while len(children) < offspring_count:
            parent_a = tournament_select(population, fitness, config.tournament_size, rng)
            parent_b = tournament_select(population, fitness, config.tournament_size, rng)
            if rng.random() < config.crossover_probability:
                child_a, child_b = uniform_crossover(parent_a, parent_b, rng)
            else:
                child_a, child_b = parent_a.copy(), parent_b.copy()
            child_a = _mutate(child_a, config, rng)
            child_b = _mutate(child_b, config, rng)
            children.append(child_a)
            if len(children) < offspring_count:
                children.append(child_b)

        offspring = np.stack(children)
        offspring_fitness, _offspring_feasible = _evaluate_population(offspring, config, limits, rng)
        evaluations += offspring.shape[0]
        generations += 1
        population = np.concatenate([elites, offspring])
        fitness = np.concatenate([elite_fitness, offspring_fitness])
        feasible_flags = np.array([is_feasible(individual, limits) for individual in population])
        candidate_cost, candidate_genome = _best_feasible(population, feasible_flags)
        if candidate_genome is not None and (best_cost is None or candidate_cost < best_cost):
            best_cost = candidate_cost
            best_genome = candidate_genome
        checkpoint_evals.append(evaluations)
        checkpoint_cost.append(np.nan if best_cost is None else best_cost)

    winner = int(np.argmin(fitness))
    return RunResult(
        method=config.name,
        display_name=config.display_name,
        seed=seed,
        best_genome=None if best_genome is None else best_genome.copy(),
        best_feasible_cost=None if best_cost is None else float(best_cost),
        best_fitness=float(fitness[winner]),
        best_fitness_genome=population[winner].copy(),
        feasible=best_genome is not None,
        evaluations=evaluations,
        generations=generations,
        runtime_sec=time.perf_counter() - started,
        checkpoint_evals=np.asarray(checkpoint_evals, dtype=int),
        checkpoint_cost=np.asarray(checkpoint_cost, dtype=float),
    )


def _mutate(genome: np.ndarray, config: GaConfig, rng: np.random.Generator) -> np.ndarray:
    if config.mutation == "reset":
        return reset_mutation(genome, config.reset_probability, rng)
    return swap_mutation(genome, config.swaps_per_meal, rng)


def _evaluate_population(population, config: GaConfig, limits: Limits, rng: np.random.Generator):
    fitness = np.empty(population.shape[0], dtype=float)
    feasible = np.empty(population.shape[0], dtype=bool)
    for index, individual in enumerate(population):
        if config.constraint_mode == "repair":
            individual = repair(individual, rng, limits)
            population[index] = individual
        fitness[index] = penalized_cost(individual, limits)
        feasible[index] = is_feasible(individual, limits)
    return fitness, feasible


def _best_feasible(population: np.ndarray, feasible: np.ndarray) -> tuple[float | None, np.ndarray | None]:
    if not np.any(feasible):
        return None, None
    costs = np.array(
        [total_cost(individual) if flag else np.inf for individual, flag in zip(population, feasible)]
    )
    winner = int(np.argmin(costs))
    return float(costs[winner]), population[winner].copy()

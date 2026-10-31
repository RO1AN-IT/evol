"""Жадная сборка меню и случайный поиск с тем же ремонтом."""

import time

import numpy as np

from src.catalog import CATALOG
from src.ga import RunResult
from src.problem import Limits, is_feasible, n_days, n_meals, n_options, penalized_cost, total_cost
from src.repair import repair


def greedy_menu(rng: np.random.Generator, limits: Limits) -> np.ndarray:
    """Самые дешёвые блюда с покрытием всех категорий, затем ремонт календаря."""
    genome = np.zeros((n_days(), n_meals()), dtype=int)
    for meal, dishes in enumerate(CATALOG):
        chosen: list[int] = []
        by_category: dict[str, list] = {}
        for dish in dishes:
            by_category.setdefault(dish.category, []).append(dish)
        for group in by_category.values():
            cheapest = min(group, key=lambda dish: (dish.cost, dish.index))
            chosen.append(cheapest.index)
        for dish in sorted(dishes, key=lambda item: (item.cost, item.index)):
            if len(chosen) == n_days():
                break
            if dish.index not in chosen:
                chosen.append(dish.index)
        for day, dish_index in zip(rng.permutation(n_days()), chosen):
            genome[int(day), meal] = dish_index
    return repair(genome, rng, limits)


def run_greedy(limits: Limits, seed: int) -> RunResult:
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    genome = greedy_menu(rng, limits)
    feasible = is_feasible(genome, limits)
    cost = float(total_cost(genome)) if feasible else None
    return RunResult(
        method="greedy",
        display_name="Жадная эвристика",
        seed=seed,
        best_genome=genome,
        best_feasible_cost=cost,
        best_fitness=float(penalized_cost(genome, limits)),
        best_fitness_genome=genome.copy(),
        feasible=feasible,
        evaluations=1,
        generations=0,
        runtime_sec=time.perf_counter() - started,
        checkpoint_evals=np.asarray([1], dtype=int),
        checkpoint_cost=np.asarray([np.nan if cost is None else cost], dtype=float),
    )


def run_random_repaired(
    limits: Limits,
    budget: int,
    seed: int,
    checkpoints: np.ndarray,
) -> RunResult:
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    best_cost = None
    best_genome = None
    best_fitness = None
    best_fitness_genome = None
    history = np.full(budget, np.nan)
    for evaluation in range(budget):
        genome = rng.integers(0, n_options(), size=(n_days(), n_meals()))
        genome = repair(genome, rng, limits)
        fitness = penalized_cost(genome, limits)
        if best_fitness is None or fitness < best_fitness:
            best_fitness = fitness
            best_fitness_genome = genome.copy()
        if is_feasible(genome, limits):
            cost = total_cost(genome)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_genome = genome.copy()
        history[evaluation] = np.nan if best_cost is None else best_cost
    indexes = np.clip(checkpoints.astype(int) - 1, 0, budget - 1)
    return RunResult(
        method="random_repair",
        display_name="Случайный поиск с ремонтом",
        seed=seed,
        best_genome=best_genome,
        best_feasible_cost=None if best_cost is None else float(best_cost),
        best_fitness=float(best_fitness),
        best_fitness_genome=best_fitness_genome,
        feasible=best_genome is not None,
        evaluations=budget,
        generations=0,
        runtime_sec=time.perf_counter() - started,
        checkpoint_evals=checkpoints.astype(int).copy(),
        checkpoint_cost=history[indexes],
    )

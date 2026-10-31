"""Операторы целочисленного вектора меню."""

import numpy as np

from src.problem import n_days, n_meals, n_options


def tournament_select(
    population: np.ndarray,
    fitness: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    contenders = rng.integers(0, population.shape[0], size=tournament_size)
    winner = contenders[np.argmin(fitness[contenders])]
    return population[winner].copy()


def uniform_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Каждая позиция дня и приёма пищи независимо берётся у одного из родителей."""
    mask = rng.random(parent_a.shape) < 0.5
    child_a = np.where(mask, parent_a, parent_b).astype(int)
    child_b = np.where(mask, parent_b, parent_a).astype(int)
    return child_a, child_b


def reset_mutation(genome: np.ndarray, probability: float, rng: np.random.Generator) -> np.ndarray:
    """Замена выбранных позиций на случайное блюдо того же приёма пищи."""
    mutant = genome.copy()
    mask = rng.random(mutant.shape) < probability
    random_dishes = rng.integers(0, n_options(), size=mutant.shape)
    mutant = np.where(mask, random_dishes, mutant).astype(int)
    return mutant


def swap_mutation(genome: np.ndarray, swaps_per_meal: int, rng: np.random.Generator) -> np.ndarray:
    """Обмен двух дней внутри одного приёма пищи.

    Набор блюд завтрака, обеда и ужина не меняется, меняется только календарь.
    Поэтому число повторов и покрытие категорий сохраняются.
    """
    mutant = genome.copy()
    for meal in range(n_meals()):
        for _ in range(swaps_per_meal):
            first, second = rng.choice(n_days(), size=2, replace=False)
            mutant[first, meal], mutant[second, meal] = mutant[second, meal], mutant[first, meal]
    return mutant

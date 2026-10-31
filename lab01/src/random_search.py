"""Базовая случайная выборка с тем же бюджетом вычислений функции."""


import time
from dataclasses import dataclass

import numpy as np

from src.problem import hgbat_population


@dataclass
class RandomSearchResult:
    method: str
    display_name: str
    seed: int
    best_x: np.ndarray
    best_f: float
    evaluations: int
    generations: int
    runtime_sec: float
    checkpoint_evals: np.ndarray
    checkpoint_best: np.ndarray


def run_random_search(
    dimension: int,
    lower: float,
    upper: float,
    budget: int,
    seed: int,
    checkpoints: np.ndarray,
) -> RandomSearchResult:
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    samples = rng.uniform(lower, upper, size=(budget, dimension))
    fitness = hgbat_population(samples)
    prefix_best = np.minimum.accumulate(fitness)
    best_index = int(np.argmin(fitness))

    # checkpoints задаёт генетический алгоритм (после начальной популяции
    # и после каждого поколения). Берём лучшее среди первых E точек.
    indexes = np.clip(checkpoints.astype(int) - 1, 0, budget - 1)
    return RandomSearchResult(
        method="random_search",
        display_name="Случайный поиск",
        seed=seed,
        best_x=samples[best_index].copy(),
        best_f=float(fitness[best_index]),
        evaluations=budget,
        generations=0,
        runtime_sec=time.perf_counter() - started,
        checkpoint_evals=checkpoints.astype(int).copy(),
        checkpoint_best=prefix_best[indexes],
    )

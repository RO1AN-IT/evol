"""Генетический алгоритм с вещественным кодированием."""


import time
from dataclasses import dataclass

import numpy as np

from src.operators import gaussian_mutation, sbx_crossover, tournament_select
from src.problem import hgbat_population


@dataclass
class GaConfig:
    name: str
    display_name: str
    population_size: int
    tournament_size: int
    crossover_probability: float
    sbx_eta: float
    sbx_gene_probability: float
    mutation_probability: float
    mutation_sigma: float
    elite_count: int


@dataclass
class RunResult:
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


def _validate(config: GaConfig, budget: int) -> None:
    if config.population_size < 30:
        raise ValueError("размер популяции должен быть не меньше 30")
    if config.elite_count < 1:
        raise ValueError("элитизм требует сохранить хотя бы одну особь")
    if config.elite_count >= config.population_size:
        raise ValueError("число элитных особей должно быть меньше размера популяции")
    if budget < config.population_size:
        raise ValueError("бюджет вычислений меньше начальной популяции")
    if not 0.0 <= config.crossover_probability <= 1.0:
        raise ValueError("вероятность кроссовера вне [0, 1]")
    if not 0.0 <= config.mutation_probability <= 1.0:
        raise ValueError("вероятность мутации вне [0, 1]")


def run_ga(
    config: GaConfig,
    dimension: int,
    lower: float,
    upper: float,
    budget: int,
    seed: int,
) -> RunResult:
    """Один запуск. Критерий остановки — число вычислений функции."""
    _validate(config, budget)
    rng = np.random.default_rng(seed)
    lo = np.full(dimension, lower)
    hi = np.full(dimension, upper)
    started = time.perf_counter()

    population = rng.uniform(lower, upper, size=(config.population_size, dimension))
    fitness = hgbat_population(population)
    evaluations = config.population_size
    best_index = int(np.argmin(fitness))
    best_f = float(fitness[best_index])
    best_x = population[best_index].copy()

    checkpoint_evals = [evaluations]
    checkpoint_best = [best_f]
    generations = 0

    while evaluations < budget:
        offspring_count = min(config.population_size - config.elite_count, budget - evaluations)
        elite_index = np.argsort(fitness)[: config.elite_count]
        elites = population[elite_index].copy()
        elite_fitness = fitness[elite_index].copy()

        children = []
        while len(children) < offspring_count:
            parent_a = tournament_select(
                population, fitness, config.tournament_size, rng
            )
            parent_b = tournament_select(
                population, fitness, config.tournament_size, rng
            )
            if rng.random() < config.crossover_probability:
                child_a, child_b = sbx_crossover(
                    parent_a,
                    parent_b,
                    lo,
                    hi,
                    config.sbx_eta,
                    config.sbx_gene_probability,
                    rng,
                )
            else:
                child_a, child_b = parent_a.copy(), parent_b.copy()
            child_a = gaussian_mutation(
                child_a, lo, hi, config.mutation_probability, config.mutation_sigma, rng
            )
            child_b = gaussian_mutation(
                child_b, lo, hi, config.mutation_probability, config.mutation_sigma, rng
            )
            children.append(child_a)
            if len(children) < offspring_count:
                children.append(child_b)

        offspring = np.vstack(children)
        offspring_fitness = hgbat_population(offspring)
        evaluations += offspring.shape[0]
        generations += 1

        population = np.vstack([elites, offspring])
        fitness = np.concatenate([elite_fitness, offspring_fitness])
        current_index = int(np.argmin(fitness))
        current_f = float(fitness[current_index])
        if current_f < best_f:
            best_f = current_f
            best_x = population[current_index].copy()

        checkpoint_evals.append(evaluations)
        checkpoint_best.append(best_f)

    if any(
        later > earlier + 1e-12
        for earlier, later in zip(checkpoint_best, checkpoint_best[1:])
    ):
        raise AssertionError("элитизм нарушен: лучшее значение выросло")

    return RunResult(
        method=config.name,
        display_name=config.display_name,
        seed=seed,
        best_x=best_x,
        best_f=best_f,
        evaluations=evaluations,
        generations=generations,
        runtime_sec=time.perf_counter() - started,
        checkpoint_evals=np.asarray(checkpoint_evals, dtype=int),
        checkpoint_best=np.asarray(checkpoint_best, dtype=float),
    )

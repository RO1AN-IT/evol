"""Операторы вещественного генетического алгоритма."""


import numpy as np


def tournament_select(
    population: np.ndarray,
    fitness: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Турнир размера k. Побеждает меньшее значение целевой функции."""
    contenders = rng.integers(0, population.shape[0], size=tournament_size)
    winner = contenders[np.argmin(fitness[contenders])]
    return population[winner].copy()


def _sbx_from_ordered_parents(
    y1: float,
    y2: float,
    lower: float,
    upper: float,
    eta: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """SBX для одной координаты. На входе y1 <= y2."""
    if abs(y1 - y2) < 1e-14:
        return y1, y2

    eta_exp = 1.0 / (eta + 1.0)
    span = y2 - y1

    rand = rng.random()
    beta = 1.0 + (2.0 * (y1 - lower) / span)
    alpha = 2.0 - beta ** (-(eta + 1.0))
    if rand <= 1.0 / alpha:
        betaq = (rand * alpha) ** eta_exp
    else:
        betaq = (1.0 / max(2.0 - rand * alpha, 1e-14)) ** eta_exp
    child_low = 0.5 * ((y1 + y2) - betaq * span)

    rand = rng.random()
    beta = 1.0 + (2.0 * (upper - y2) / span)
    alpha = 2.0 - beta ** (-(eta + 1.0))
    if rand <= 1.0 / alpha:
        betaq = (rand * alpha) ** eta_exp
    else:
        betaq = (1.0 / max(2.0 - rand * alpha, 1e-14)) ** eta_exp
    child_high = 0.5 * ((y1 + y2) + betaq * span)

    child_low = min(max(child_low, lower), upper)
    child_high = min(max(child_high, lower), upper)
    return child_low, child_high


def sbx_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eta: float,
    gene_probability: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Имитация двоичного кроссовера (SBX) с учётом границ.

    Каждая координата скрещивается с вероятностью gene_probability.
    Остальные координаты копируются от родителей.
    """
    child_a = parent_a.copy()
    child_b = parent_b.copy()
    for index in range(parent_a.size):
        if rng.random() > gene_probability:
            continue
        y1 = float(parent_a[index])
        y2 = float(parent_b[index])
        if y1 > y2:
            y1, y2 = y2, y1
        low, high = _sbx_from_ordered_parents(
            y1, y2, float(lower[index]), float(upper[index]), eta, rng
        )
        if rng.random() < 0.5:
            child_a[index], child_b[index] = low, high
        else:
            child_a[index], child_b[index] = high, low
    return child_a, child_b


def gaussian_mutation(
    individual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    probability: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Независимая гауссовская мутация координат и проекция на область поиска."""
    mutant = individual.copy()
    mask = rng.random(mutant.size) < probability
    if np.any(mask):
        mutant[mask] += rng.normal(0.0, sigma, size=int(np.sum(mask)))
    return np.clip(mutant, lower, upper)


def assert_operators_respect_bounds(dimension: int, lower: float, upper: float) -> None:
    rng = np.random.default_rng(0)
    lo = np.full(dimension, lower)
    hi = np.full(dimension, upper)
    for _ in range(200):
        parents = rng.uniform(lower, upper, size=(2, dimension))
        child_a, child_b = sbx_crossover(parents[0], parents[1], lo, hi, 15.0, 0.5, rng)
        child_a = gaussian_mutation(child_a, lo, hi, 0.5, 2.0, rng)
        child_b = gaussian_mutation(child_b, lo, hi, 0.5, 2.0, rng)
        if np.any(child_a < lower - 1e-12) or np.any(child_a > upper + 1e-12):
            raise AssertionError("потомок A вышел за границы")
        if np.any(child_b < lower - 1e-12) or np.any(child_b > upper + 1e-12):
            raise AssertionError("потомок B вышел за границы")

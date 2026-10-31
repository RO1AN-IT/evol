"""Целевая функция HGBat (вариант 20)."""

import numpy as np

from src.variant import KNOWN_MINIMUM, LOWER, OPTIMUM_COORDINATE, UPPER


def hgbat(x: np.ndarray) -> float:
    """Значение HGBat в одной точке.

    f(x) = | (Σ x_i²)² − (Σ x_i)² |^(1/2)
           + (0.5 Σ x_i² + Σ x_i) / d + 0.5
    """
    point = np.asarray(x, dtype=float)
    return float(hgbat_population(point.reshape(1, -1))[0])


def hgbat_population(population: np.ndarray) -> np.ndarray:
    """Значения HGBat для набора строк (n, d)."""
    points = np.asarray(population, dtype=float)
    sum_sq = np.sum(points * points, axis=1)
    sum_lin = np.sum(points, axis=1)
    dimension = points.shape[1]
    coupled = np.sqrt(np.abs(sum_sq * sum_sq - sum_lin * sum_lin))
    smooth = (0.5 * sum_sq + sum_lin) / dimension + 0.5
    return coupled + smooth


def optimum_point(dimension: int) -> np.ndarray:
    return np.full(dimension, OPTIMUM_COORDINATE, dtype=float)


def assert_problem(dimension: int) -> None:
    """Проверка аналитического минимума и пары контрольных точек."""
    optimum = optimum_point(dimension)
    value = hgbat(optimum)
    if abs(value - KNOWN_MINIMUM) > 1e-10:
        raise AssertionError(f"f(-1) = {value}, ожидался 0")

    zeros = hgbat(np.zeros(dimension))
    # f(0) = 0.5 для любой размерности: оба слагаемых дают 0 и 0.5.
    if abs(zeros - 0.5) > 1e-10:
        raise AssertionError(f"f(0) = {zeros}, ожидалось 0.5")

    outside = hgbat(np.full(dimension, UPPER))
    if outside <= value:
        raise AssertionError("значение на границе не больше минимума")

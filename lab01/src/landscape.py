"""Проверка, почему локальный поиск не доходит до глобального минимума HGBat."""


import json
from pathlib import Path

import numpy as np

from src.problem import hgbat, optimum_point


def _coordinate_descent(start: np.ndarray, lower: float, upper: float) -> tuple[float, np.ndarray, int]:
    point = np.clip(start.astype(float), lower, upper)
    value = hgbat(point)
    evaluations = 1
    step = 1.0
    while step >= 1e-3:
        improved = True
        while improved:
            improved = False
            for index in range(point.size):
                for direction in (step, -step):
                    candidate = point.copy()
                    candidate[index] = float(np.clip(candidate[index] + direction, lower, upper))
                    candidate_value = hgbat(candidate)
                    evaluations += 1
                    if candidate_value < value:
                        point, value = candidate, candidate_value
                        improved = True
        step *= 0.5
    return value, point, evaluations


def _gradient_descent(start: np.ndarray, lower: float, upper: float) -> tuple[float, int]:
    point = np.clip(start.astype(float), lower, upper)
    value = hgbat(point)
    evaluations = 1
    step = 0.05
    for _ in range(1500):
        gradient = np.zeros_like(point)
        for index in range(point.size):
            shifted = point.copy()
            shifted[index] += 1e-5
            gradient[index] = (hgbat(shifted) - value) / 1e-5
            evaluations += 1
        norm = float(np.linalg.norm(gradient))
        if norm < 1e-8:
            break
        trial = np.clip(point - step * gradient, lower, upper)
        trial_value = hgbat(trial)
        evaluations += 1
        if trial_value < value:
            point, value = trial, trial_value
        else:
            step *= 0.5
            if step < 1e-8:
                break
    return value, evaluations


def probe_landscape(dimension: int, lower: float, upper: float, path: Path) -> dict:
    """Серия локальных спусков и срез функции около известного минимума."""
    rng = np.random.default_rng(20261028)
    coordinate_runs = []
    for _ in range(15):
        start = rng.uniform(lower, upper, size=dimension)
        value, point, evaluations = _coordinate_descent(start, lower, upper)
        coordinate_runs.append(
            {
                "fitness": value,
                "distance": float(np.linalg.norm(point - optimum_point(dimension))),
                "evaluations": evaluations,
            }
        )
    gradient_runs = []
    for _ in range(12):
        start = rng.uniform(lower, upper, size=dimension)
        value, evaluations = _gradient_descent(start, lower, upper)
        gradient_runs.append({"fitness": value, "evaluations": evaluations})

    optimum = optimum_point(dimension)
    spheres = []
    for radius in (0.05, 0.2, 0.5, 1.0):
        samples = []
        for _ in range(40):
            direction = rng.normal(size=dimension)
            direction /= np.linalg.norm(direction)
            samples.append(hgbat(optimum + radius * direction))
        array = np.asarray(samples)
        spheres.append(
            {
                "radius": radius,
                "min": float(array.min()),
                "median": float(np.median(array)),
                "max": float(array.max()),
            }
        )

    coordinate_values = np.asarray([item["fitness"] for item in coordinate_runs])
    gradient_values = np.asarray([item["fitness"] for item in gradient_runs])
    report = {
        "coordinate_descent": {
            "starts": len(coordinate_runs),
            "best": float(coordinate_values.min()),
            "mean": float(coordinate_values.mean()),
            "worst": float(coordinate_values.max()),
        },
        "gradient_descent": {
            "starts": len(gradient_runs),
            "best": float(gradient_values.min()),
            "mean": float(gradient_values.mean()),
            "worst": float(gradient_values.max()),
        },
        "sphere_around_optimum": spheres,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report

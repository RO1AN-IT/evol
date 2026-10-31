"""Восстановление допустимости меню."""

import numpy as np

from src.catalog import CATALOG, categories_of
from src.problem import Limits, calorie_gap, day_kcal, n_days, n_meals, n_options, total_cost


def repair(genome: np.ndarray, rng: np.random.Generator, limits: Limits) -> np.ndarray:
    repaired = np.array(genome, dtype=int, copy=True)
    _make_unique(repaired, rng)
    _cover_categories(repaired, rng)
    for _ in range(3):
        if not _reduce_calorie_gaps(repaired, limits):
            break
    _cover_categories(repaired, rng)
    for _ in range(8):
        if not _reduce_budget(repaired, limits):
            break
    return repaired


def _used(genome: np.ndarray, meal: int) -> set[int]:
    return {int(value) for value in genome[:, meal]}


def _make_unique(genome: np.ndarray, rng: np.random.Generator) -> None:
    for meal in range(n_meals()):
        seen: set[int] = set()
        order = rng.permutation(n_days())
        for day in order:
            gene = int(genome[day, meal])
            if gene not in seen:
                seen.add(gene)
                continue
            choices = [index for index in range(n_options()) if index not in seen]
            if not choices:
                continue
            genome[day, meal] = int(rng.choice(choices))
            seen.add(int(genome[day, meal]))


def _cover_categories(genome: np.ndarray, rng: np.random.Generator) -> None:
    for meal in range(n_meals()):
        required = list(categories_of(meal))
        for _ in range(len(required)):
            present = _category_counts(genome, meal)
            missing = [category for category in required if present.get(category, 0) == 0]
            if not missing:
                break
            surplus_days = [
                day
                for day in range(n_days())
                if present[CATALOG[meal][int(genome[day, meal])].category] >= 2
            ]
            if not surplus_days:
                break
            day = int(rng.choice(surplus_days))
            target = str(rng.choice(missing))
            choices = [
                index
                for index in range(n_options())
                if index not in _used(genome, meal)
                and CATALOG[meal][index].category == target
            ]
            if not choices:
                break
            genome[day, meal] = int(rng.choice(choices))


def _category_counts(genome: np.ndarray, meal: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for day in range(n_days()):
        category = CATALOG[meal][int(genome[day, meal])].category
        counts[category] = counts.get(category, 0) + 1
    return counts


def _categories_survive(genome: np.ndarray, day: int, meal: int, alternative: int) -> bool:
    previous = int(genome[day, meal])
    genome[day, meal] = alternative
    survived = all(
        _category_counts(genome, meal).get(category, 0) > 0 for category in categories_of(meal)
    )
    genome[day, meal] = previous
    return survived


def _reduce_calorie_gaps(genome: np.ndarray, limits: Limits) -> bool:
    changed = False
    for day in range(n_days()):
        current_gap = calorie_gap(day_kcal(genome, day), limits)
        if current_gap == 0:
            continue
        best: tuple[int, int, int] | None = None
        for meal in range(n_meals()):
            current = int(genome[day, meal])
            for alternative in range(n_options()):
                if alternative == current or alternative in _used(genome, meal):
                    continue
                if not _categories_survive(genome, day, meal, alternative):
                    continue
                genome[day, meal] = alternative
                gap = calorie_gap(day_kcal(genome, day), limits)
                genome[day, meal] = current
                if gap < current_gap and (best is None or gap < best[0]):
                    best = (gap, meal, alternative)
        if best is not None:
            genome[day, best[1]] = best[2]
            changed = True
    return changed


def _reduce_budget(genome: np.ndarray, limits: Limits) -> bool:
    if total_cost(genome) <= limits.budget:
        return False
    best: tuple[int, int, int, int] | None = None
    for day in range(n_days()):
        for meal in range(n_meals()):
            current = int(genome[day, meal])
            current_cost = CATALOG[meal][current].cost
            old_gap = calorie_gap(day_kcal(genome, day), limits)
            for alternative in range(n_options()):
                if alternative == current or alternative in _used(genome, meal):
                    continue
                saving = current_cost - CATALOG[meal][alternative].cost
                if saving <= 0:
                    continue
                if not _categories_survive(genome, day, meal, alternative):
                    continue
                genome[day, meal] = alternative
                new_gap = calorie_gap(day_kcal(genome, day), limits)
                genome[day, meal] = current
                if new_gap > old_gap:
                    continue
                if best is None or saving > best[0]:
                    best = (saving, day, meal, alternative)
    if best is None:
        return False
    genome[best[1], best[2]] = best[3]
    return True

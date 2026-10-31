"""Декодирование меню, стоимость и нарушения ограничений."""

from dataclasses import dataclass

import numpy as np

from src.catalog import CATALOG, DAY_NAMES, MEALS, categories_of


@dataclass(frozen=True)
class Limits:
    calorie_min: int
    calorie_max: int
    budget: int
    penalty_per_kcal: float
    penalty_per_repeat: float
    penalty_per_missing_category: float
    penalty_per_ruble_over: float


def n_days() -> int:
    return 7


def n_meals() -> int:
    return len(MEALS)


def n_options() -> int:
    return len(CATALOG[0])


def empty_genome() -> np.ndarray:
    return np.zeros((n_days(), n_meals()), dtype=int)


def day_kcal(genome: np.ndarray, day: int) -> int:
    return sum(CATALOG[meal][int(genome[day, meal])].kcal for meal in range(n_meals()))


def total_cost(genome: np.ndarray) -> int:
    return sum(
        CATALOG[meal][int(genome[day, meal])].cost
        for day in range(n_days())
        for meal in range(n_meals())
    )


def calorie_gap(total: int, limits: Limits) -> int:
    return max(0, limits.calorie_min - total) + max(0, total - limits.calorie_max)


def repeats_of(genome: np.ndarray, meal: int) -> int:
    column = [int(value) for value in genome[:, meal]]
    return len(column) - len(set(column))


def missing_categories(genome: np.ndarray, meal: int) -> int:
    present = {CATALOG[meal][int(genome[day, meal])].category for day in range(n_days())}
    return len(set(categories_of(meal)) - present)


def violation_details(genome: np.ndarray, limits: Limits) -> dict:
    day_gaps = [calorie_gap(day_kcal(genome, day), limits) for day in range(n_days())]
    repeats = [repeats_of(genome, meal) for meal in range(n_meals())]
    missing = [missing_categories(genome, meal) for meal in range(n_meals())]
    cost = total_cost(genome)
    over_budget = max(0, cost - limits.budget)
    return {
        "cost": cost,
        "kcal_outside": int(sum(day_gaps)),
        "repeats": int(sum(repeats)),
        "missing_categories": int(sum(missing)),
        "rubles_over_budget": int(over_budget),
        "day_gaps": day_gaps,
        "repeats_by_meal": repeats,
        "missing_by_meal": missing,
    }


def is_feasible(genome: np.ndarray, limits: Limits) -> bool:
    details = violation_details(genome, limits)
    return (
        details["kcal_outside"] == 0
        and details["repeats"] == 0
        and details["missing_categories"] == 0
        and details["rubles_over_budget"] == 0
    )


def penalized_cost(genome: np.ndarray, limits: Limits) -> float:
    details = violation_details(genome, limits)
    return float(
        details["cost"]
        + limits.penalty_per_kcal * details["kcal_outside"]
        + limits.penalty_per_repeat * details["repeats"]
        + limits.penalty_per_missing_category * details["missing_categories"]
        + limits.penalty_per_ruble_over * details["rubles_over_budget"]
    )


def describe_violations(genome: np.ndarray, limits: Limits) -> list[str]:
    details = violation_details(genome, limits)
    lines = []
    if details["repeats"]:
        lines.append(f"повторы блюд внутри приёма пищи: {details['repeats']}")
    if details["missing_categories"]:
        lines.append(f"не хватает категорий: {details['missing_categories']}")
    if details["kcal_outside"]:
        lines.append(f"выход калорий за сутки из диапазона, суммарно {details['kcal_outside']} ккал")
    if details["rubles_over_budget"]:
        lines.append(f"превышение бюджета на {details['rubles_over_budget']} руб.")
    if not lines:
        lines.append("ограничения выполнены")
    return lines


def format_menu(genome: np.ndarray) -> str:
    lines = ["День | Завтрак | Обед | Ужин | Ккал | Руб."]
    for day in range(n_days()):
        dishes = [CATALOG[meal][int(genome[day, meal])] for meal in range(n_meals())]
        names = " | ".join(dish.name for dish in dishes)
        kcal = sum(dish.kcal for dish in dishes)
        cost = sum(dish.cost for dish in dishes)
        lines.append(f"{DAY_NAMES[day]} | {names} | {kcal} | {cost}")
    lines.append(f"Итого за неделю: {total_cost(genome)} руб.")
    return "\n".join(lines)

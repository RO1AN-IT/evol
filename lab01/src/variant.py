"""Индивидуальный вариант по формуле из методических указаний."""


# Область поиска и формула HGBat — приложение к комплексу лабораторных работ,
# вариант 20. Глобальный минимум выводится аналитически в отчёте.
LOWER = -15.0
UPPER = 15.0
KNOWN_MINIMUM = 0.0
OPTIMUM_COORDINATE = -1.0


def variant_number(index_in_group: int, group_index: int, year_digits: int) -> int:
    """V = 1 + ((17N + 7Q + Y) mod 20), V ∈ {1, …, 20}."""
    return 1 + (17 * index_in_group + 7 * group_index + year_digits) % 20


def dimension(variant: int) -> int:
    """d = 5 + (V mod 6), поэтому размерность лежит от 5 до 10."""
    return 5 + variant % 6


def describe(index_in_group: int, group_index: int, year_digits: int) -> dict:
    variant = variant_number(index_in_group, group_index, year_digits)
    if variant != 20:
        raise ValueError(
            f"Эта программа реализует вариант 20 (HGBat), получено V = {variant}."
        )
    total = 17 * index_in_group + 7 * group_index + year_digits
    return {
        "N": index_in_group,
        "Q": group_index,
        "Y": year_digits,
        "expression": total,
        "mod20": total % 20,
        "V": variant,
        "function": "HGBat",
        "dimension": dimension(variant),
        "lower": LOWER,
        "upper": UPPER,
        "known_minimum": KNOWN_MINIMUM,
        "optimum_coordinate": OPTIMUM_COORDINATE,
    }

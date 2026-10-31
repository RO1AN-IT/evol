"""Индивидуальный вариант по формуле из методических указаний."""


def variant_number(index_in_group: int, group_index: int, year_digits: int) -> int:
    """V = 1 + ((17N + 7Q + Y) mod 20)."""
    return 1 + (17 * index_in_group + 7 * group_index + year_digits) % 20


def describe(index_in_group: int, group_index: int, year_digits: int) -> dict:
    variant = variant_number(index_in_group, group_index, year_digits)
    if variant != 20:
        raise ValueError(
            f"Эта программа реализует вариант 20 (меню на неделю), получено V = {variant}."
        )
    total = 17 * index_in_group + 7 * group_index + year_digits
    return {
        "N": index_in_group,
        "Q": group_index,
        "Y": year_digits,
        "expression": total,
        "mod20": total % 20,
        "V": variant,
        "task": "Формирование меню на неделю",
        "representation": "целочисленный вектор",
    }

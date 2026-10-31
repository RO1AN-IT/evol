"""Графики серии запусков и блок-схема алгоритма."""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.problem import hgbat

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


COLORS = {
    "pm_0.10": "#1f4e79",
    "pm_0.40": "#c46b2d",
    "random_search": "#5c6b73",
}


def _method_map(summary: dict) -> dict:
    return {item["name"]: item for item in summary["methods"]}


def _band(trajectories: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        trajectories.min(axis=0),
        trajectories.mean(axis=0),
        trajectories.max(axis=0),
    )


def save_figures(summary: dict, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    methods = _method_map(summary)
    checkpoints = np.asarray(summary["checkpoints"])
    generations = np.arange(checkpoints.size)
    n_runs = summary["n_runs"]
    budget = summary["evaluation_budget"]

    _plot_generation_band(
        figures_dir / "convergence_generations.png",
        generations,
        methods["pm_0.10"],
        f"Лучшее значение HGBat по поколениям\n{n_runs} запусков, p_m = 0.10",
    )
    _plot_two_configs(figures_dir / "config_comparison.png", generations, methods, n_runs)
    _plot_against_random(figures_dir / "ga_vs_random.png", checkpoints, methods, budget)
    _plot_boxplot(figures_dir / "final_boxplot.png", methods, n_runs)
    _plot_flowchart(figures_dir / "flowchart.png")
    _plot_diagonal_barrier(
        figures_dir / "diagonal_barrier.png",
        summary["variant"]["dimension"],
    )


def _plot_generation_band(path: Path, generations: np.ndarray, method: dict, title: str) -> None:
    values = np.maximum(np.asarray(method["trajectories"], dtype=float), 1e-16)
    low, mean, high = _band(values)
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    axis.fill_between(generations, low, high, color=COLORS["pm_0.10"], alpha=0.18, label="Диапазон min–max")
    axis.plot(generations, low, color=COLORS["pm_0.10"], linewidth=1.0, linestyle="--", label="Минимум по запускам")
    axis.plot(generations, mean, color=COLORS["pm_0.10"], linewidth=2.2, label="Среднее по запускам")
    axis.plot(generations, high, color="#9bb8d3", linewidth=1.0, linestyle=":", label="Максимум по запускам")
    axis.set_yscale("log")
    axis.set_xlabel("Поколение")
    axis.set_ylabel("Лучшее значение f")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_two_configs(path: Path, generations: np.ndarray, methods: dict, n_runs: int) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    for name in ("pm_0.10", "pm_0.40"):
        method = methods[name]
        values = np.maximum(np.asarray(method["trajectories"], dtype=float), 1e-16)
        low, mean, high = _band(values)
        color = COLORS[name]
        axis.fill_between(generations, low, high, color=color, alpha=0.12)
        axis.plot(generations, mean, color=color, linewidth=2.2, label=method["display_name"] + ", среднее")
    axis.set_yscale("log")
    axis.set_xlabel("Поколение")
    axis.set_ylabel("Лучшее значение f")
    axis.set_title(
        "Сравнение двух вероятностей мутации\n"
        f"меняется только p_m, полоса — min–max по {n_runs} запускам"
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_against_random(path: Path, checkpoints: np.ndarray, methods: dict, budget: int) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    for name in ("pm_0.10", "pm_0.40", "random_search"):
        method = methods[name]
        mean = np.maximum(np.asarray(method["trajectories"], dtype=float), 1e-16).mean(axis=0)
        axis.plot(checkpoints, mean, color=COLORS[name], linewidth=2.2, label=method["display_name"])
    axis.set_yscale("log")
    axis.set_xlabel("Число вычислений функции")
    axis.set_ylabel("Среднее лучшее значение f")
    axis.set_title(
        "Генетический алгоритм и случайная выборка\n"
        f"одинаковый бюджет {budget:,} вычислений".replace(",", " ")
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_boxplot(path: Path, methods: dict, n_runs: int) -> None:
    names = ["pm_0.10", "pm_0.40", "random_search"]
    data = [np.maximum(methods[name]["final_fitness"], 1e-16) for name in names]
    labels = [methods[name]["display_name"] for name in names]
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    box = axis.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch, name in zip(box["boxes"], names):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.55)
    axis.set_yscale("log")
    axis.set_ylabel("Лучшее f в запуске")
    axis.set_title(f"Распределение итогового результата, {n_runs} запусков")
    axis.grid(True, axis="y", which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_diagonal_barrier(path: Path, dimension: int) -> None:
    coordinates = np.linspace(-2.5, 1.5, 500)
    values = np.asarray([hgbat(np.full(dimension, value)) for value in coordinates])
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    axis.plot(coordinates, values, color="#1f4e79", linewidth=2.2)
    axis.scatter([-1.0], [0.0], color="#c46b2d", zorder=3, label="Глобальный минимум, c = −1")
    axis.set_xlabel("Общее значение всех координат c")
    axis.set_ylabel("f(c, …, c)")
    axis.set_title("HGBat на диагонали x₁ = … = x₇ = c")
    axis.set_ylim(0, 8)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_flowchart(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.2, 12.4))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 18)
    axis.axis("off")
    axis.set_title("Блок-схема генетического алгоритма", fontsize=14, pad=6)

    main_x = 4.15
    boxes = {
        "start": _box(axis, main_x, 16.55, 6.6, 0.95, "Начало. Задать seed, бюджет и параметры", "#e8eef5"),
        "init": _box(
            axis,
            main_x,
            14.85,
            6.6,
            1.25,
            "Инициализация: равномерно в [−15, 15]^7\nОценить f (HGBat) у всей популяции",
            "#e8eef5",
        ),
        "decision": _box(axis, main_x, 12.95, 6.6, 1.05, "Бюджет вычислений исчерпан?", "#fff4d6"),
        "select": _box(
            axis,
            main_x,
            10.95,
            6.6,
            1.2,
            "Турнирная селекция родителей, k = 3\nПобеждает меньшее значение f",
            "#e7f2ea",
        ),
        "cross": _box(
            axis,
            main_x,
            9.05,
            6.6,
            1.25,
            "С вероятностью p_c — SBX-кроссовер,\nиначе потомки копируют родителей",
            "#e7f2ea",
        ),
        "mutate": _box(
            axis,
            main_x,
            7.05,
            6.6,
            1.4,
            "Гауссовская мутация каждой координаты\nс вероятностью p_m и проекция на границы",
            "#e7f2ea",
        ),
        "elite": _box(
            axis,
            main_x,
            4.85,
            6.6,
            1.45,
            "Оценить потомков. Элитизм:\nлучшие переходят без изменений.\nЗапомнить рекорд",
            "#e7f2ea",
        ),
        "stop": _box(
            axis,
            9.55,
            2.15,
            4.3,
            1.35,
            "Вернуть лучшую особь,\nзначение f и траекторию",
            "#e8eef5",
        ),
    }
    order = ["start", "init", "decision", "select", "cross", "mutate", "elite"]
    for previous, current in zip(order, order[1:]):
        _arrow_vertical(axis, boxes[previous], boxes[current])

    axis.text(main_x + 0.18, 12.15, "нет", color="#1f4e79", fontsize=10)

    # Выход из цикла: «да» уходит вправо и вниз к результату.
    decision_right = boxes["decision"][0] + boxes["decision"][2] / 2
    stop_top = boxes["stop"][1] + boxes["stop"][3] / 2
    axis.annotate(
        "",
        xy=(boxes["stop"][0], stop_top),
        xytext=(decision_right, boxes["decision"][1]),
        arrowprops={"arrowstyle": "-|>", "color": "#1f4e79", "lw": 1.2,
                    "connectionstyle": "angle,angleA=0,angleB=90,rad=0"},
    )
    axis.text(decision_right + 0.12, boxes["decision"][1] + 0.22, "да", color="#1f4e79", fontsize=10)

    # Возврат к проверке бюджета по левому краю.
    elite_left = boxes["elite"][0] - boxes["elite"][2] / 2
    decision_left = boxes["decision"][0] - boxes["decision"][2] / 2
    axis.plot(
        [elite_left, 0.55, 0.55, decision_left],
        [boxes["elite"][1], boxes["elite"][1], boxes["decision"][1], boxes["decision"][1]],
        color="#1f4e79",
        lw=1.2,
    )
    axis.annotate(
        "",
        xy=(decision_left, boxes["decision"][1]),
        xytext=(decision_left - 0.01, boxes["decision"][1]),
        arrowprops={"arrowstyle": "-|>", "color": "#1f4e79", "lw": 1.2},
    )
    axis.text(0.62, 9.0, "следующее\nпоколение", color="#1f4e79", fontsize=8.5)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _box(axis, x: float, y: float, width: float, height: float, text: str, color: str):
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=color,
        edgecolor="#1f4e79",
        linewidth=1.2,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=9)
    return (x, y, width, height)


def _arrow_vertical(axis, source, target) -> None:
    start = (source[0], source[1] - source[3] / 2)
    end = (target[0], target[1] + target[3] / 2)
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#1f4e79",
        )
    )

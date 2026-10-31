"""Графики серии и блок-схема."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "repair_reset": "#1f4e79",
    "penalty_reset": "#8c3a4b",
    "repair_swap": "#c46b2d",
    "random_repair": "#5c6b73",
    "greedy": "#2f6f4e",
}


def save_figures(summary: dict, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    methods = {item["name"]: item for item in summary["methods"]}
    checkpoints = np.asarray(summary["checkpoints"])
    generations = np.arange(checkpoints.size)
    _plot_band(
        figures_dir / "convergence_generations.png",
        generations,
        methods["repair_reset"],
        "Лучшая допустимая стоимость по поколениям\nремонт и замена блюда, 20 запусков",
        "Поколение",
    )
    _plot_methods(
        figures_dir / "constraint_comparison.png",
        checkpoints,
        [methods["repair_reset"], methods["penalty_reset"]],
        "Штраф и ремонт при одной и той же мутации замены",
        "Число оценок меню",
    )
    _plot_methods(
        figures_dir / "mutation_comparison.png",
        checkpoints,
        [methods["repair_reset"], methods["repair_swap"]],
        "Две мутации при ремонте допустимости",
        "Число оценок меню",
    )
    _plot_methods(
        figures_dir / "ga_vs_random.png",
        checkpoints,
        [methods["repair_reset"], methods["random_repair"]],
        "Генетический алгоритм и случайный поиск с тем же ремонтом",
        "Число оценок меню",
    )
    _plot_boxplot(figures_dir / "final_boxplot.png", methods)
    _plot_flowchart(figures_dir / "flowchart.png")


def _finite_mean(values: np.ndarray) -> np.ndarray:
    mean = np.full(values.shape[1], np.nan)
    for index in range(values.shape[1]):
        column = values[:, index]
        column = column[np.isfinite(column)]
        if column.size:
            mean[index] = float(column.mean())
    return mean


def _curves(method: dict) -> np.ndarray:
    values = np.asarray(method["trajectories"], dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values


def _plot_band(path: Path, x: np.ndarray, method: dict, title: str, xlabel: str) -> None:
    values = _curves(method)
    if values.shape[1] != x.size:
        return
    low = np.nanmin(values, axis=0)
    mean = _finite_mean(values)
    high = np.nanmax(values, axis=0)
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    axis.fill_between(x, low, high, color=COLORS["repair_reset"], alpha=0.18, label="Диапазон min–max")
    axis.plot(x, low, color=COLORS["repair_reset"], linestyle="--", linewidth=1.0, label="Минимум по запускам")
    axis.plot(x, mean, color=COLORS["repair_reset"], linewidth=2.2, label="Среднее по запускам")
    axis.plot(x, high, color="#9bb8d3", linestyle=":", linewidth=1.0, label="Максимум по запускам")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Лучшая допустимая стоимость, руб.")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_methods(path: Path, x: np.ndarray, methods: list[dict], title: str, xlabel: str) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    for method in methods:
        values = _curves(method)
        if values.shape[1] != x.size:
            continue
        axis.plot(
            x,
            _finite_mean(values),
            color=COLORS[method["name"]],
            linewidth=2.2,
            label=method["display_name"],
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Средняя лучшая допустимая стоимость, руб.")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_boxplot(path: Path, methods: dict) -> None:
    names = ["repair_reset", "repair_swap", "random_repair", "greedy", "penalty_reset"]
    data = []
    labels = []
    colors = []
    for name in names:
        costs = [value for value in methods[name]["final_costs"] if value is not None]
        if not costs:
            continue
        data.append(costs)
        labels.append(methods[name]["display_name"])
        colors.append(COLORS[name])
    figure, axis = plt.subplots(figsize=(8.6, 5.0))
    box = axis.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    axis.set_ylabel("Лучшая допустимая стоимость, руб.")
    axis.set_title("Итоговая стоимость допустимого меню, 20 запусков")
    axis.tick_params(axis="x", labelrotation=15)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_flowchart(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.2, 12.2))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 18)
    axis.axis("off")
    axis.set_title("Блок-схема генетического алгоритма меню", fontsize=14, pad=6)
    main_x = 4.2
    boxes = {
        "start": _box(axis, main_x, 16.5, 6.8, 0.95, "Задать зерно, бюджет и параметры", "#e8eef5"),
        "init": _box(
            axis,
            main_x,
            14.8,
            6.8,
            1.25,
            "Случайные индексы блюд на 7 дней\nи 3 приёма пищи. Оценить меню",
            "#e8eef5",
        ),
        "decision": _box(axis, main_x, 12.9, 6.8, 1.05, "Бюджет оценок исчерпан?", "#fff4d6"),
        "select": _box(axis, main_x, 11.0, 6.8, 1.15, "Турнир родителей, k = 3\nПобеждает меньшая оценка", "#e7f2ea"),
        "cross": _box(axis, main_x, 9.15, 6.8, 1.2, "Равномерный кроссовер\nпо слотам меню", "#e7f2ea"),
        "mutate": _box(
            axis,
            main_x,
            7.15,
            6.8,
            1.35,
            "Мутация: замена блюда\nили обмен двух дней одного приёма пищи",
            "#e7f2ea",
        ),
        "repair": _box(
            axis,
            main_x,
            5.05,
            6.8,
            1.4,
            "Ремонт или штраф:\nповторы, категории, калории, бюджет.\nЭлитизм и архив допустимого меню",
            "#e7f2ea",
        ),
        "stop": _box(axis, 9.55, 2.15, 4.3, 1.35, "Вернуть лучшее\nдопустимое меню", "#e8eef5"),
    }
    order = ["start", "init", "decision", "select", "cross", "mutate", "repair"]
    for previous, current in zip(order, order[1:]):
        _arrow_vertical(axis, boxes[previous], boxes[current])
    axis.text(main_x + 0.18, 12.1, "нет", color="#1f4e79", fontsize=10)
    decision_right = boxes["decision"][0] + boxes["decision"][2] / 2
    stop_top = boxes["stop"][1] + boxes["stop"][3] / 2
    axis.annotate(
        "",
        xy=(boxes["stop"][0], stop_top),
        xytext=(decision_right, boxes["decision"][1]),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#1f4e79",
            "lw": 1.2,
            "connectionstyle": "angle,angleA=0,angleB=90,rad=0",
        },
    )
    axis.text(decision_right + 0.12, boxes["decision"][1] + 0.22, "да", color="#1f4e79", fontsize=10)
    elite_left = boxes["repair"][0] - boxes["repair"][2] / 2
    decision_left = boxes["decision"][0] - boxes["decision"][2] / 2
    axis.plot(
        [elite_left, 0.55, 0.55, decision_left],
        [boxes["repair"][1], boxes["repair"][1], boxes["decision"][1], boxes["decision"][1]],
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
    axis.add_patch(
        FancyArrowPatch(
            (source[0], source[1] - source[3] / 2),
            (target[0], target[1] + target[3] / 2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#1f4e79",
        )
    )

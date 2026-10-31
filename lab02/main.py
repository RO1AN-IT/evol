"""Запуск экспериментов лабораторной работы № 2."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

from src.experiment import run_series


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    summary = run_series(project_dir)
    from src.plotting import save_figures

    save_figures(summary, project_dir / "results" / "figures")
    print("\nДопустимая стоимость (лучшее / среднее / медиана / СКО / худшее), число успешных запусков:")
    for method in summary["methods"]:
        stats = method["stats"]
        print(
            f"  {method['display_name']}: "
            f"{_fmt(stats['best'])} / {_fmt(stats['mean'])} / {_fmt(stats['median'])} / "
            f"{_fmt(stats['std'])} / {_fmt(stats['worst'])} "
            f"({stats['feasible_runs']}/{stats['runs']})"
        )
    print(
        f"\nЛучшее меню: {summary['best_menu_method']}, "
        f"зерно {summary['best_menu_seed']}, {summary['best_menu_cost']:.0f} руб."
    )
    print(summary["best_menu_text"])
    print(f"\nРезультаты: {project_dir / 'results'}")


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.1f}"


if __name__ == "__main__":
    main()

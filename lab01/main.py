

import os
from pathlib import Path

# Каталог кэша matplotlib внутри проекта: домашний каталог может быть закрыт.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

from src.experiment import run_series


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    summary = run_series(project_dir)
    from src.landscape import probe_landscape
    from src.plotting import save_figures

    variant = summary["variant"]
    landscape = probe_landscape(
        variant["dimension"],
        variant["lower"],
        variant["upper"],
        project_dir / "results" / "landscape.json",
    )
    save_figures(summary, project_dir / "results" / "figures")
    coordinate = landscape["coordinate_descent"]
    print(
        "Координатный спуск, 15 стартов: "
        f"лучшее {coordinate['best']:.4f}, среднее {coordinate['mean']:.4f}"
    )
    print("\nСводка (лучшее / среднее / медиана / СКО / худшее):")
    for method in summary["methods"]:
        stats = method["stats"]
        print(
            f"  {method['display_name']}: "
            f"{stats['best']:.6e} / {stats['mean']:.6e} / "
            f"{stats['median']:.6e} / {stats['std']:.6e} / {stats['worst']:.6e}"
        )
    print(f"\nРезультаты: {project_dir / 'results'}")


if __name__ == "__main__":
    main()

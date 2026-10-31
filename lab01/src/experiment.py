"""Серия запусков, сводка и сохранение таблиц."""


import csv
import json
import shutil
from pathlib import Path

import numpy as np

from src.ga import GaConfig, RunResult, run_ga
from src.problem import assert_problem, optimum_point
from src.random_search import run_random_search
from src.variant import describe


THRESHOLDS = (1e-1, 1e-2, 1e-3, 1e-4)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def configs_from_document(document: dict) -> list[GaConfig]:
    configs = [GaConfig(**item) for item in document["experiment"]["configs"]]
    if len(configs) < 2:
        raise ValueError("нужны минимум две конфигурации")
    _assert_single_factor(configs[0], configs[1])
    return configs


def _assert_single_factor(first: GaConfig, second: GaConfig) -> None:
    compared = (
        "population_size",
        "tournament_size",
        "crossover_probability",
        "sbx_eta",
        "sbx_gene_probability",
        "mutation_probability",
        "mutation_sigma",
        "elite_count",
    )
    changed = [name for name in compared if getattr(first, name) != getattr(second, name)]
    if changed != ["mutation_probability"]:
        raise ValueError(
            "конфигурации должны различаться только вероятностью мутации, "
            f"сейчас различаются: {changed}"
        )


def _distance(point: np.ndarray, dimension: int) -> float:
    return float(np.linalg.norm(point - optimum_point(dimension)))


def _evals_to_threshold(evals: np.ndarray, best: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(best <= threshold)
    if hits.size == 0:
        return None
    return int(evals[hits[0]])


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "best": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "worst": float(np.max(array)),
    }


def run_series(project_dir: Path) -> dict:
    document = load_config(project_dir / "config.json")
    student = document["student"]
    variant = describe(
        student["index_in_group"],
        student["group_index_Q"],
        student["year_digits_Y"],
    )
    assert_problem(variant["dimension"])

    experiment = document["experiment"]
    n_runs = int(experiment["n_runs"])
    if n_runs < 20:
        raise ValueError("методичка требует не меньше 20 независимых запусков")
    budget = int(experiment["evaluation_budget"])
    base_seed = int(experiment["base_seed"])
    ga_configs = configs_from_document(document)

    from src.operators import assert_operators_respect_bounds

    assert_operators_respect_bounds(variant["dimension"], variant["lower"], variant["upper"])

    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(project_dir / "config.json", results_dir / "config_used.json")

    runs: list[RunResult] = []
    print(
        f"Вариант V = {variant['V']} ({variant['function']}), "
        f"d = {variant['dimension']}, бюджет = {budget}, запусков = {n_runs}"
    )
    print(
        f"V = 1 + ((17*{variant['N']} + 7*{variant['Q']} + {variant['Y']}) mod 20) "
        f"= 1 + ({variant['expression']} mod 20) = 1 + {variant['mod20']} = {variant['V']}"
    )

    reference_checkpoints = None
    for config in ga_configs:
        for index in range(n_runs):
            seed = base_seed + index
            result = run_ga(
                config,
                variant["dimension"],
                variant["lower"],
                variant["upper"],
                budget,
                seed,
            )
            runs.append(result)
            if reference_checkpoints is None:
                reference_checkpoints = result.checkpoint_evals
            elif not np.array_equal(reference_checkpoints, result.checkpoint_evals):
                raise AssertionError("контрольные точки запусков не совпали")
            print(
                f"[{config.display_name}] {index + 1:02d}/{n_runs} "
                f"seed={seed} f={result.best_f:.6e} "
                f"||x+1||={_distance(result.best_x, variant['dimension']):.3e} "
                f"t={result.runtime_sec:.2f}s"
            )

    random_runs = []
    for index in range(n_runs):
        seed = base_seed + index
        result = run_random_search(
            variant["dimension"],
            variant["lower"],
            variant["upper"],
            budget,
            seed,
            reference_checkpoints,
        )
        random_runs.append(result)
        print(
            f"[Случайный поиск] {index + 1:02d}/{n_runs} "
            f"seed={seed} f={result.best_f:.6e} "
            f"||x+1||={_distance(result.best_x, variant['dimension']):.3e} "
            f"t={result.runtime_sec:.2f}s"
        )

    all_runs = runs + random_runs
    summary = _summarize(all_runs, variant, experiment, reference_checkpoints)
    _write_runs_csv(results_dir / "runs.csv", all_runs, variant["dimension"])
    _write_summary_csv(results_dir / "summary.csv", summary)
    with (results_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(_without_trajectories(summary), file, ensure_ascii=False, indent=2)

    trajectories = {
        item["name"]: np.vstack(item["trajectories"]) for item in summary["methods"]
    }
    np.savez(
        results_dir / "trajectories.npz",
        checkpoint_evals=reference_checkpoints,
        **trajectories,
    )
    return summary


def _summarize(all_runs, variant: dict, experiment: dict, checkpoints: np.ndarray) -> dict:
    methods = []
    order = []
    for result in all_runs:
        if result.method not in order:
            order.append(result.method)
    grouped = {name: [item for item in all_runs if item.method == name] for name in order}

    for name, group in grouped.items():
        finals = [item.best_f for item in group]
        distances = [_distance(item.best_x, variant["dimension"]) for item in group]
        threshold_stats = {}
        for threshold in THRESHOLDS:
            reached = []
            for item in group:
                moment = _evals_to_threshold(
                    item.checkpoint_evals, item.checkpoint_best, threshold
                )
                if moment is not None:
                    reached.append(moment)
            key = f"{threshold:.0e}"
            threshold_stats[key] = {
                "successes": len(reached),
                "rate": len(reached) / len(group),
                "median_evals": float(np.median(reached)) if reached else None,
            }
        best_run = min(group, key=lambda item: item.best_f)
        methods.append(
            {
                "name": name,
                "display_name": group[0].display_name,
                "stats": _stats(finals),
                "distance_stats": _stats(distances),
                "mean_runtime_sec": float(np.mean([item.runtime_sec for item in group])),
                "evaluations": group[0].evaluations,
                "generations": group[0].generations,
                "thresholds": threshold_stats,
                "best_solution": {
                    "seed": best_run.seed,
                    "fitness": best_run.best_f,
                    "x": best_run.best_x.tolist(),
                    "distance": _distance(best_run.best_x, variant["dimension"]),
                },
                "final_fitness": finals,
                "trajectories": [item.checkpoint_best.tolist() for item in group],
            }
        )

    # Траектории нужны для графиков, но в JSON сводке они дублируют npz.
    # Оставляем их: отчёт и графики читают summary до отдельной перезаписи.
    payload = {
        "variant": variant,
        "n_runs": len(next(iter(grouped.values()))),
        "base_seed": experiment["base_seed"],
        "evaluation_budget": experiment["evaluation_budget"],
        "checkpoints": checkpoints.tolist(),
        "methods": methods,
        "compared_factor": "mutation_probability",
    }
    return payload


def _without_trajectories(summary: dict) -> dict:
    stored = {key: value for key, value in summary.items() if key != "methods"}
    stored["methods"] = [
        {key: value for key, value in method.items() if key != "trajectories"}
        for method in summary["methods"]
    ]
    return stored


def _write_runs_csv(path: Path, runs, dimension: int) -> None:
    fieldnames = [
        "method",
        "display_name",
        "run_in_method",
        "seed",
        "best_fitness",
        "distance_to_optimum",
        "evaluations",
        "generations",
        "runtime_sec",
        *[f"x{index}" for index in range(1, dimension + 1)],
    ]
    counters: dict[str, int] = {}
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in runs:
            counters[result.method] = counters.get(result.method, 0) + 1
            row = {
                "method": result.method,
                "display_name": result.display_name,
                "run_in_method": counters[result.method],
                "seed": result.seed,
                "best_fitness": f"{result.best_f:.16e}",
                "distance_to_optimum": f"{_distance(result.best_x, dimension):.16e}",
                "evaluations": result.evaluations,
                "generations": result.generations,
                "runtime_sec": f"{result.runtime_sec:.6f}",
            }
            for index, coordinate in enumerate(result.best_x, start=1):
                row[f"x{index}"] = f"{coordinate:.16e}"
            writer.writerow(row)


def _write_summary_csv(path: Path, summary: dict) -> None:
    fieldnames = [
        "method",
        "display_name",
        "best",
        "mean",
        "median",
        "std",
        "worst",
        "mean_distance",
        "mean_runtime_sec",
        "evaluations",
        "success_rate_1e-2",
        "median_evals_to_1e-2",
        "success_rate_1e-3",
        "median_evals_to_1e-3",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for method in summary["methods"]:
            stats = method["stats"]
            writer.writerow(
                {
                    "method": method["name"],
                    "display_name": method["display_name"],
                    "best": f"{stats['best']:.16e}",
                    "mean": f"{stats['mean']:.16e}",
                    "median": f"{stats['median']:.16e}",
                    "std": f"{stats['std']:.16e}",
                    "worst": f"{stats['worst']:.16e}",
                    "mean_distance": f"{method['distance_stats']['mean']:.16e}",
                    "mean_runtime_sec": f"{method['mean_runtime_sec']:.6f}",
                    "evaluations": method["evaluations"],
                    "success_rate_1e-2": method["thresholds"]["1e-02"]["rate"],
                    "median_evals_to_1e-2": method["thresholds"]["1e-02"]["median_evals"],
                    "success_rate_1e-3": method["thresholds"]["1e-03"]["rate"],
                    "median_evals_to_1e-3": method["thresholds"]["1e-03"]["median_evals"],
                }
            )

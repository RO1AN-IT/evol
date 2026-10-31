"""Серия запусков, примеры меню и сводка."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from src.baselines import run_greedy, run_random_repaired
from src.catalog import CATALOG, dishes_as_dicts
from src.ga import GaConfig, RunResult, run_ga
from src.problem import (
    Limits,
    describe_violations,
    format_menu,
    is_feasible,
    total_cost,
    violation_details,
)
from src.repair import repair
from src.variant import describe


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def limits_from(document: dict) -> Limits:
    return Limits(**document["problem"])


def configs_from(document: dict) -> list[GaConfig]:
    configs = [GaConfig(**item) for item in document["experiment"]["configs"]]
    by_name = {item.name: item for item in configs}
    _assert_only(by_name["repair_reset"], by_name["penalty_reset"], "constraint_mode")
    _assert_only(by_name["repair_reset"], by_name["repair_swap"], "mutation")
    return configs


def _assert_only(first: GaConfig, second: GaConfig, expected: str) -> None:
    fields = (
        "constraint_mode",
        "mutation",
        "population_size",
        "tournament_size",
        "crossover_probability",
        "reset_probability",
        "swaps_per_meal",
        "elite_count",
    )
    changed = [name for name in fields if getattr(first, name) != getattr(second, name)]
    if changed != [expected]:
        raise ValueError(f"ожидалось различие только в {expected}, сейчас {changed}")


def run_series(project_dir: Path) -> dict:
    document = load_config(project_dir / "config.json")
    student = document["student"]
    variant = describe(
        student["index_in_group"],
        student["group_index_Q"],
        student["year_digits_Y"],
    )
    limits = limits_from(document)
    experiment = document["experiment"]
    n_runs = int(experiment["n_runs"])
    if n_runs < 20:
        raise ValueError("нужно не меньше 20 запусков")
    budget = int(experiment["evaluation_budget"])
    base_seed = int(experiment["base_seed"])
    ga_configs = configs_from(document)

    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "dishes.json").open("w", encoding="utf-8") as file:
        json.dump(dishes_as_dicts(), file, ensure_ascii=False, indent=2)
    shutil.copyfile(project_dir / "config.json", results_dir / "config_used.json")

    examples = _examples(limits)
    (results_dir / "examples.txt").write_text(examples["text"], encoding="utf-8")

    print(
        f"Вариант V = {variant['V']} ({variant['task']}), "
        f"бюджет = {budget}, запусков = {n_runs}"
    )
    print(
        f"V = 1 + ((17*{variant['N']} + 7*{variant['Q']} + {variant['Y']}) mod 20) "
        f"= 1 + ({variant['expression']} mod 20) = {variant['V']}"
    )

    runs: list[RunResult] = []
    reference_checkpoints = None
    for config in ga_configs:
        for index in range(n_runs):
            seed = base_seed + index
            result = run_ga(config, limits, budget, seed)
            runs.append(result)
            if reference_checkpoints is None:
                reference_checkpoints = result.checkpoint_evals
            elif not np.array_equal(reference_checkpoints, result.checkpoint_evals):
                raise AssertionError("контрольные точки запусков не совпали")
            cost_text = "нет" if result.best_feasible_cost is None else f"{result.best_feasible_cost:.0f}"
            print(
                f"[{config.display_name}] {index + 1:02d}/{n_runs} "
                f"seed={seed} стоимость={cost_text} t={result.runtime_sec:.2f}s"
            )

    for index in range(n_runs):
        seed = base_seed + index
        result = run_random_repaired(limits, budget, seed, reference_checkpoints)
        runs.append(result)
        print(
            f"[Случайный поиск с ремонтом] {index + 1:02d}/{n_runs} "
            f"seed={seed} стоимость={result.best_feasible_cost:.0f} t={result.runtime_sec:.2f}s"
        )
        greedy = run_greedy(limits, seed)
        runs.append(greedy)
        print(
            f"[Жадная эвристика] {index + 1:02d}/{n_runs} "
            f"seed={seed} стоимость={greedy.best_feasible_cost:.0f}"
        )

    summary = _summarize(runs, variant, experiment, limits, reference_checkpoints)
    _write_runs_csv(results_dir / "runs.csv", runs, limits)
    _write_summary_csv(results_dir / "summary.csv", summary)
    with (results_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(_without_trajectories(summary), file, ensure_ascii=False, indent=2)
    trajectories = {
        item["name"]: np.vstack(item["trajectories"]) for item in summary["methods"] if item["trajectories"]
    }
    np.savez(results_dir / "trajectories.npz", checkpoint_evals=reference_checkpoints, **trajectories)
    best = _overall_best(runs)
    menu_text = format_menu(best.best_genome)
    (results_dir / "best_menu.txt").write_text(
        f"{best.display_name}, зерно {best.seed}, {best.best_feasible_cost:.0f} руб.\n\n{menu_text}\n",
        encoding="utf-8",
    )
    summary["best_menu_text"] = menu_text
    summary["best_menu_method"] = best.display_name
    summary["best_menu_seed"] = best.seed
    summary["best_menu_cost"] = best.best_feasible_cost
    return summary


def _examples(limits: Limits) -> dict:
    repeats = np.zeros((7, 3), dtype=int)
    heavy = np.zeros((7, 3), dtype=int)
    for meal, dishes in enumerate(CATALOG):
        ranked = sorted(dishes, key=lambda dish: (-dish.cost, dish.index))
        for day in range(7):
            heavy[day, meal] = ranked[day].index
    if is_feasible(repeats, limits) or is_feasible(heavy, limits):
        raise AssertionError("контрольные недопустимые меню оказались допустимыми")
    feasible = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        genome = repair(rng.integers(0, 12, size=(7, 3)), rng, limits)
        if is_feasible(genome, limits):
            feasible.append(genome)
        if len(feasible) == 2:
            break
    if len(feasible) < 2:
        raise RuntimeError("не собраны два допустимых примера")
    blocks = []
    named = [
        ("Недопустимое меню 1: один и тот же набор каждый день", repeats),
        ("Недопустимое меню 2: самые дорогие блюда без повтора внутри приёма пищи", heavy),
        ("Допустимое меню 1", feasible[0]),
        ("Допустимое меню 2", feasible[1]),
    ]
    for title, genome in named:
        details = violation_details(genome, limits)
        blocks.append(
            "\n".join(
                [
                    title,
                    format_menu(genome),
                    "Нарушения: " + "; ".join(describe_violations(genome, limits)),
                    f"Стоимость {details['cost']} руб.",
                    "",
                ]
            )
        )
    return {"text": "\n".join(blocks)}


def _summarize(runs, variant, experiment, limits: Limits, checkpoints) -> dict:
    order = []
    for result in runs:
        if result.method not in order:
            order.append(result.method)
    methods = []
    for name in order:
        group = [item for item in runs if item.method == name]
        costs = [item.best_feasible_cost for item in group]
        finite = [item for item in costs if item is not None]
        array = np.asarray(finite, dtype=float)
        stats = {
            "best": None if array.size == 0 else float(array.min()),
            "mean": None if array.size == 0 else float(array.mean()),
            "median": None if array.size == 0 else float(np.median(array)),
            "std": None if array.size < 2 else float(array.std(ddof=1)),
            "worst": None if array.size == 0 else float(array.max()),
            "feasible_runs": int(array.size),
            "runs": len(group),
        }
        best_run = min(
            (item for item in group if item.best_feasible_cost is not None),
            key=lambda item: item.best_feasible_cost,
            default=None,
        )
        penalty_case = min(group, key=lambda item: item.best_fitness)
        methods.append(
            {
                "name": name,
                "display_name": group[0].display_name,
                "stats": stats,
                "mean_runtime_sec": float(np.mean([item.runtime_sec for item in group])),
                "evaluations": group[0].evaluations,
                "best_solution": None
                if best_run is None
                else {
                    "seed": best_run.seed,
                    "cost": best_run.best_feasible_cost,
                    "genome": best_run.best_genome.tolist(),
                    "menu": format_menu(best_run.best_genome),
                },
                "closest_penalty": {
                    "seed": penalty_case.seed,
                    "fitness": penalty_case.best_fitness,
                    "violations": violation_details(penalty_case.best_fitness_genome, limits),
                    "lines": describe_violations(penalty_case.best_fitness_genome, limits),
                },
                "final_costs": costs,
                "trajectories": [item.checkpoint_cost.tolist() for item in group],
            }
        )
    return {
        "variant": variant,
        "limits": {
            "calorie_min": limits.calorie_min,
            "calorie_max": limits.calorie_max,
            "budget": limits.budget,
        },
        "n_runs": n_runs_of(runs),
        "base_seed": experiment["base_seed"],
        "evaluation_budget": experiment["evaluation_budget"],
        "checkpoints": checkpoints.tolist(),
        "methods": methods,
    }


def n_runs_of(runs) -> int:
    first = runs[0].method
    return sum(1 for item in runs if item.method == first)


def _overall_best(runs: list[RunResult]) -> RunResult:
    feasible_runs = [item for item in runs if item.best_feasible_cost is not None]
    return min(feasible_runs, key=lambda item: item.best_feasible_cost)


def _without_trajectories(summary: dict) -> dict:
    stored = {key: value for key, value in summary.items() if key != "methods"}
    stored["methods"] = [
        {key: value for key, value in method.items() if key != "trajectories"}
        for method in summary["methods"]
    ]
    return stored


def _write_runs_csv(path: Path, runs: list[RunResult], limits: Limits) -> None:
    fields = [
        "method",
        "display_name",
        "run_in_method",
        "seed",
        "feasible",
        "best_feasible_cost",
        "best_fitness",
        "kcal_outside",
        "repeats",
        "missing_categories",
        "rubles_over_budget",
        "evaluations",
        "runtime_sec",
    ]
    counters: dict[str, int] = {}
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in runs:
            counters[result.method] = counters.get(result.method, 0) + 1
            details = violation_details(result.best_fitness_genome, limits)
            writer.writerow(
                {
                    "method": result.method,
                    "display_name": result.display_name,
                    "run_in_method": counters[result.method],
                    "seed": result.seed,
                    "feasible": int(result.feasible),
                    "best_feasible_cost": ""
                    if result.best_feasible_cost is None
                    else f"{result.best_feasible_cost:.0f}",
                    "best_fitness": f"{result.best_fitness:.4f}",
                    "kcal_outside": details["kcal_outside"],
                    "repeats": details["repeats"],
                    "missing_categories": details["missing_categories"],
                    "rubles_over_budget": details["rubles_over_budget"],
                    "evaluations": result.evaluations,
                    "runtime_sec": f"{result.runtime_sec:.6f}",
                }
            )


def _write_summary_csv(path: Path, summary: dict) -> None:
    fields = ["method", "display_name", "feasible_runs", "best", "mean", "median", "std", "worst", "mean_runtime_sec"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for method in summary["methods"]:
            stats = method["stats"]
            writer.writerow(
                {
                    "method": method["name"],
                    "display_name": method["display_name"],
                    "feasible_runs": stats["feasible_runs"],
                    "best": _num(stats["best"]),
                    "mean": _num(stats["mean"]),
                    "median": _num(stats["median"]),
                    "std": _num(stats["std"]),
                    "worst": _num(stats["worst"]),
                    "mean_runtime_sec": f"{method['mean_runtime_sec']:.6f}",
                }
            )


def _num(value) -> str:
    return "" if value is None else f"{value:.4f}"

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .exact import exact_cvrp
from .operators import OPERATOR_NAMES
from .problem import CVRPInstance, generate_instance
from .search import SearchResult, run_search
from .selectors import (
    AdaptiveWeightSelector,
    FixedSelector,
    QLearningSelector,
    RandomSelector,
    RoundRobinSelector,
)


@dataclass(frozen=True)
class MethodSummary:
    split: str
    method: str
    observations: int
    feasibility_rate: float
    final_objective_mean: float
    final_objective_std: float
    final_objective_median: float
    final_objective_p90: float
    improvement_pct_mean: float
    optimality_gap_pct_mean: float | None
    runtime_s_mean: float
    time_to_best_s_mean: float
    iteration_to_best_mean: float


def _instances(split: str, spec: dict[str, Any]) -> list[CVRPInstance]:
    count = int(spec["count"])
    seed_start = int(spec["seed_start"])
    low, high = (int(x) for x in spec["n_customers_range"])
    target_routes = int(spec["target_routes"])
    items = []
    for offset in range(count):
        seed = seed_start + offset
        rng = np.random.default_rng(seed)
        n = int(rng.integers(low, high + 1))
        items.append(
            generate_instance(
                n_customers=n,
                seed=seed,
                target_routes=target_routes,
                instance_id=f"{split}-{offset:04d}-s{seed}",
            )
        )
    return items


def _train_q(
    train_instances: list[CVRPInstance], config: dict[str, Any], training_seed: int
) -> QLearningSelector:
    qcfg = config["q_learning"]
    selector = QLearningSelector(
        len(OPERATOR_NAMES),
        alpha=float(qcfg["alpha"]),
        gamma=float(qcfg["gamma"]),
        epsilon=float(qcfg["epsilon"]),
        epsilon_min=float(qcfg["epsilon_min"]),
        epsilon_decay=float(qcfg["epsilon_decay"]),
    )
    episodes = int(qcfg["epochs"])
    iterations = int(config["search"]["iterations"])
    for epoch in range(episodes):
        for index, instance in enumerate(train_instances):
            seed = int(np.random.SeedSequence([training_seed, epoch, index]).generate_state(1)[0])
            run_search(instance, selector, iterations=iterations, search_seed=seed)
    return selector.freeze()


def _choose_fixed_operator(
    validation: list[CVRPInstance], config: dict[str, Any], comparison_seeds: list[int]
) -> int:
    iterations = int(config["search"]["iterations"])
    scores = np.zeros(len(OPERATOR_NAMES), dtype=float)
    counts = np.zeros(len(OPERATOR_NAMES), dtype=int)
    for action in range(len(OPERATOR_NAMES)):
        for instance in validation:
            for comparison_seed in comparison_seeds:
                seed = instance.seed ^ comparison_seed
                result = run_search(
                    instance,
                    FixedSelector(len(OPERATOR_NAMES), action),
                    iterations=iterations,
                    search_seed=seed,
                )
                scores[action] += result.best_objective / result.initial_objective
                counts[action] += 1
    return int(np.argmin(scores / np.maximum(counts, 1)))


def _selector(method: str, fixed_action: int, q_selector: QLearningSelector):
    n = len(OPERATOR_NAMES)
    if method == "fixed_best_validation":
        return FixedSelector(n, fixed_action)
    if method == "random":
        return RandomSelector(n)
    if method == "round_robin":
        return RoundRobinSelector(n)
    if method == "adaptive_weights":
        return AdaptiveWeightSelector(n)
    if method == "q_learning":
        return q_selector
    raise KeyError(method)


def _record(
    *,
    split: str,
    instance: CVRPInstance,
    method: str,
    comparison_seed: int,
    training_seed: int,
    result: SearchResult,
    optimal: float | None,
) -> dict[str, Any]:
    gap = None if optimal is None else 100.0 * (result.best_objective - optimal) / optimal
    return {
        "split": split,
        "instance_id": instance.instance_id,
        "instance_seed": instance.seed,
        "n_customers": instance.n_customers,
        "capacity": instance.capacity,
        "method": method,
        "comparison_seed": comparison_seed,
        "training_seed": training_seed if method == "q_learning" else None,
        "initial_objective": result.initial_objective,
        "final_objective": result.best_objective,
        "improvement_pct": 100.0
        * (result.initial_objective - result.best_objective)
        / result.initial_objective,
        "optimal_objective": optimal,
        "optimality_gap_pct": gap,
        "feasibility_rate": result.feasibility_rate,
        "runtime_s": result.runtime_s,
        "time_to_best_s": result.time_to_best_s,
        "iteration_to_best": result.iteration_to_best,
        "operator_usage": result.operator_usage,
        "trajectory": result.trace,
    }


def _summary(records: list[dict[str, Any]]) -> list[MethodSummary]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["split"], row["method"])].append(row)
    summaries = []
    for (split, method), rows in sorted(grouped.items()):
        final = np.asarray([r["final_objective"] for r in rows], dtype=float)
        improvement = np.asarray([r["improvement_pct"] for r in rows], dtype=float)
        gaps = np.asarray(
            [r["optimality_gap_pct"] for r in rows if r["optimality_gap_pct"] is not None],
            dtype=float,
        )
        summaries.append(
            MethodSummary(
                split=split,
                method=method,
                observations=len(rows),
                feasibility_rate=float(np.mean([r["feasibility_rate"] for r in rows])),
                final_objective_mean=float(final.mean()),
                final_objective_std=float(final.std(ddof=1)) if len(final) > 1 else 0.0,
                final_objective_median=float(median(final.tolist())),
                final_objective_p90=float(np.quantile(final, 0.90)),
                improvement_pct_mean=float(improvement.mean()),
                optimality_gap_pct_mean=float(gaps.mean()) if gaps.size else None,
                runtime_s_mean=float(np.mean([r["runtime_s"] for r in rows])),
                time_to_best_s_mean=float(np.mean([r["time_to_best_s"] for r in rows])),
                iteration_to_best_mean=float(np.mean([r["iteration_to_best"] for r in rows])),
            )
        )
    return summaries


def _bootstrap_ci(values: np.ndarray, seed: int, draws: int = 2000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _paired(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in sorted({r["split"] for r in records}):
        q_rows = [r for r in records if r["split"] == split and r["method"] == "q_learning"]
        for baseline in (
            "fixed_best_validation",
            "random",
            "round_robin",
            "adaptive_weights",
        ):
            base = {
                (r["instance_id"], r["comparison_seed"]): r["final_objective"]
                for r in records
                if r["split"] == split and r["method"] == baseline
            }
            diffs = []
            for row in q_rows:
                key = (row["instance_id"], row["comparison_seed"])
                if key in base:
                    diffs.append(row["final_objective"] - base[key])
            arr = np.asarray(diffs, dtype=float)
            if not arr.size:
                continue
            low, high = _bootstrap_ci(arr, seed=991 + len(output))
            output[f"{split}/q_learning-minus-{baseline}"] = {
                "n_pairs": int(arr.size),
                "mean_objective_difference": float(arr.mean()),
                "median_objective_difference": float(np.median(arr)),
                "ci95_low": low,
                "ci95_high": high,
            }
    return output


def run_benchmark(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_instances = {name: _instances(name, spec) for name, spec in config["splits"].items()}
    comparison_seeds = [int(x) for x in config["evaluation"]["comparison_seeds"]]
    training_seeds = [int(x) for x in config["q_learning"]["training_seeds"]]
    if comparison_seeds != training_seeds:
        raise ValueError("comparison_seeds must equal training_seeds for paired evaluation")
    fixed_action = _choose_fixed_operator(split_instances["validation"], config, comparison_seeds)
    q_models = {seed: _train_q(split_instances["train"], config, seed) for seed in training_seeds}
    methods = (
        "fixed_best_validation",
        "random",
        "round_robin",
        "adaptive_weights",
        "q_learning",
    )
    iterations = int(config["search"]["iterations"])
    exact_max = int(config["evaluation"]["exact_max_customers"])
    records: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []

    for split in ("test", "ood"):
        for instance in split_instances[split]:
            optimal = None
            if instance.n_customers <= exact_max:
                exact = exact_cvrp(instance, max_customers=exact_max)
                optimal = exact.objective
                oracle_rows.append(
                    {
                        "split": split,
                        "instance_id": instance.instance_id,
                        "objective": exact.objective,
                        "certified_optimal": exact.certified_optimal,
                        "states_evaluated": exact.states_evaluated,
                    }
                )
            for comparison_seed in comparison_seeds:
                search_seed = instance.seed ^ comparison_seed
                for method in methods:
                    selector = _selector(method, fixed_action, q_models[comparison_seed])
                    result = run_search(
                        instance,
                        selector,
                        iterations=iterations,
                        search_seed=search_seed,
                        initial_temperature_ratio=float(
                            config["search"]["initial_temperature_ratio"]
                        ),
                        cooling=float(config["search"]["cooling"]),
                    )
                    records.append(
                        _record(
                            split=split,
                            instance=instance,
                            method=method,
                            comparison_seed=comparison_seed,
                            training_seed=comparison_seed,
                            result=result,
                            optimal=optimal,
                        )
                    )

    summaries = _summary(records)
    operator_usage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in records:
        for name, count in row["operator_usage"].items():
            operator_usage[row["method"]][name] += int(count)
    payload = {
        "schema_version": 1,
        "problem": "synthetic_euclidean_cvrp",
        "policy_role": "select_low_level_search_operator_only",
        "reward": "accepted objective improvement normalized by initial objective",
        "final_metric": "best feasible CVRP distance",
        "fixed_operator_selected_on": "validation",
        "fixed_operator": OPERATOR_NAMES[fixed_action],
        "training_seeds": training_seeds,
        "comparison_seeds": comparison_seeds,
        "operator_names": list(OPERATOR_NAMES),
        "summaries": [asdict(item) for item in summaries],
        "paired_differences": _paired(records),
        "operator_usage": {method: dict(counts) for method, counts in operator_usage.items()},
        "oracle": oracle_rows,
        "records": records,
        "claims_note": (
            "CI/synthetic benchmark outputs are methodological evidence only; "
            "no paper-level or universal-superiority claim is implied."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(item) for item in summaries)
    with (output / "raw_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return payload

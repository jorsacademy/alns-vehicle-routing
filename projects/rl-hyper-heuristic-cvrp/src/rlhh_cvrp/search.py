from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import exp
from time import perf_counter
from typing import Any

import numpy as np

from .operators import OPERATOR_NAMES, OPERATORS
from .problem import (
    CVRPInstance,
    Solution,
    audit_solution,
    greedy_initial_solution,
    solution_distance,
    solution_statistics,
)
from .selectors import SearchState, Selector, TransitionInfo


@dataclass(frozen=True)
class SearchResult:
    initial_solution: Solution
    best_solution: Solution
    initial_objective: float
    best_objective: float
    runtime_s: float
    time_to_best_s: float
    iteration_to_best: int
    feasibility_rate: float
    operator_usage: dict[str, int]
    trace: list[dict[str, Any]]


def _bin(value: float, thresholds: tuple[float, ...]) -> int:
    return int(np.digitize([value], thresholds)[0])


def build_state(
    *,
    instance: CVRPInstance,
    solution: Solution,
    current_objective: float,
    best_objective: float,
    initial_objective: float,
    recent_rewards: deque[float],
    recent_acceptance: deque[bool],
    stagnation: int,
    iteration: int,
    budget: int,
    last_action: int | None,
) -> SearchState:
    stats = solution_statistics(instance, solution)
    recent_improvement = float(np.mean(recent_rewards)) if recent_rewards else 0.0
    acceptance_ratio = float(np.mean(recent_acceptance)) if recent_acceptance else 0.0
    stagnation_fraction = stagnation / max(1, budget)
    remaining_fraction = max(0.0, (budget - iteration) / max(1, budget))
    current_ratio = current_objective / max(initial_objective, 1e-9)
    best_ratio = best_objective / max(initial_objective, 1e-9)
    features = np.asarray(
        [
            current_ratio,
            best_ratio,
            recent_improvement,
            stagnation_fraction,
            acceptance_ratio,
            remaining_fraction,
            stats["load_cv"],
            stats["route_length_cv"],
            stats["near_capacity_fraction"],
            stats["route_count"] / max(1.0, instance.n_customers),
            -1.0 if last_action is None else float(last_action),
        ],
        dtype=float,
    )
    key = (
        _bin(current_ratio, (0.80, 0.90, 0.97, 1.02)),
        _bin(recent_improvement, (-0.005, -0.0005, 0.0005, 0.005)),
        _bin(stagnation_fraction, (0.05, 0.20, 0.50)),
        _bin(acceptance_ratio, (0.25, 0.50, 0.75)),
        _bin(remaining_fraction, (0.25, 0.50, 0.75)),
        _bin(stats["load_cv"], (0.10, 0.25, 0.50)),
        len(OPERATOR_NAMES) if last_action is None else int(last_action),
    )
    return SearchState(features=features, key=key)


def _rng(seed: int, iteration: int, stream: int, action: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, iteration, stream, action]))


def run_search(
    instance: CVRPInstance,
    selector: Selector,
    *,
    iterations: int,
    search_seed: int,
    initial_temperature_ratio: float = 0.03,
    cooling: float = 0.995,
) -> SearchResult:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    initial = greedy_initial_solution(instance)
    current = initial
    best = initial
    initial_objective = solution_distance(instance, initial)
    current_objective = initial_objective
    best_objective = initial_objective
    temperature = max(1e-9, initial_temperature_ratio * initial_objective)
    recent_rewards: deque[float] = deque(maxlen=20)
    recent_acceptance: deque[bool] = deque(maxlen=20)
    stagnation = 0
    last_action: int | None = None
    usage: Counter[str] = Counter()
    feasible_steps = 0
    trace: list[dict[str, Any]] = []
    iteration_to_best = 0
    time_to_best = 0.0
    selector.reset_episode()
    selector_rng = np.random.default_rng(np.random.SeedSequence([search_seed, 17]))
    started = perf_counter()

    for iteration in range(iterations):
        state = build_state(
            instance=instance,
            solution=current,
            current_objective=current_objective,
            best_objective=best_objective,
            initial_objective=initial_objective,
            recent_rewards=recent_rewards,
            recent_acceptance=recent_acceptance,
            stagnation=stagnation,
            iteration=iteration,
            budget=iterations,
            last_action=last_action,
        )
        action = int(selector.select(state, selector_rng))
        if not 0 <= action < len(OPERATOR_NAMES):
            raise RuntimeError(f"selector returned invalid action {action}")
        operator_name = OPERATOR_NAMES[action]
        candidate = OPERATORS[operator_name](
            instance, current, _rng(search_seed, iteration, 31, action)
        )
        audit = audit_solution(instance, candidate)
        if audit.feasible:
            feasible_steps += 1
        else:
            raise RuntimeError(f"candidate failed independent feasibility audit: {audit}")
        candidate_objective = audit.objective
        before = current_objective
        delta = candidate_objective - current_objective
        accept_draw = float(_rng(search_seed, iteration, 47).random())
        accepted = delta <= 0.0 or accept_draw < exp(-delta / max(temperature, 1e-12))
        if accepted:
            current = candidate
            current_objective = candidate_objective
        reward = (before - current_objective) / max(initial_objective, 1e-9)
        improved_current = current_objective < before - 1e-12
        new_best = current_objective < best_objective - 1e-12
        if new_best:
            best = current
            best_objective = current_objective
            iteration_to_best = iteration + 1
            time_to_best = perf_counter() - started
            stagnation = 0
        else:
            stagnation += 1
        recent_rewards.append(reward)
        recent_acceptance.append(accepted)
        usage[operator_name] += 1
        last_action = action
        next_state = build_state(
            instance=instance,
            solution=current,
            current_objective=current_objective,
            best_objective=best_objective,
            initial_objective=initial_objective,
            recent_rewards=recent_rewards,
            recent_acceptance=recent_acceptance,
            stagnation=stagnation,
            iteration=iteration + 1,
            budget=iterations,
            last_action=last_action,
        )
        selector.update(
            state,
            action,
            reward,
            next_state,
            iteration + 1 == iterations,
            TransitionInfo(
                accepted=accepted,
                improved_current=improved_current,
                new_best=new_best,
            ),
        )
        trace.append(
            {
                "iteration": iteration + 1,
                "operator": operator_name,
                "accepted": accepted,
                "reward": reward,
                "current_objective": current_objective,
                "best_objective": best_objective,
                "temperature": temperature,
                "stagnation": stagnation,
            }
        )
        temperature *= cooling

    selector.end_episode()
    runtime = perf_counter() - started
    best_audit = audit_solution(instance, best)
    if not best_audit.feasible:
        raise RuntimeError("best solution is infeasible")
    return SearchResult(
        initial_solution=initial,
        best_solution=best,
        initial_objective=initial_objective,
        best_objective=best_objective,
        runtime_s=runtime,
        time_to_best_s=time_to_best,
        iteration_to_best=iteration_to_best,
        feasibility_rate=feasible_steps / iterations,
        operator_usage=dict(usage),
        trace=trace,
    )

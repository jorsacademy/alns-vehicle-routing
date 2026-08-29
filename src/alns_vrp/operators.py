from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .data import CVRPInstance
from .solution import Solution, insertion_delta, removal_saving

DestroyOperator = Callable[[CVRPInstance, Solution, int, random.Random], tuple[Solution, list[int]]]
RepairOperator = Callable[[CVRPInstance, Solution, list[int], random.Random], Solution]


def _remove_customers(solution: Solution, customers: set[int]) -> Solution:
    routes = [[c for c in route if c not in customers] for route in solution.routes]
    return Solution([route for route in routes if route])


def random_removal(
    instance: CVRPInstance,
    solution: Solution,
    q: int,
    rng: random.Random,
) -> tuple[Solution, list[int]]:
    del instance
    customers = solution.customers()
    removed = rng.sample(customers, k=min(q, len(customers)))
    return _remove_customers(solution, set(removed)), removed


def worst_removal(
    instance: CVRPInstance,
    solution: Solution,
    q: int,
    rng: random.Random,
) -> tuple[Solution, list[int]]:
    scored: list[tuple[float, float, int]] = []
    for route in solution.routes:
        for position, customer in enumerate(route):
            saving = removal_saving(instance, route, position)
            scored.append((saving, rng.random(), customer))
    scored.sort(reverse=True)
    removed = [customer for _, _, customer in scored[: min(q, len(scored))]]
    return _remove_customers(solution, set(removed)), removed


def related_removal(
    instance: CVRPInstance,
    solution: Solution,
    q: int,
    rng: random.Random,
) -> tuple[Solution, list[int]]:
    customers = solution.customers()
    if not customers:
        return solution.copy(), []

    seed = rng.choice(customers)
    max_distance = max(instance.distance(i, j) for i in customers for j in customers) or 1.0
    max_demand = max(instance.demands[1:]) or 1

    def relatedness(customer: int) -> tuple[float, float]:
        spatial = instance.distance(seed, customer) / max_distance
        demand = abs(instance.demands[seed] - instance.demands[customer]) / max_demand
        return (0.8 * spatial + 0.2 * demand, rng.random())

    ordered = sorted(customers, key=relatedness)
    removed = ordered[: min(q, len(ordered))]
    return _remove_customers(solution, set(removed)), removed


def _insertion_options(
    instance: CVRPInstance,
    solution: Solution,
    customer: int,
) -> list[tuple[float, int, int]]:
    options: list[tuple[float, int, int]] = []
    demand = instance.demands[customer]
    for route_index, route in enumerate(solution.routes):
        if instance.route_load(route) + demand > instance.capacity:
            continue
        for position in range(len(route) + 1):
            options.append((insertion_delta(instance, route, position, customer), route_index, position))
    if len(solution.routes) < instance.max_vehicles:
        options.append((2.0 * instance.distance(0, customer), len(solution.routes), 0))
    return sorted(options)


def _insert(solution: Solution, customer: int, route_index: int, position: int) -> None:
    if route_index == len(solution.routes):
        solution.routes.append([customer])
    else:
        solution.routes[route_index].insert(position, customer)


def greedy_repair(
    instance: CVRPInstance,
    partial: Solution,
    removed: list[int],
    rng: random.Random,
) -> Solution:
    del rng
    solution = partial.copy()
    pending = set(removed)
    while pending:
        best: tuple[float, int, int, int] | None = None
        for customer in sorted(pending):
            options = _insertion_options(instance, solution, customer)
            if not options:
                continue
            delta, route_index, position = options[0]
            candidate = (delta, customer, route_index, position)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise RuntimeError("greedy repair could not restore feasibility")
        _, customer, route_index, position = best
        _insert(solution, customer, route_index, position)
        pending.remove(customer)
    return solution.normalized()


def regret2_repair(
    instance: CVRPInstance,
    partial: Solution,
    removed: list[int],
    rng: random.Random,
) -> Solution:
    solution = partial.copy()
    pending = set(removed)
    while pending:
        choice: tuple[float, float, float, int, int, int] | None = None
        for customer in sorted(pending):
            options = _insertion_options(instance, solution, customer)
            if not options:
                continue
            best_cost, route_index, position = options[0]
            second_cost = options[1][0] if len(options) > 1 else best_cost + 2.0 * instance.distance(0, customer)
            regret = second_cost - best_cost
            candidate = (regret, -best_cost, rng.random(), customer, route_index, position)
            if choice is None or candidate > choice:
                choice = candidate
        if choice is None:
            raise RuntimeError("regret repair could not restore feasibility")
        _, _, _, customer, route_index, position = choice
        _insert(solution, customer, route_index, position)
        pending.remove(customer)
    return solution.normalized()


@dataclass
class OperatorState:
    name: str
    weight: float = 1.0
    score: float = 0.0
    uses: int = 0


class AdaptivePool:
    def __init__(self, names: list[str], reaction: float) -> None:
        if not names:
            raise ValueError("operator pool cannot be empty")
        if not 0.0 < reaction <= 1.0:
            raise ValueError("reaction must be in (0, 1]")
        self.states = {name: OperatorState(name=name) for name in names}
        self.reaction = reaction

    def choose(self, rng: random.Random) -> str:
        states = list(self.states.values())
        total = sum(state.weight for state in states)
        threshold = rng.random() * total
        cumulative = 0.0
        for state in states:
            cumulative += state.weight
            if threshold <= cumulative:
                state.uses += 1
                return state.name
        states[-1].uses += 1
        return states[-1].name

    def reward(self, name: str, amount: float) -> None:
        self.states[name].score += amount

    def update(self) -> None:
        for state in self.states.values():
            if state.uses > 0:
                performance = state.score / state.uses
                state.weight = (1.0 - self.reaction) * state.weight + self.reaction * max(performance, 1e-6)
            state.score = 0.0
            state.uses = 0

    def weights(self) -> dict[str, float]:
        return {name: state.weight for name, state in self.states.items()}


DESTROY_OPERATORS: dict[str, DestroyOperator] = {
    "random": random_removal,
    "worst": worst_removal,
    "related": related_removal,
}

REPAIR_OPERATORS: dict[str, RepairOperator] = {
    "greedy": greedy_repair,
    "regret2": regret2_repair,
}

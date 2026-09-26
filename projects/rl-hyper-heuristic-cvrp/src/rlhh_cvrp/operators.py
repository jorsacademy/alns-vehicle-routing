from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .problem import CVRPInstance, Solution, audit_solution, route_load, solution_distance

Operator = Callable[[CVRPInstance, Solution, np.random.Generator], Solution]


def _mutable(solution: Solution) -> list[list[int]]:
    return [list(route) for route in solution]


def _finish(instance: CVRPInstance, routes: list[list[int]]) -> Solution:
    candidate = tuple(tuple(route) for route in routes if route)
    audit = audit_solution(instance, candidate)
    if not audit.feasible:
        raise RuntimeError(f"operator produced infeasible solution: {audit}")
    return candidate


def two_opt(instance: CVRPInstance, solution: Solution, rng: np.random.Generator) -> Solution:
    eligible = [idx for idx, route in enumerate(solution) if len(route) >= 3]
    if not eligible:
        return solution
    routes = _mutable(solution)
    ridx = int(rng.choice(eligible))
    i, j = sorted(rng.choice(len(routes[ridx]), size=2, replace=False).tolist())
    if i == j:
        return solution
    routes[ridx][i : j + 1] = reversed(routes[ridx][i : j + 1])
    return _finish(instance, routes)


def swap(instance: CVRPInstance, solution: Solution, rng: np.random.Generator) -> Solution:
    positions = [(r, p) for r, route in enumerate(solution) for p in range(len(route))]
    if len(positions) < 2:
        return solution
    for _ in range(24):
        a_idx, b_idx = rng.choice(len(positions), size=2, replace=False)
        ra, pa = positions[int(a_idx)]
        rb, pb = positions[int(b_idx)]
        routes = _mutable(solution)
        a, b = routes[ra][pa], routes[rb][pb]
        routes[ra][pa], routes[rb][pb] = b, a
        if ra != rb:
            if route_load(instance, routes[ra]) > instance.capacity:
                continue
            if route_load(instance, routes[rb]) > instance.capacity:
                continue
        return _finish(instance, routes)
    return solution


def relocate(instance: CVRPInstance, solution: Solution, rng: np.random.Generator) -> Solution:
    positions = [(r, p) for r, route in enumerate(solution) for p in range(len(route))]
    if not positions:
        return solution
    for _ in range(24):
        source_r, source_p = positions[int(rng.integers(0, len(positions)))]
        routes = _mutable(solution)
        customer = routes[source_r][source_p]
        target_r = int(rng.integers(0, len(routes) + 1))
        if target_r == len(routes):
            routes.append([])
        if target_r != source_r and (
            route_load(instance, routes[target_r]) + int(instance.demands[customer])
            > instance.capacity
        ):
            continue
        routes[source_r].pop(source_p)
        target_pos = int(rng.integers(0, len(routes[target_r]) + 1))
        routes[target_r].insert(target_pos, customer)
        return _finish(instance, routes)
    return solution


def _best_insertion(
    instance: CVRPInstance, routes: list[list[int]], customer: int
) -> tuple[int, int]:
    best: tuple[float, int, int] | None = None
    base_solution = tuple(tuple(route) for route in routes if route)
    base_distance = solution_distance(instance, base_solution)
    for ridx, route in enumerate(routes):
        if route_load(instance, route) + int(instance.demands[customer]) > instance.capacity:
            continue
        for pos in range(len(route) + 1):
            candidate_routes = [list(r) for r in routes]
            candidate_routes[ridx].insert(pos, customer)
            candidate = tuple(tuple(r) for r in candidate_routes if r)
            delta = solution_distance(instance, candidate) - base_distance
            item = (delta, ridx, pos)
            if best is None or item < best:
                best = item
    if best is None:
        return len(routes), 0
    return best[1], best[2]


def ruin_recreate(
    instance: CVRPInstance, solution: Solution, rng: np.random.Generator
) -> Solution:
    customers = [node for route in solution for node in route]
    if len(customers) < 2:
        return solution
    remove_count = min(len(customers), max(2, int(np.ceil(0.2 * len(customers)))))
    removed = [int(x) for x in rng.choice(customers, size=remove_count, replace=False)]
    removed_set = set(removed)
    routes = [[node for node in route if node not in removed_set] for route in solution]
    routes = [route for route in routes if route]
    rng.shuffle(removed)
    for customer in removed:
        ridx, pos = _best_insertion(instance, routes, customer)
        if ridx == len(routes):
            routes.append([customer])
        else:
            routes[ridx].insert(pos, customer)
    return _finish(instance, routes)


def random_perturbation(
    instance: CVRPInstance, solution: Solution, rng: np.random.Generator
) -> Solution:
    candidate = solution
    moves = int(rng.integers(2, 5))
    for _ in range(moves):
        if rng.random() < 0.5:
            candidate = swap(instance, candidate, rng)
        else:
            candidate = relocate(instance, candidate, rng)
    return candidate


OPERATORS: dict[str, Operator] = {
    "two_opt": two_opt,
    "swap": swap,
    "relocate": relocate,
    "ruin_recreate": ruin_recreate,
    "random_perturbation": random_perturbation,
}
OPERATOR_NAMES = tuple(OPERATORS)

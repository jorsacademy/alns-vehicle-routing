from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil

import numpy as np

Route = tuple[int, ...]
Solution = tuple[Route, ...]


@dataclass(frozen=True)
class CVRPInstance:
    instance_id: str
    seed: int
    coordinates: np.ndarray
    demands: np.ndarray
    capacity: int

    @property
    def n_customers(self) -> int:
        return len(self.demands) - 1

    def validate(self) -> None:
        if self.coordinates.shape != (self.n_customers + 1, 2):
            raise ValueError("coordinates must include depot plus all customers")
        if self.demands.shape != (self.n_customers + 1,):
            raise ValueError("demands must include depot plus all customers")
        if int(self.demands[0]) != 0:
            raise ValueError("depot demand must be zero")
        if np.any(self.demands[1:] <= 0):
            raise ValueError("customer demands must be positive")
        if np.any(self.demands[1:] > self.capacity):
            raise ValueError("every customer must individually fit in a vehicle")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")


@dataclass(frozen=True)
class SolutionAudit:
    feasible: bool
    objective: float
    duplicate_count: int
    missing_count: int
    invalid_count: int
    capacity_violation: float
    empty_route_count: int

    @property
    def violation_count(self) -> int:
        return (
            self.duplicate_count
            + self.missing_count
            + self.invalid_count
            + int(self.capacity_violation > 0)
            + self.empty_route_count
        )


def generate_instance(
    *,
    n_customers: int,
    seed: int,
    target_routes: int = 3,
    instance_id: str | None = None,
) -> CVRPInstance:
    if n_customers < 2:
        raise ValueError("n_customers must be at least two")
    if target_routes <= 0:
        raise ValueError("target_routes must be positive")
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(0.0, 100.0, size=(n_customers + 1, 2))
    demands = np.zeros(n_customers + 1, dtype=np.int64)
    demands[1:] = rng.integers(1, 10, size=n_customers)
    capacity = max(
        int(demands[1:].max()),
        ceil(float(demands[1:].sum()) / target_routes * 1.15),
    )
    instance = CVRPInstance(
        instance_id=instance_id or f"cvrp-n{n_customers}-s{seed}",
        seed=seed,
        coordinates=coordinates,
        demands=demands,
        capacity=capacity,
    )
    instance.validate()
    return instance


def distance_matrix(instance: CVRPInstance) -> np.ndarray:
    delta = instance.coordinates[:, None, :] - instance.coordinates[None, :, :]
    return np.sqrt(np.square(delta).sum(axis=2))


def route_load(instance: CVRPInstance, route: Iterable[int]) -> int:
    return int(sum(int(instance.demands[node]) for node in route))


def route_distance(instance: CVRPInstance, route: Route) -> float:
    if not route:
        return 0.0
    d = distance_matrix(instance)
    total = d[0, route[0]] + d[route[-1], 0]
    total += sum(d[a, b] for a, b in zip(route[:-1], route[1:], strict=True))
    return float(total)


def solution_distance(instance: CVRPInstance, solution: Solution) -> float:
    return float(sum(route_distance(instance, route) for route in solution))


def audit_solution(instance: CVRPInstance, solution: Solution) -> SolutionAudit:
    customers = [node for route in solution for node in route]
    valid = [node for node in customers if 1 <= node <= instance.n_customers]
    invalid_count = len(customers) - len(valid)
    unique = set(valid)
    duplicate_count = len(valid) - len(unique)
    missing_count = instance.n_customers - len(unique)
    capacity_violation = float(
        sum(max(0, route_load(instance, route) - instance.capacity) for route in solution)
    )
    empty_route_count = sum(1 for route in solution if not route)
    feasible = (
        invalid_count == 0
        and duplicate_count == 0
        and missing_count == 0
        and capacity_violation == 0
        and empty_route_count == 0
    )
    objective = solution_distance(instance, solution) if feasible else float("inf")
    return SolutionAudit(
        feasible=feasible,
        objective=objective,
        duplicate_count=duplicate_count,
        missing_count=missing_count,
        invalid_count=invalid_count,
        capacity_violation=capacity_violation,
        empty_route_count=empty_route_count,
    )


def greedy_initial_solution(instance: CVRPInstance) -> Solution:
    d = distance_matrix(instance)
    remaining = set(range(1, instance.n_customers + 1))
    routes: list[Route] = []
    while remaining:
        route: list[int] = []
        load = 0
        current = 0
        while True:
            feasible = [
                customer
                for customer in remaining
                if load + int(instance.demands[customer]) <= instance.capacity
            ]
            if not feasible:
                break
            nxt = min(feasible, key=lambda customer: (d[current, customer], customer))
            route.append(nxt)
            remaining.remove(nxt)
            load += int(instance.demands[nxt])
            current = nxt
        if not route:
            raise RuntimeError("greedy construction could not place a remaining customer")
        routes.append(tuple(route))
    solution = tuple(routes)
    if not audit_solution(instance, solution).feasible:
        raise RuntimeError("greedy constructor created an infeasible solution")
    return solution


def solution_statistics(instance: CVRPInstance, solution: Solution) -> dict[str, float]:
    loads = np.asarray([route_load(instance, route) for route in solution], dtype=float)
    route_lengths = np.asarray(
        [route_distance(instance, route) for route in solution], dtype=float
    )
    load_mean = float(loads.mean()) if loads.size else 0.0
    return {
        "route_count": float(len(solution)),
        "load_mean_fraction": load_mean / instance.capacity if loads.size else 0.0,
        "load_cv": float(loads.std() / max(load_mean, 1e-9)) if loads.size else 0.0,
        "route_length_cv": (
            float(route_lengths.std() / max(float(route_lengths.mean()), 1e-9))
            if route_lengths.size
            else 0.0
        ),
        "near_capacity_fraction": (
            float(np.mean(loads >= 0.9 * instance.capacity)) if loads.size else 0.0
        ),
    }

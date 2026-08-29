from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .data import CVRPInstance


@dataclass
class Solution:
    routes: list[list[int]]

    def copy(self) -> "Solution":
        return Solution([route.copy() for route in self.routes])

    def normalized(self) -> "Solution":
        return Solution([route.copy() for route in self.routes if route])

    def customers(self) -> list[int]:
        return [customer for route in self.routes for customer in route]

    def total_distance(self, instance: CVRPInstance) -> float:
        total = 0.0
        for route in self.routes:
            if not route:
                continue
            previous = 0
            for customer in route:
                total += instance.distance(previous, customer)
                previous = customer
            total += instance.distance(previous, 0)
        return total

    def route_loads(self, instance: CVRPInstance) -> tuple[int, ...]:
        return tuple(instance.route_load(route) for route in self.routes)

    def is_feasible(self, instance: CVRPInstance, require_complete: bool = True) -> bool:
        if len([route for route in self.routes if route]) > instance.max_vehicles:
            return False
        if any(instance.route_load(route) > instance.capacity for route in self.routes):
            return False
        seen = self.customers()
        if len(seen) != len(set(seen)):
            return False
        if any(customer <= 0 or customer > instance.n_customers for customer in seen):
            return False
        if require_complete:
            return set(seen) == set(instance.customers)
        return True


def route_distance(instance: CVRPInstance, route: Iterable[int]) -> float:
    sequence = list(route)
    if not sequence:
        return 0.0
    total = instance.distance(0, sequence[0])
    total += sum(instance.distance(a, b) for a, b in zip(sequence, sequence[1:]))
    total += instance.distance(sequence[-1], 0)
    return total


def insertion_delta(instance: CVRPInstance, route: list[int], position: int, customer: int) -> float:
    before = 0 if position == 0 else route[position - 1]
    after = 0 if position == len(route) else route[position]
    return (
        instance.distance(before, customer)
        + instance.distance(customer, after)
        - instance.distance(before, after)
    )


def removal_saving(instance: CVRPInstance, route: list[int], position: int) -> float:
    customer = route[position]
    before = 0 if position == 0 else route[position - 1]
    after = 0 if position == len(route) - 1 else route[position + 1]
    return (
        instance.distance(before, customer)
        + instance.distance(customer, after)
        - instance.distance(before, after)
    )

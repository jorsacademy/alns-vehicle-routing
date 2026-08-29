from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable


@dataclass(frozen=True)
class CVRPInstance:
    coordinates: tuple[tuple[float, float], ...]
    demands: tuple[int, ...]
    capacity: int
    max_vehicles: int

    def __post_init__(self) -> None:
        if len(self.coordinates) != len(self.demands):
            raise ValueError("coordinates and demands must have equal length")
        if len(self.coordinates) < 2:
            raise ValueError("instance must contain a depot and at least one customer")
        if self.demands[0] != 0:
            raise ValueError("depot demand must be zero")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.max_vehicles <= 0:
            raise ValueError("max_vehicles must be positive")
        if any(d < 0 for d in self.demands):
            raise ValueError("demands must be nonnegative")
        if any(d > self.capacity for d in self.demands[1:]):
            raise ValueError("every customer demand must fit in one vehicle")
        if sum(self.demands) > self.capacity * self.max_vehicles:
            raise ValueError("fleet capacity is insufficient for total demand")
        for point in self.coordinates:
            if len(point) != 2 or not all(math.isfinite(v) for v in point):
                raise ValueError("coordinates must be finite 2D points")

    @property
    def n_customers(self) -> int:
        return len(self.coordinates) - 1

    @property
    def customers(self) -> range:
        return range(1, len(self.coordinates))

    def distance(self, i: int, j: int) -> float:
        xi, yi = self.coordinates[i]
        xj, yj = self.coordinates[j]
        return math.hypot(xi - xj, yi - yj)

    def route_load(self, route: Iterable[int]) -> int:
        return sum(self.demands[i] for i in route)


def demo_instance(seed: int = 2026, n_customers: int = 36) -> CVRPInstance:
    """Create a deterministic synthetic Euclidean CVRP instance."""
    if n_customers < 4:
        raise ValueError("n_customers must be at least 4")

    rng = random.Random(seed)
    depot = (50.0, 50.0)
    centers = ((22.0, 26.0), (76.0, 28.0), (28.0, 76.0), (74.0, 74.0))
    coordinates: list[tuple[float, float]] = [depot]
    demands: list[int] = [0]

    for customer in range(n_customers):
        cx, cy = centers[customer % len(centers)]
        x = min(98.0, max(2.0, rng.gauss(cx, 8.0)))
        y = min(98.0, max(2.0, rng.gauss(cy, 8.0)))
        coordinates.append((x, y))
        demands.append(rng.randint(2, 8))

    total_demand = sum(demands)
    capacity = 32
    max_vehicles = max(6, math.ceil(total_demand / capacity) + 1)
    return CVRPInstance(tuple(coordinates), tuple(demands), capacity, max_vehicles)

from __future__ import annotations

from .data import CVRPInstance
from .solution import Solution, route_distance


def two_opt(solution: Solution, instance: CVRPInstance) -> Solution:
    """Apply deterministic best-improvement 2-opt independently to every route."""
    improved = solution.copy()
    for route_index, route in enumerate(improved.routes):
        if len(route) < 4:
            continue
        while True:
            base = route_distance(instance, route)
            best_delta = 0.0
            best_route: list[int] | None = None
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route) + 1):
                    if i == 0 and j == len(route):
                        continue
                    candidate = route[:i] + list(reversed(route[i:j])) + route[j:]
                    delta = route_distance(instance, candidate) - base
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_route = candidate
            if best_route is None:
                break
            route = best_route
            improved.routes[route_index] = route
    return improved

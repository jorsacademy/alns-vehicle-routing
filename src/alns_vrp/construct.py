from __future__ import annotations

from .data import CVRPInstance
from .solution import Solution, insertion_delta


def greedy_initial_solution(instance: CVRPInstance) -> Solution:
    """Build a deterministic capacity-feasible solution with global cheapest insertion."""
    routes: list[list[int]] = []
    unassigned = set(instance.customers)

    while unassigned:
        best: tuple[float, int, int, int] | None = None
        for customer in sorted(unassigned):
            demand = instance.demands[customer]
            for route_index, route in enumerate(routes):
                if instance.route_load(route) + demand > instance.capacity:
                    continue
                for position in range(len(route) + 1):
                    delta = insertion_delta(instance, route, position, customer)
                    candidate = (delta, customer, route_index, position)
                    if best is None or candidate < best:
                        best = candidate

            if len(routes) < instance.max_vehicles:
                delta = 2.0 * instance.distance(0, customer)
                candidate = (delta, customer, len(routes), 0)
                if best is None or candidate < best:
                    best = candidate

        if best is None:
            raise RuntimeError("could not construct a feasible initial solution")

        _, customer, route_index, position = best
        if route_index == len(routes):
            routes.append([customer])
        else:
            routes[route_index].insert(position, customer)
        unassigned.remove(customer)

    result = Solution(routes).normalized()
    if not result.is_feasible(instance):
        raise RuntimeError("initial solution construction produced an infeasible solution")
    return result

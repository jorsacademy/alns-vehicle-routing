from __future__ import annotations

from functools import lru_cache
import math

from .data import CVRPInstance


def exact_cvrp_cost(instance: CVRPInstance, max_customers: int = 11) -> float:
    """Return the exact CVRP cost for small instances using subset dynamic programming."""
    n = instance.n_customers
    if n > max_customers:
        raise ValueError(f"exact solver is limited to {max_customers} customers")

    size = 1 << n
    subset_demand = [0] * size
    for mask in range(1, size):
        bit = mask & -mask
        customer_index = bit.bit_length() - 1
        subset_demand[mask] = subset_demand[mask ^ bit] + instance.demands[customer_index + 1]

    tsp = [[math.inf] * n for _ in range(size)]
    for j in range(n):
        tsp[1 << j][j] = instance.distance(0, j + 1)

    for mask in range(1, size):
        if subset_demand[mask] > instance.capacity:
            continue
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            prev_mask = mask ^ (1 << j)
            if prev_mask == 0:
                continue
            best = math.inf
            for k in range(n):
                if prev_mask & (1 << k):
                    best = min(best, tsp[prev_mask][k] + instance.distance(k + 1, j + 1))
            tsp[mask][j] = best

    route_cost = [math.inf] * size
    route_cost[0] = 0.0
    for mask in range(1, size):
        if subset_demand[mask] <= instance.capacity:
            route_cost[mask] = min(
                tsp[mask][j] + instance.distance(j + 1, 0)
                for j in range(n)
                if mask & (1 << j)
            )

    @lru_cache(maxsize=None)
    def partition(mask: int, vehicles_left: int) -> float:
        if mask == 0:
            return 0.0
        if vehicles_left == 0:
            return math.inf
        anchor = mask & -mask
        best = math.inf
        subset = mask
        while subset:
            if subset & anchor and route_cost[subset] < math.inf:
                best = min(best, route_cost[subset] + partition(mask ^ subset, vehicles_left - 1))
            subset = (subset - 1) & mask
        return best

    optimum = partition(size - 1, instance.max_vehicles)
    if not math.isfinite(optimum):
        raise RuntimeError("exact solver found no feasible solution")
    return optimum

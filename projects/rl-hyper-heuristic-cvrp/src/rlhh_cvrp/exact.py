from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import CVRPInstance, distance_matrix


@dataclass(frozen=True)
class ExactResult:
    objective: float
    certified_optimal: bool
    states_evaluated: int


def exact_cvrp(instance: CVRPInstance, *, max_customers: int = 10) -> ExactResult:
    """Exact small-CVRP oracle using subset route DP plus set partitioning."""
    n = instance.n_customers
    if n > max_customers:
        raise ValueError(f"exact oracle limited to {max_customers} customers")
    d = distance_matrix(instance)
    size = 1 << n
    demand_sum = np.zeros(size, dtype=np.int64)
    for mask in range(1, size):
        bit = mask & -mask
        idx = bit.bit_length() - 1
        demand_sum[mask] = demand_sum[mask ^ bit] + int(instance.demands[idx + 1])

    dp = np.full((size, n), np.inf, dtype=float)
    states = 0
    for j in range(n):
        dp[1 << j, j] = d[0, j + 1]
        states += 1
    for mask in range(1, size):
        for last in range(n):
            if not (mask & (1 << last)) or not np.isfinite(dp[mask, last]):
                continue
            nxt_mask = (size - 1) ^ mask
            while nxt_mask:
                bit = nxt_mask & -nxt_mask
                nxt = bit.bit_length() - 1
                new_mask = mask | bit
                value = dp[mask, last] + d[last + 1, nxt + 1]
                if value < dp[new_mask, nxt]:
                    dp[new_mask, nxt] = value
                nxt_mask ^= bit
                states += 1

    route_cost = np.full(size, np.inf, dtype=float)
    route_cost[0] = 0.0
    for mask in range(1, size):
        if demand_sum[mask] > instance.capacity:
            continue
        endings = [
            dp[mask, j] + d[j + 1, 0]
            for j in range(n)
            if mask & (1 << j) and np.isfinite(dp[mask, j])
        ]
        if endings:
            route_cost[mask] = min(endings)

    best = np.full(size, np.inf, dtype=float)
    best[0] = 0.0
    for mask in range(1, size):
        anchor = mask & -mask
        sub = mask
        while sub:
            if sub & anchor and np.isfinite(route_cost[sub]):
                candidate = route_cost[sub] + best[mask ^ sub]
                if candidate < best[mask]:
                    best[mask] = candidate
            sub = (sub - 1) & mask
            states += 1
    value = float(best[size - 1])
    if not np.isfinite(value):
        raise RuntimeError("exact oracle found no feasible partition")
    return ExactResult(objective=value, certified_optimal=True, states_evaluated=states)

from itertools import permutations
from math import isclose

import numpy as np

from rlhh_cvrp.exact import exact_cvrp
from rlhh_cvrp.problem import CVRPInstance, distance_matrix


def test_exact_oracle_matches_bruteforce_tsp_when_one_route_is_feasible() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    instance = CVRPInstance(
        instance_id="square",
        seed=0,
        coordinates=coords,
        demands=np.array([0, 1, 1, 1]),
        capacity=3,
    )
    d = distance_matrix(instance)
    brute = min(
        d[0, p[0]] + d[p[0], p[1]] + d[p[1], p[2]] + d[p[2], 0]
        for p in permutations((1, 2, 3))
    )
    exact = exact_cvrp(instance, max_customers=4)
    assert exact.certified_optimal
    assert isclose(exact.objective, brute, rel_tol=1e-10, abs_tol=1e-10)


def test_exact_oracle_respects_capacity_partition() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    instance = CVRPInstance(
        instance_id="capacity",
        seed=0,
        coordinates=coords,
        demands=np.array([0, 2, 2]),
        capacity=2,
    )
    exact = exact_cvrp(instance, max_customers=3)
    assert isclose(exact.objective, 4.0, rel_tol=1e-10, abs_tol=1e-10)

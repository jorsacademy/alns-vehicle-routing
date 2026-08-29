import random

import pytest

from alns_vrp.construct import greedy_initial_solution
from alns_vrp.data import demo_instance
from alns_vrp.operators import DESTROY_OPERATORS, REPAIR_OPERATORS, AdaptivePool


@pytest.mark.parametrize("destroy_name", sorted(DESTROY_OPERATORS))
@pytest.mark.parametrize("repair_name", sorted(REPAIR_OPERATORS))
def test_every_operator_pair_restores_a_feasible_solution(destroy_name, repair_name):
    instance = demo_instance(seed=11, n_customers=16)
    initial = greedy_initial_solution(instance)
    rng = random.Random(99)
    partial, removed = DESTROY_OPERATORS[destroy_name](instance, initial, 4, rng)
    repaired = REPAIR_OPERATORS[repair_name](instance, partial, removed, rng)
    assert repaired.is_feasible(instance)
    assert set(repaired.customers()) == set(instance.customers)


def test_adaptive_pool_rewards_successful_operator():
    pool = AdaptivePool(["a", "b"], reaction=1.0)
    pool.states["a"].uses = 2
    pool.states["a"].score = 10.0
    pool.states["b"].uses = 2
    pool.states["b"].score = 2.0
    pool.update()
    assert pool.weights()["a"] > pool.weights()["b"]

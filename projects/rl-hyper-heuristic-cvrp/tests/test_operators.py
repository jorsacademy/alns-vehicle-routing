import numpy as np
import pytest

from rlhh_cvrp.operators import OPERATORS
from rlhh_cvrp.problem import audit_solution, generate_instance, greedy_initial_solution


@pytest.mark.parametrize("operator_name", sorted(OPERATORS))
def test_each_operator_preserves_cvrp_feasibility(operator_name: str) -> None:
    instance = generate_instance(n_customers=18, seed=73, target_routes=4)
    solution = greedy_initial_solution(instance)
    operator = OPERATORS[operator_name]
    for seed in range(10):
        candidate = operator(instance, solution, np.random.default_rng(seed))
        audit = audit_solution(instance, candidate)
        assert audit.feasible, (operator_name, audit)
        assert audit.violation_count == 0


def test_operator_is_deterministic_given_rng_seed() -> None:
    instance = generate_instance(n_customers=15, seed=9, target_routes=3)
    solution = greedy_initial_solution(instance)
    for operator in OPERATORS.values():
        first = operator(instance, solution, np.random.default_rng(99))
        second = operator(instance, solution, np.random.default_rng(99))
        assert first == second

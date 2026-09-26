import numpy as np

from rlhh_cvrp.problem import audit_solution, generate_instance, greedy_initial_solution


def test_generation_and_initial_solution_are_deterministic_and_feasible() -> None:
    first = generate_instance(n_customers=10, seed=123, target_routes=3)
    second = generate_instance(n_customers=10, seed=123, target_routes=3)
    assert np.array_equal(first.coordinates, second.coordinates)
    assert np.array_equal(first.demands, second.demands)
    assert first.capacity == second.capacity
    solution = greedy_initial_solution(first)
    audit = audit_solution(first, solution)
    assert audit.feasible
    assert audit.objective > 0.0


def test_audit_detects_duplicate_and_missing_customer() -> None:
    instance = generate_instance(n_customers=5, seed=4, target_routes=2)
    audit = audit_solution(instance, ((1, 2, 2), (3, 4)))
    assert not audit.feasible
    assert audit.duplicate_count == 1
    assert audit.missing_count == 1

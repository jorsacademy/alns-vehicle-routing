from rlhh_cvrp.operators import OPERATOR_NAMES
from rlhh_cvrp.problem import audit_solution, generate_instance
from rlhh_cvrp.search import run_search
from rlhh_cvrp.selectors import RoundRobinSelector


def test_search_is_reproducible_and_never_uses_policy_as_constructor() -> None:
    instance = generate_instance(n_customers=12, seed=200, target_routes=3)
    first = run_search(
        instance,
        RoundRobinSelector(len(OPERATOR_NAMES)),
        iterations=30,
        search_seed=99,
    )
    second = run_search(
        instance,
        RoundRobinSelector(len(OPERATOR_NAMES)),
        iterations=30,
        search_seed=99,
    )
    assert first.initial_solution == second.initial_solution
    assert first.best_solution == second.best_solution
    assert first.best_objective == second.best_objective
    assert audit_solution(instance, first.best_solution).feasible
    assert first.feasibility_rate == 1.0
    assert sum(first.operator_usage.values()) == 30

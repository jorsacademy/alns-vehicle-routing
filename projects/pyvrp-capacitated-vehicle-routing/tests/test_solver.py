import pytest

from pyvrp_cvrp.model import demo_instance, solve_instance


def test_solver_returns_feasible_cvrp_solution():
    instance = demo_instance()
    result = solve_instance(instance, max_iterations=1_000, seed=7)

    assert result.is_feasible()
    assert result.cost() > 0
    assert result.best.distance() > 0
    assert 1 <= result.best.num_routes() <= instance.num_vehicles
    assert result.best.num_clients() == len(instance.customers)
    assert not result.best.has_excess_load()


def test_solver_rejects_non_positive_iteration_budget():
    with pytest.raises(ValueError, match="max_iterations"):
        solve_instance(demo_instance(), max_iterations=0)

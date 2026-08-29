from alns_vrp.alns import ALNSConfig, solve_alns
from alns_vrp.construct import greedy_initial_solution
from alns_vrp.data import CVRPInstance, demo_instance
from alns_vrp.exact import exact_cvrp_cost


def tiny_instance() -> CVRPInstance:
    return CVRPInstance(
        coordinates=(
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (0.0, 2.0),
            (2.0, 2.0),
            (1.0, 2.0),
        ),
        demands=(0, 1, 1, 1, 1, 1, 1),
        capacity=3,
        max_vehicles=2,
    )


def test_exact_solver_known_small_case_is_finite_and_not_above_greedy():
    instance = tiny_instance()
    exact = exact_cvrp_cost(instance)
    greedy = greedy_initial_solution(instance).total_distance(instance)
    assert exact <= greedy + 1e-9
    assert exact > 0.0


def test_alns_is_reproducible_and_preserves_feasibility():
    instance = demo_instance(seed=123, n_customers=18)
    config = ALNSConfig(iterations=250, segment_length=50)
    a = solve_alns(instance, config, seed=321)
    b = solve_alns(instance, config, seed=321)
    assert abs(a.best_cost - b.best_cost) < 1e-12
    assert a.best_solution.routes == b.best_solution.routes
    assert a.best_solution.is_feasible(instance)


def test_alns_never_loses_best_so_far_and_improves_or_matches_initial():
    instance = demo_instance(seed=4, n_customers=20)
    result = solve_alns(instance, ALNSConfig(iterations=300, segment_length=50), seed=4)
    best_history = [record.best_cost for record in result.records]
    assert all(b <= a + 1e-12 for a, b in zip(best_history, best_history[1:]))
    assert result.best_cost <= result.initial_cost + 1e-9


def test_exact_solver_and_alns_match_known_two_route_optimum():
    instance = CVRPInstance(
        coordinates=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (0.0, 2.0)),
        demands=(0, 1, 1, 1, 1),
        capacity=2,
        max_vehicles=2,
    )
    exact = exact_cvrp_cost(instance)
    result = solve_alns(
        instance,
        ALNSConfig(iterations=100, min_removal=1, max_removal_fraction=0.5, segment_length=20),
        seed=0,
    )
    assert abs(exact - 8.0) < 1e-12
    assert abs(result.best_cost - exact) < 1e-12

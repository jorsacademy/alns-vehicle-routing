import pytest

from alns_vrp.data import CVRPInstance, demo_instance
from alns_vrp.solution import Solution


def test_instance_validation_rejects_insufficient_fleet_capacity():
    with pytest.raises(ValueError):
        CVRPInstance(
            coordinates=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
            demands=(0, 6, 6),
            capacity=10,
            max_vehicles=1,
        )


def test_demo_instance_is_reproducible():
    a = demo_instance(seed=7, n_customers=12)
    b = demo_instance(seed=7, n_customers=12)
    assert a == b


def test_solution_feasibility_detects_duplicates():
    instance = CVRPInstance(
        coordinates=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        demands=(0, 1, 1),
        capacity=2,
        max_vehicles=1,
    )
    assert not Solution([[1, 1]]).is_feasible(instance)
    assert Solution([[1, 2]]).is_feasible(instance)

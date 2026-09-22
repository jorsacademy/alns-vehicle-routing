import pytest

from pyvrp_cvrp.model import (
    CVRPInstance,
    Customer,
    build_model,
    demo_instance,
    manhattan_distance,
)


def test_demo_instance_capacity_accounting():
    instance = demo_instance()

    assert instance.total_demand == 27
    assert instance.fleet_capacity == 30


def test_build_model_contains_expected_entities():
    model = build_model(demo_instance())

    assert len(model.depots) == 1
    assert len(model.clients) == 6
    assert len(model.vehicle_types) == 1
    assert len(model.locations) == 7


def test_manhattan_distance_is_integer_scaled():
    assert manhattan_distance((0.0, 0.0), (3.2, 4.4)) == 8


def test_customer_requires_positive_demand():
    with pytest.raises(ValueError, match="positive"):
        Customer(1, 2, 0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"customers": ()}, "at least one customer"),
        ({"num_vehicles": 0}, "num_vehicles"),
        ({"vehicle_capacity": 0}, "vehicle_capacity"),
    ],
)
def test_instance_rejects_invalid_basic_configuration(kwargs, message):
    base = {
        "depot": (0, 0),
        "customers": (Customer(1, 1, 1),),
        "num_vehicles": 1,
        "vehicle_capacity": 2,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        CVRPInstance(**base)


def test_instance_rejects_customer_larger_than_vehicle():
    with pytest.raises(ValueError, match="customer demand"):
        CVRPInstance(
            depot=(0, 0),
            customers=(Customer(1, 1, 6),),
            num_vehicles=2,
            vehicle_capacity=5,
        )


def test_instance_rejects_demand_larger_than_fleet_capacity():
    with pytest.raises(ValueError, match="total demand"):
        CVRPInstance(
            depot=(0, 0),
            customers=(Customer(1, 1, 4), Customer(2, 2, 4)),
            num_vehicles=1,
            vehicle_capacity=5,
        )

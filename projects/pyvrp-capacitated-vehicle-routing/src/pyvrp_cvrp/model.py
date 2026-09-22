"""Small, testable CVRP model built on PyVRP's high-level API."""

from __future__ import annotations

from dataclasses import dataclass

import pyvrp


@dataclass(frozen=True)
class Customer:
    """A delivery customer with integer demand."""

    x: float
    y: float
    demand: int
    name: str = ""

    def __post_init__(self) -> None:
        if self.demand <= 0:
            raise ValueError("customer demand must be positive")


@dataclass(frozen=True)
class CVRPInstance:
    """Single-depot capacitated vehicle-routing instance."""

    depot: tuple[float, float]
    customers: tuple[Customer, ...]
    num_vehicles: int
    vehicle_capacity: int

    def __post_init__(self) -> None:
        if not self.customers:
            raise ValueError("at least one customer is required")
        if self.num_vehicles <= 0:
            raise ValueError("num_vehicles must be positive")
        if self.vehicle_capacity <= 0:
            raise ValueError("vehicle_capacity must be positive")
        if max(customer.demand for customer in self.customers) > self.vehicle_capacity:
            raise ValueError("a customer demand exceeds vehicle capacity")
        if self.total_demand > self.fleet_capacity:
            raise ValueError("total demand exceeds total fleet capacity")

    @property
    def total_demand(self) -> int:
        return sum(customer.demand for customer in self.customers)

    @property
    def fleet_capacity(self) -> int:
        return self.num_vehicles * self.vehicle_capacity


def manhattan_distance(a: tuple[float, float], b: tuple[float, float]) -> int:
    """Returns PyVRP-compatible integer Manhattan distance."""

    return int(round(abs(a[0] - b[0]) + abs(a[1] - b[1])))


def build_model(instance: CVRPInstance) -> pyvrp.Model:
    """Builds a PyVRP model for a validated CVRP instance."""

    model = pyvrp.Model()
    depot_location = model.add_location(*instance.depot, name="Depot location")
    model.add_depot(location=depot_location, name="Depot")
    model.add_vehicle_type(
        num_available=instance.num_vehicles,
        capacity=instance.vehicle_capacity,
    )

    for idx, customer in enumerate(instance.customers, start=1):
        location = model.add_location(
            customer.x,
            customer.y,
            name=f"Customer {idx} location",
        )
        model.add_client(
            location=location,
            delivery=customer.demand,
            name=customer.name or f"Customer {idx}",
        )

    for frm in model.locations:
        for to in model.locations:
            distance = manhattan_distance((frm.x, frm.y), (to.x, to.y))
            model.add_edge(frm, to, distance=distance)

    return model


def solve_instance(
    instance: CVRPInstance,
    *,
    max_iterations: int = 2_000,
    seed: int = 42,
) -> pyvrp.Result:
    """Solves a CVRP instance with deterministic iteration budget and seed."""

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    model = build_model(instance)
    return model.solve(
        stop=pyvrp.stop.MaxIterations(max_iterations),
        seed=seed,
        collect_stats=False,
        display=False,
    )


def demo_instance() -> CVRPInstance:
    """Returns a small warehouse-delivery CVRP used by the CLI and tests."""

    return CVRPInstance(
        depot=(50, 50),
        customers=(
            Customer(39, 76, 5, "Northwest"),
            Customer(64, 47, 3, "East"),
            Customer(73, 61, 7, "Northeast"),
            Customer(32, 39, 2, "Southwest"),
            Customer(45, 30, 6, "South"),
            Customer(58, 72, 4, "North"),
        ),
        num_vehicles=2,
        vehicle_capacity=15,
    )
